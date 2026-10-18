"""Stage 2: use the learned world model to CHOOSE actions, not just predict their effect.

A CEM (cross-entropy method) shooting planner: to pick a velocity command, it samples many
candidate CONSTANT commands, rolls each one forward through the trained WorldModel for a
short horizon (dead-reckoning world pose alongside it -- see wam/pose.py), scores each
rollout by how close it ends up to the goal (plus a heading-alignment and a stability
penalty), keeps the best-scoring "elite" samples, and refits its sampling distribution
around them -- repeating a few iterations before committing to an action. This is the same
shooting-method idea MPC/MPPI/PETS use, except the "model" being shot through is the
learned one from Stage 1, not a hand-written dynamics model or a hand-written steering law
like navigate.py's compute_nav_command.

Candidates are a single constant command held across the whole planning horizon (not a
time-varying sequence) deliberately: Stage 1's training data resampled a target command
only every `hold_ticks` (0.5s), many macro-steps at a time -- a constant-action rollout
matches that distribution, where a freely time-varying per-macro-step sequence would ask
the model to extrapolate to action-switching rates it never saw in training.

Stage 3 adds an optional obstacle cost: if `plan()` is given a set of currently-known
occupied-cell centers (e.g. from a `slam_sim`-style lidar-built occupancy grid -- see
`navigate_wam_obstacles.py`), each rollout step's predicted (x, y) is scored against the
nearest occupied point, same vectorized shooting loop, no structural change to the method
itself -- it's just one more additive cost term alongside distance-to-goal and stability.

Stage 9 adds an optional SECOND obstacle term: `dynamic_obstacles=[(pos0, vel), ...]` is a
list of tracked points (e.g. `wam/obstacle_tracker.py`'s `DynamicObstacleTracker` estimates
of one or more moving obstacles -- Stage 10 generalized it from exactly one to however many
tracks are live) that each get velocity-EXTRAPOLATED forward to each rollout step's own
future time (`(h+1) * macro_dt` ahead of now) before being scored the same way -- unlike the
static `obstacle_points` term, which stays frozen at wherever it was sensed for the whole
horizon. Distance is taken to the NEAREST of the extrapolated obstacles at each step, the
same `cdist` + `amin` reduction the static term already uses, so multiple obstacles don't
double-count a penalty for being near just one of them.

Stage 11 adds an optional smoothness/hysteresis term: `w_smooth` penalizes a candidate
command for differing from the PREVIOUS plan()'s chosen action (`warm_start`), independent
of warm_start's existing role of just centering the first iteration's sampling distribution.
Without this, nothing in the cost function penalizes switching commands. This was tried as a
fix for the indecisive-lingering behavior Stage 10 observed with two dynamic obstacles, but
repeated multi-seed evals found no reliable net benefit (it fixed some seeds' dithering while
making others worse, noise-level in aggregate) and too high a weight (5.0) caused an actual
timeout by over-committing to a bad action -- see README Stage 11 for the full measured
result. Kept available (default 0.0, a no-op) as verified infrastructure, not adopted.

Pixel-rendering mini-plan, step 17 adds an optional CAMERA-based obstacle term: instead of
(or alongside) the lidar-built `obstacle_points` grid above, `plan()` accepts a `pixel_latent0`
(the onboard camera's current depth frame, already encoded via wam/pixel_encoder.py's
ConvAutoencoder) plus a `pixel_dynamics`/`pixel_cost_head` pair (wam/pixel_dynamics.py's
LatentDynamics and wam/pixel_cost.py's LatentCostHead from step 15, both frozen, trained on
step 16's turning-inclusive exploration data). Each rollout step reads a predicted
minimum-depth (distance to the nearest point the camera sees) straight off the PIXEL
dynamics' own rolled-forward latent -- the decoder is never called, exactly step 15's speed
fix -- and penalizes it with the same soft-margin/hard-collision shape the lidar term uses,
just in physical meters read from a camera instead of grid cells read from a lidar scan. This
runs as a SEPARATE batched rollout alongside the state-based model's own rollout (same
`samples` actions, same horizon, same n_samples -- just a second, independent latent being
advanced every step), not a replacement for it: the state model still owns pose/goal/
stability, since (unlike the lidar term) a bare min-depth number carries no (x, y) position
to dead-reckon a goal distance from -- see README's pixel-vs-state comparison for why this
split plays to each model's own strength rather than forcing one to do both jobs.

Needs torch (see repo's own venv note in requirements.txt).
"""
from dataclasses import dataclass

import torch

from .pose import integrate_pose
from .state import STATE_LAYOUT

_LAYOUT = dict(STATE_LAYOUT)
_V_FWD = _LAYOUT["v_body"].start + 0
_V_LAT = _LAYOUT["v_body"].start + 1
_YAW_RATE = _LAYOUT["w_body"].start + 2
_ROLL = _LAYOUT["roll"].start
_PITCH = _LAYOUT["pitch"].start
_HEIGHT = _LAYOUT["base_height"].start


@dataclass
class PlanResult:
    action: torch.Tensor          # (3,) best constant command found
    cost: float                   # its predicted cost
    pred_final_xy: tuple          # where the model predicts this command ends up


class CEMPlanner:
    def __init__(self, model, speed_range, lateral_range, yaw_rate_range,
                 horizon_steps: int = 10, macro_dt: float = 0.05,
                 n_samples: int = 200, n_elites: int = 20, n_iters: int = 3,
                 w_dist: float = 1.0, w_heading: float = 0.3, w_stability: float = 3.0,
                 w_obstacle: float = 40.0, w_collision: float = 200.0,
                 safety_margin: float = 0.5, collision_margin: float = 0.35,
                 w_smooth: float = 0.0,
                 pixel_dynamics=None, pixel_cost_head=None,
                 w_pixel_obstacle: float = 40.0, w_pixel_collision: float = 200.0,
                 pixel_safety_margin: float = 0.6, pixel_collision_margin: float = 0.35,
                 pixel_horizon_steps=None):
        self.model = model
        self.pixel_dynamics = pixel_dynamics
        self.pixel_cost_head = pixel_cost_head
        self.w_pixel_obstacle = w_pixel_obstacle
        self.w_pixel_collision = w_pixel_collision
        self.pixel_safety_margin = pixel_safety_margin
        self.pixel_collision_margin = pixel_collision_margin
        # Step 17's own train_pixel_cost.py rollout check (over turning-inclusive data) found
        # the cost head's predicted-latent correlation collapsing much faster than step 15's
        # single-scenario result did -- 0.80/0.53/0.35 at steps 1/2/3, near-zero by ~10 -- so
        # scoring every one of horizon_steps's steps with it would inject noisy/anti-correlated
        # cost at the later ones. Defaults to horizon_steps (score everything, step 15's
        # original behavior) but navigate_wam_pixel.py caps it lower based on that finding.
        self.pixel_horizon_steps = (pixel_horizon_steps if pixel_horizon_steps is not None
                                    else horizon_steps)
        self.lo = torch.tensor([speed_range[0], lateral_range[0], yaw_rate_range[0]])
        self.hi = torch.tensor([speed_range[1], lateral_range[1], yaw_rate_range[1]])
        self.horizon_steps = horizon_steps
        self.macro_dt = macro_dt
        self.n_samples = n_samples
        self.n_elites = n_elites
        self.n_iters = n_iters
        self.w_dist = w_dist
        self.w_heading = w_heading
        self.w_stability = w_stability
        self.w_obstacle = w_obstacle
        self.w_collision = w_collision
        self.safety_margin = safety_margin
        self.collision_margin = collision_margin
        self.w_smooth = w_smooth
        # Stability penalty is measured relative to the TRAINING distribution's own typical
        # standing state, not a hand-picked constant -- ties the penalty to what the model
        # actually learned from rather than a guessed nominal height/roll/pitch.
        self.nominal_height = float(model.state_mean[_HEIGHT])
        self.nominal_roll = float(model.state_mean[_ROLL])
        self.nominal_pitch = float(model.state_mean[_PITCH])

    @torch.no_grad()
    def plan(self, state, pose, goal_xy, warm_start=None, obstacle_points=None,
             dynamic_obstacles=None, dynamic_obstacle_radius: float = 0.0,
             pixel_latent0=None) -> PlanResult:
        """state: (STATE_DIM,) current real state. pose: (x, y, yaw) current real pose
        (ground truth -- only the planner's internal ROLLOUT dead-reckons with the model;
        the state it starts from is always the real, measured one). goal_xy: (gx, gy).
        warm_start: previous plan()'s chosen action, used to center the first CEM
        iteration's sampling distribution (cheap and usually close to the new answer, since
        goals move slowly relative to the replan rate). obstacle_points: optional (n_obs, 2)
        array/tensor of currently-known occupied-cell centers (Stage 3) -- None (default)
        recovers exactly Stage 2's goal-only behavior; treated as frozen for the whole
        horizon. dynamic_obstacles: optional list of (pos0, vel) pairs, each a 2-vector --
        Stage 9/10's velocity-extrapolated moving-obstacle term(s), re-evaluated at each
        rollout step's own future time instead of staying frozen; any number of them (empty
        or None both mean "no dynamic obstacles"). dynamic_obstacle_radius: a shared
        physical footprint radius, added onto safety_margin/collision_margin for THIS term
        only -- `obstacle_points`'s static safety-margin/collision-margin already gets a
        comparable allowance "for free" from the occupancy grid's sensing/clearing lag (a
        few recently-vacated cells near a moving obstacle stay marked occupied for a few
        scans), but a single extrapolated point has no such built-in slack, so collapsing a
        sensed cluster down to its centroid without this would silently shrink the avoided
        region from the real object's full width down to a bare point. pixel_latent0: optional
        (latent_dim,) encoded onboard-camera depth frame (step 17) -- None (default) recovers
        exactly the lidar-only behavior above; requires pixel_dynamics/pixel_cost_head to have
        been passed to the constructor, otherwise it's silently ignored (use_pixel stays False)."""
        state0 = torch.as_tensor(state, dtype=torch.float32)
        x0, y0, yaw0 = (torch.as_tensor(v, dtype=torch.float32) for v in pose)
        gx, gy = goal_xy
        obstacles = None
        if obstacle_points is not None and len(obstacle_points) > 0:
            obstacles = torch.as_tensor(obstacle_points, dtype=torch.float32)
        dyn_pos0 = dyn_vel = None
        if dynamic_obstacles:
            dyn_pos0 = torch.as_tensor(
                [d[0] for d in dynamic_obstacles], dtype=torch.float32)  # (n_dyn, 2)
            dyn_vel = torch.as_tensor(
                [d[1] for d in dynamic_obstacles], dtype=torch.float32)  # (n_dyn, 2)

        prev_action = (torch.as_tensor(warm_start, dtype=torch.float32)
                       if warm_start is not None else None)
        mean = prev_action if prev_action is not None else torch.zeros(3)
        std = (self.hi - self.lo) / 4.0

        use_pixel = (self.pixel_dynamics is not None and self.pixel_cost_head is not None
                     and pixel_latent0 is not None)
        z0 = torch.as_tensor(pixel_latent0, dtype=torch.float32) if use_pixel else None

        best_cost, best_xy = float("inf"), (x0.item(), y0.item())
        for _ in range(self.n_iters):
            samples = mean + std * torch.randn(self.n_samples, 3)
            samples = torch.max(torch.min(samples, self.hi), self.lo)

            state_b = state0.unsqueeze(0).expand(self.n_samples, -1).clone()
            x_b = x0.expand(self.n_samples).clone()
            y_b = y0.expand(self.n_samples).clone()
            yaw_b = yaw0.expand(self.n_samples).clone()
            cost = torch.zeros(self.n_samples)
            z_b = z0.unsqueeze(0).expand(self.n_samples, -1).clone() if use_pixel else None

            for h in range(self.horizon_steps):
                state_b = self.model(state_b, samples)
                x_b, y_b, yaw_b = integrate_pose(
                    x_b, y_b, yaw_b, state_b[:, _V_FWD], state_b[:, _V_LAT],
                    state_b[:, _YAW_RATE], self.macro_dt)
                roll, pitch, height = state_b[:, _ROLL], state_b[:, _PITCH], state_b[:, _HEIGHT]
                cost = cost + self.w_stability * (
                    (roll - self.nominal_roll) ** 2 + (pitch - self.nominal_pitch) ** 2
                    + (height - self.nominal_height) ** 2)

                if obstacles is not None:
                    nearest_dist = torch.cdist(
                        torch.stack([x_b, y_b], dim=1), obstacles).amin(dim=1)
                    soft = torch.clamp(self.safety_margin - nearest_dist, min=0.0) ** 2
                    hard = (nearest_dist < self.collision_margin).float()
                    cost = cost + self.w_obstacle * soft + self.w_collision * hard

                if dyn_pos0 is not None:
                    t_future = (h + 1) * self.macro_dt
                    dyn_now = dyn_pos0 + dyn_vel * t_future  # (n_dyn, 2)
                    dyn_dist = torch.cdist(
                        torch.stack([x_b, y_b], dim=1), dyn_now).amin(dim=1)
                    dyn_safety = self.safety_margin + dynamic_obstacle_radius
                    dyn_collision = self.collision_margin + dynamic_obstacle_radius
                    soft = torch.clamp(dyn_safety - dyn_dist, min=0.0) ** 2
                    hard = (dyn_dist < dyn_collision).float()
                    cost = cost + self.w_obstacle * soft + self.w_collision * hard

                if use_pixel and h < self.pixel_horizon_steps:
                    z_b = self.pixel_dynamics(z_b, samples)
                    pred_min_depth = self.pixel_cost_head(z_b).squeeze(1)
                    soft = torch.clamp(self.pixel_safety_margin - pred_min_depth, min=0.0) ** 2
                    hard = (pred_min_depth < self.pixel_collision_margin).float()
                    cost = cost + self.w_pixel_obstacle * soft + self.w_pixel_collision * hard

            dist = torch.hypot(gx - x_b, gy - y_b)
            heading_err = torch.atan2(gy - y_b, gx - x_b) - yaw_b
            heading_err = (heading_err + torch.pi) % (2 * torch.pi) - torch.pi
            cost = cost + self.w_dist * dist + self.w_heading * heading_err ** 2

            if prev_action is not None and self.w_smooth > 0:
                cost = cost + self.w_smooth * ((samples - prev_action) ** 2).sum(dim=1)

            elite_idx = torch.topk(-cost, self.n_elites).indices
            elites = samples[elite_idx]
            mean = elites.mean(0)
            std = elites.std(0).clamp_min(1e-3)

            iter_best = torch.argmin(cost)
            if cost[iter_best] < best_cost:
                best_cost = float(cost[iter_best])
                best_xy = (float(x_b[iter_best]), float(y_b[iter_best]))

        return PlanResult(action=mean, cost=best_cost, pred_final_xy=best_xy)
