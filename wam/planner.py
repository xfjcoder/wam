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
                 safety_margin: float = 0.5, collision_margin: float = 0.35):
        self.model = model
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
        # Stability penalty is measured relative to the TRAINING distribution's own typical
        # standing state, not a hand-picked constant -- ties the penalty to what the model
        # actually learned from rather than a guessed nominal height/roll/pitch.
        self.nominal_height = float(model.state_mean[_HEIGHT])
        self.nominal_roll = float(model.state_mean[_ROLL])
        self.nominal_pitch = float(model.state_mean[_PITCH])

    @torch.no_grad()
    def plan(self, state, pose, goal_xy, warm_start=None, obstacle_points=None) -> PlanResult:
        """state: (STATE_DIM,) current real state. pose: (x, y, yaw) current real pose
        (ground truth -- only the planner's internal ROLLOUT dead-reckons with the model;
        the state it starts from is always the real, measured one). goal_xy: (gx, gy).
        warm_start: previous plan()'s chosen action, used to center the first CEM
        iteration's sampling distribution (cheap and usually close to the new answer, since
        goals move slowly relative to the replan rate). obstacle_points: optional (n_obs, 2)
        array/tensor of currently-known occupied-cell centers (Stage 3) -- None (default)
        recovers exactly Stage 2's goal-only behavior."""
        state0 = torch.as_tensor(state, dtype=torch.float32)
        x0, y0, yaw0 = (torch.as_tensor(v, dtype=torch.float32) for v in pose)
        gx, gy = goal_xy
        obstacles = None
        if obstacle_points is not None and len(obstacle_points) > 0:
            obstacles = torch.as_tensor(obstacle_points, dtype=torch.float32)

        mean = (torch.as_tensor(warm_start, dtype=torch.float32) if warm_start is not None
                else torch.zeros(3))
        std = (self.hi - self.lo) / 4.0

        best_cost, best_xy = float("inf"), (x0.item(), y0.item())
        for _ in range(self.n_iters):
            samples = mean + std * torch.randn(self.n_samples, 3)
            samples = torch.max(torch.min(samples, self.hi), self.lo)

            state_b = state0.unsqueeze(0).expand(self.n_samples, -1).clone()
            x_b = x0.expand(self.n_samples).clone()
            y_b = y0.expand(self.n_samples).clone()
            yaw_b = yaw0.expand(self.n_samples).clone()
            cost = torch.zeros(self.n_samples)

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

            dist = torch.hypot(gx - x_b, gy - y_b)
            heading_err = torch.atan2(gy - y_b, gx - x_b) - yaw_b
            heading_err = (heading_err + torch.pi) % (2 * torch.pi) - torch.pi
            cost = cost + self.w_dist * dist + self.w_heading * heading_err ** 2

            elite_idx = torch.topk(-cost, self.n_elites).indices
            elites = samples[elite_idx]
            mean = elites.mean(0)
            std = elites.std(0).clamp_min(1e-3)

            iter_best = torch.argmin(cost)
            if cost[iter_best] < best_cost:
                best_cost = float(cost[iter_best])
                best_xy = (float(x_b[iter_best]), float(y_b[iter_best]))

        return PlanResult(action=mean, cost=best_cost, pred_final_xy=best_xy)
