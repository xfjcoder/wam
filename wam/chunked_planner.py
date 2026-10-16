"""Stage 6: a CEM planner that shoots through the chunked, history-conditioned world model
(wam/chunked_model.py) instead of the single-step MLP (wam/planner.py's CEMPlanner) --
same shooting-method structure (sample candidates, score, keep elites, refit, iterate),
but each CEM iteration costs ONE batched model call instead of `horizon_steps` sequential
ones: the chunked model predicts the whole chunk's states in one non-autoregressive
forward pass, for all candidates at once.

Candidates are still a single constant command held across the whole chunk (same
reasoning as wam/planner.py's own CEMPlanner -- matches how the model's own chunk_len
training windows were built from the real, mostly-constant-per-hold-period command data).

Unlike the single-step planner, this one needs a real, MAINTAINED history of the last
`history_len` (state, action) pairs, not just the current state -- see
ChunkedHistoryBuffer below and navigate_wam_chunked.py for how a caller keeps this
updated across replans.
"""
import torch

from .pose import integrate_pose
from .planner import PlanResult, _V_FWD, _V_LAT, _YAW_RATE, _ROLL, _PITCH, _HEIGHT


class ChunkedHistoryBuffer:
    """Rolling buffer of the last `history_len` REAL (state, action) pairs -- ground
    truth, never model-predicted (the planner always starts its imagined rollout from
    real history/state, same principle as the single-step planner always starting from
    the real current state, never its own past imagined rollout). Before `history_len`
    real transitions have happened yet (episode startup), pads with repeats of the first
    real state and a zero action -- a documented cold-start approximation (the model sees
    a few "nothing happened yet" steps at the very start of an episode, a regime it was
    never specifically trained on), not a hidden one.
    """

    def __init__(self, history_len: int, state_dim: int, action_dim: int):
        self.history_len = history_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._states = None
        self._actions = None

    def reset(self, initial_state):
        s0 = torch.as_tensor(initial_state, dtype=torch.float32)
        self._states = s0.unsqueeze(0).repeat(self.history_len, 1)
        self._actions = torch.zeros(self.history_len, self.action_dim)

    def push(self, state, action):
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        a = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
        self._states = torch.cat([self._states[1:], s], dim=0)
        self._actions = torch.cat([self._actions[1:], a], dim=0)

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions


class ChunkedCEMPlanner:
    def __init__(self, model, speed_range, lateral_range, yaw_rate_range,
                 macro_dt: float = 0.05, n_samples: int = 200, n_elites: int = 20,
                 n_iters: int = 3, w_dist: float = 1.0, w_heading: float = 0.3,
                 w_stability: float = 3.0, w_obstacle: float = 40.0, w_collision: float = 200.0,
                 safety_margin: float = 0.5, collision_margin: float = 0.35):
        self.model = model
        self.lo = torch.tensor([speed_range[0], lateral_range[0], yaw_rate_range[0]])
        self.hi = torch.tensor([speed_range[1], lateral_range[1], yaw_rate_range[1]])
        self.horizon_steps = model.chunk_len   # fixed by the checkpoint, not a free choice
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
        self.nominal_height = float(model.state_mean[_HEIGHT])
        self.nominal_roll = float(model.state_mean[_ROLL])
        self.nominal_pitch = float(model.state_mean[_PITCH])

    @torch.no_grad()
    def plan(self, history: ChunkedHistoryBuffer, pose, goal_xy, warm_start=None,
             obstacle_points=None) -> PlanResult:
        x0, y0, yaw0 = (torch.as_tensor(v, dtype=torch.float32) for v in pose)
        gx, gy = goal_xy
        obstacles = None
        if obstacle_points is not None and len(obstacle_points) > 0:
            obstacles = torch.as_tensor(obstacle_points, dtype=torch.float32)

        current_state = history.states[-1]
        hist_states, hist_actions = history.states, history.actions

        mean = (torch.as_tensor(warm_start, dtype=torch.float32) if warm_start is not None
                else torch.zeros(3))
        std = (self.hi - self.lo) / 4.0

        best_cost, best_xy = float("inf"), (x0.item(), y0.item())
        for _ in range(self.n_iters):
            samples = mean + std * torch.randn(self.n_samples, 3)
            samples = torch.max(torch.min(samples, self.hi), self.lo)

            hist_s_b = hist_states.unsqueeze(0).expand(self.n_samples, -1, -1)
            hist_a_b = hist_actions.unsqueeze(0).expand(self.n_samples, -1, -1)
            cur_s_b = current_state.unsqueeze(0).expand(self.n_samples, -1)
            fut_a_b = samples.unsqueeze(1).expand(-1, self.horizon_steps, -1)

            # ONE batched model call for the whole chunk, all n_samples candidates --
            # vs. wam/planner.py's CEMPlanner, which calls the single-step model
            # horizon_steps times (also batched over samples, but horizon_steps calls).
            pred_states = self.model(hist_s_b, hist_a_b, cur_s_b, fut_a_b)   # (B, H, S)

            x_b = x0.expand(self.n_samples).clone()
            y_b = y0.expand(self.n_samples).clone()
            yaw_b = yaw0.expand(self.n_samples).clone()
            cost = torch.zeros(self.n_samples)

            for h in range(self.horizon_steps):
                state_h = pred_states[:, h, :]
                x_b, y_b, yaw_b = integrate_pose(
                    x_b, y_b, yaw_b, state_h[:, _V_FWD], state_h[:, _V_LAT],
                    state_h[:, _YAW_RATE], self.macro_dt)
                roll, pitch, height = state_h[:, _ROLL], state_h[:, _PITCH], state_h[:, _HEIGHT]
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
