"""Dead-reckoning pose (world x, y, yaw) integration from body-frame forward/lateral
velocity and yaw-rate -- used ONLY inside planner.py's model-based rollouts (the real
executed pose in navigate_wam.py always comes straight from MPCResidualEnv's own
info["base_pos"]/info["yaw"], same as navigate.py/navigate_mpc.py already do; the learned
world model has no x/y/yaw in its own 33-dim state -- see wam/state.py's "Scoping
decisions" -- so a planner that wants to reason about a world-frame goal has to integrate
pose itself from the state components that ARE predicted).

Rotation convention matches mpc_residual_env.py's own body->world mapping used in
_local_heightmap (wx = x0 + F*cos(yaw) - L*sin(yaw), wy = y0 + F*sin(yaw) + L*cos(yaw),
F=forward, L=lateral) and yaw's own atan2(2*(qw*qz+qx*qy), 1-2*(qy^2+qz^2)) extraction --
confirmed directly from that file rather than assumed, so dead-reckoned pose stays
consistent with how the real simulator's ground-truth pose is defined.

All functions accept torch tensors with an arbitrary leading batch shape (last dim is the
scalar being integrated) so a CEM planner can roll out many candidate action sequences at
once.
"""
import torch


def integrate_pose(x, y, yaw, v_fwd, v_lat, yaw_rate, dt: float):
    """One Euler step of (x, y, yaw) given body-frame forward/lateral velocity and yaw-rate,
    all same-shaped tensors (typically (n_samples,)). Returns the updated (x, y, yaw)."""
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    dx = v_fwd * cos_yaw - v_lat * sin_yaw
    dy = v_fwd * sin_yaw + v_lat * cos_yaw
    x_next = x + dx * dt
    y_next = y + dy * dt
    yaw_next = wrap_to_pi(yaw + yaw_rate * dt)
    return x_next, y_next, yaw_next


def wrap_to_pi(angle):
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi
