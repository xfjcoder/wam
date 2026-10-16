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
    all same-shaped tensors (typically (n_samples,)). Returns the updated (x, y, yaw).

    Freezes yaw at its START-of-step value for the whole dt, so it under/over-shoots whenever
    yaw_rate is nonzero across the step -- see integrate_pose_exact for a closed-form fix to
    exactly this error, and eval_pose_integration.py for how much it actually matters here."""
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    dx = v_fwd * cos_yaw - v_lat * sin_yaw
    dy = v_fwd * sin_yaw + v_lat * cos_yaw
    x_next = x + dx * dt
    y_next = y + dy * dt
    yaw_next = wrap_to_pi(yaw + yaw_rate * dt)
    return x_next, y_next, yaw_next


def integrate_pose_exact(x, y, yaw, v_fwd, v_lat, yaw_rate, dt: float, eps: float = 1e-4):
    """Closed-form unicycle-arc integration over one step: exact (not just first-order) for a
    body-frame v_fwd/v_lat/yaw_rate held truly constant across the whole dt, since it accounts
    for yaw continuously rotating during the step instead of freezing it at the start (what
    integrate_pose does). Derived by integrating the body->world velocity rotation in closed
    form: dx = integral_0^dt [v_fwd*cos(yaw+yaw_rate*s) - v_lat*sin(yaw+yaw_rate*s)] ds, etc.

    Falls back to the ordinary Euler step when |yaw_rate * dt| is too small for the closed
    form's division by yaw_rate to be numerically safe -- the two formulas agree in that limit
    (no discontinuity at the switch)."""
    small = (yaw_rate.abs() * dt) < eps
    yaw_end = yaw + yaw_rate * dt
    w_safe = torch.where(small, torch.ones_like(yaw_rate), yaw_rate)

    sin_yaw, cos_yaw = torch.sin(yaw), torch.cos(yaw)
    sin_end, cos_end = torch.sin(yaw_end), torch.cos(yaw_end)
    dx_exact = (v_fwd / w_safe) * (sin_end - sin_yaw) + (v_lat / w_safe) * (cos_end - cos_yaw)
    dy_exact = (v_fwd / w_safe) * (cos_yaw - cos_end) + (v_lat / w_safe) * (sin_end - sin_yaw)

    dx_euler = v_fwd * cos_yaw * dt - v_lat * sin_yaw * dt
    dy_euler = v_fwd * sin_yaw * dt + v_lat * cos_yaw * dt

    x_next = x + torch.where(small, dx_euler, dx_exact)
    y_next = y + torch.where(small, dy_euler, dy_exact)
    return x_next, y_next, wrap_to_pi(yaw_end)


def wrap_to_pi(angle):
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi
