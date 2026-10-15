"""The reduced 33-dim physical state this world model predicts, sliced out of
`MPCResidualEnv`'s own 47-dim RL observation (`obs = [v_body(3), w_body(3), roll, pitch,
qpos(12), qvel(12), mpc_tau(12), cmd(3)]`) plus `info["base_height"]` -- see README's
"Scoping decisions" for why `mpc_tau` and `cmd` are excluded (control-internal / the action
itself, not world state to predict).

Kept in its own module (not duplicated in collect_data.py / eval_rollout.py) so the exact
same slicing is used everywhere a state vector is built or interpreted.
"""
import numpy as np

STATE_DIM = 33
ACTION_DIM = 3   # (target_speed, target_lateral, target_yaw_rate)

# (name, slice) pairs describing the 33-dim state vector, in order -- used for per-component
# error reporting and plotting, not by the model itself (which just sees a flat vector).
STATE_LAYOUT = [
    ("v_body", slice(0, 3)),
    ("w_body", slice(3, 6)),
    ("roll", slice(6, 7)),
    ("pitch", slice(7, 8)),
    ("base_height", slice(8, 9)),
    ("qpos", slice(9, 21)),
    ("qvel", slice(21, 33)),
]


def extract_state(obs: np.ndarray, info: dict) -> np.ndarray:
    """obs, info: whatever MPCResidualEnv.step()/reset() returned for this tick."""
    v_body = obs[0:3]
    w_body = obs[3:6]
    roll_pitch = obs[6:8]
    qpos12 = obs[8:20]
    qvel12 = obs[20:32]
    base_height = np.array([info["base_height"]], dtype=np.float32)
    return np.concatenate([v_body, w_body, roll_pitch, base_height, qpos12, qvel12]).astype(np.float32)


def extract_action(target_speed: float, target_lateral: float, target_yaw_rate: float) -> np.ndarray:
    return np.array([target_speed, target_lateral, target_yaw_rate], dtype=np.float32)
