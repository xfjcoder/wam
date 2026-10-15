"""Stage 1 verification: does the learned world model's OPEN-LOOP rollout actually track
the real simulator's, not just predict one step ahead well? Collects a fresh held-out
episode from the real Go2 (reusing collect_data.py's own collect_episode() unmodified, same
procedure training/validation data was collected with), then autoregressively rolls the
trained model forward from the SAME starting state through the SAME action sequence --
feeding the model's own predictions back in as the next state, never peeking at ground
truth along the way -- and compares the two trajectories both numerically (per-component
RMSE) and visually (a time-series plot, true vs. predicted, for the quantities that matter
most: base height, roll/pitch, forward velocity, yaw rate).

Usage (needs go1_rl_walk/mpc's own venv for mujoco/casadi/pinocchio/torch):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_rollout.py \
        --checkpoint checkpoints/world_model.pt --record media/rollout_comparison.png
"""
import argparse
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

GO1_RL_WALK = Path(__file__).resolve().parent.parent / "go1_rl_walk"
sys.path.insert(0, str(GO1_RL_WALK / "mpc"))
sys.path.insert(0, str(GO1_RL_WALK))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mpc_residual_env import MPCResidualEnv                                # noqa: E402
from collect_data import collect_episode                                   # noqa: E402
from wam.model import WorldModel                                           # noqa: E402
from wam.state import STATE_DIM, ACTION_DIM, STATE_LAYOUT                   # noqa: E402

# A handful of state components worth actually looking at, by (component_name, index_within_
# component, display label) -- not every one of the 33 dims, most of which (individual joint
# angles) aren't individually interesting to eyeball.
PLOT_DIMS = [
    ("base_height", 0, "base height (m)"),
    ("roll", 0, "roll (rad)"),
    ("pitch", 0, "pitch (rad)"),
    ("v_body", 0, "forward velocity (m/s)"),
    ("w_body", 2, "yaw rate (rad/s)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/world_model.pt")
    ap.add_argument("--episode-seconds", type=float, default=10.0)
    ap.add_argument("--hold-ticks", type=int, default=100)
    ap.add_argument("--macro-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345, help="distinct from training/val seeds")
    ap.add_argument("--record", type=str, default="media/rollout_comparison.png")
    ap.add_argument("--terrain-amplitude-max", type=float, default=0.0,
                     help="m peak-to-peak random bumps for the held-out episode -- see "
                          "collect_data.py's own flag of the same name.")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WorldModel(STATE_DIM, ACTION_DIM, hidden=ckpt["hidden"], n_hidden=ckpt["n_hidden"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.5f}).")

    rng = np.random.default_rng(args.seed)
    terrain_range = (0.0, args.terrain_amplitude_max) if args.terrain_amplitude_max > 0 else None
    env = MPCResidualEnv(episode_seconds=args.episode_seconds, seed=args.seed,
                          terrain_amplitude_range=terrain_range)
    states, actions, next_states = collect_episode(
        env, rng, args.episode_seconds, args.hold_ticks, args.macro_k)
    true_traj = np.concatenate([states[0:1], next_states], axis=0)   # (T+1, STATE_DIM)
    n_steps = len(actions)
    print(f"Held-out episode: {n_steps} macro-steps "
          f"({n_steps * args.macro_k / 200.0:.1f}s simulated).")

    pred_traj = np.zeros_like(true_traj)
    pred_traj[0] = true_traj[0]
    with torch.no_grad():
        state_t = torch.from_numpy(true_traj[0:1]).float()
        for t in range(n_steps):
            action_t = torch.from_numpy(actions[t:t + 1]).float()
            state_t = model(state_t, action_t)   # feed the model's OWN prediction back in
            pred_traj[t + 1] = state_t.numpy()[0]

    rmse_per_dim = np.sqrt(np.mean((pred_traj - true_traj) ** 2, axis=0))
    print("Per-component open-loop rollout RMSE:")
    for name, sl in STATE_LAYOUT:
        vals = rmse_per_dim[sl]
        print(f"  {name:12s}: {', '.join(f'{v:.4f}' for v in vals)}")
    overall_rmse = float(np.sqrt(np.mean((pred_traj - true_traj) ** 2)))
    print(f"Overall RMSE across all {STATE_DIM} state dims: {overall_rmse:.4f}")

    layout = dict(STATE_LAYOUT)
    t_axis = np.arange(len(true_traj)) * args.macro_k / 200.0
    fig, axes = plt.subplots(len(PLOT_DIMS), 1, figsize=(8, 2.2 * len(PLOT_DIMS)), sharex=True)
    for ax, (comp, idx, label) in zip(axes, PLOT_DIMS):
        col = layout[comp].start + idx
        ax.plot(t_axis, true_traj[:, col], color="#2a9d4a", label="true (real MuJoCo)")
        ax.plot(t_axis, pred_traj[:, col], color="#8844cc", linestyle="--",
                label="predicted (learned model, open-loop)")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"World model open-loop rollout vs. real simulator "
                 f"(overall RMSE={overall_rmse:.4f})")
    fig.tight_layout()

    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.record, dpi=120)
    print(f"Saved comparison plot to {args.record}")


if __name__ == "__main__":
    main()
