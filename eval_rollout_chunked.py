"""Stage 6 verification: does the chunked model's open-loop rollout track the real
simulator at least as well as the original single-step MLP model (wam/model.py)? Same
discipline as every other checkpoint comparison in this project (see
[[go1-review-before-trusting]]/eval_terrain.py): multiple held-out seeds, paired (same
seed/action-sequence for both models), never a single episode.

The chunked model is non-autoregressive WITHIN a chunk (one forward call predicts all
`chunk_len` future states at once) but still autoregressive ACROSS chunks: each new
chunk's history/current-state context is built from the model's OWN previous predictions,
never peeking at ground truth beyond the first `history_len` transitions needed to seed
the very first chunk -- the same "never peeks at ground truth mid-rollout" discipline
eval_rollout.py's own single-step autoregressive rollout already follows.

Usage:
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_rollout_chunked.py \
        --chunked-checkpoint checkpoints/world_model_chunked.pt \
        --mlp-checkpoint checkpoints/world_model.pt --seeds 6
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

GO1_RL_WALK = Path(__file__).resolve().parent.parent / "go1_rl_walk"
sys.path.insert(0, str(GO1_RL_WALK / "mpc"))
sys.path.insert(0, str(GO1_RL_WALK))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mpc_residual_env import MPCResidualEnv                                # noqa: E402
from collect_data import collect_episode                                   # noqa: E402
from wam.model import WorldModel                                           # noqa: E402
from wam.chunked_model import ChunkedWorldModel                            # noqa: E402
from wam.state import STATE_DIM, ACTION_DIM                                 # noqa: E402


def load_chunked(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ChunkedWorldModel(STATE_DIM, ACTION_DIM, ckpt["history_len"], ckpt["chunk_len"],
                              d_model=ckpt["d_model"], n_heads=ckpt["n_heads"],
                              n_layers=ckpt["n_layers"], d_ff=ckpt["d_ff"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_mlp(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = WorldModel(STATE_DIM, ACTION_DIM, hidden=ckpt["hidden"], n_hidden=ckpt["n_hidden"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def rollout_mlp(model, states, actions, next_states):
    true_traj = np.concatenate([states[0:1], next_states], axis=0)
    pred_traj = np.zeros_like(true_traj)
    pred_traj[0] = true_traj[0]
    with torch.no_grad():
        state_t = torch.from_numpy(true_traj[0:1]).float()
        for t in range(len(actions)):
            action_t = torch.from_numpy(actions[t:t + 1]).float()
            state_t = model(state_t, action_t)
            pred_traj[t + 1] = state_t.numpy()[0]
    return true_traj, pred_traj


def rollout_chunked(model, states, actions, next_states):
    K, H = model.history_len, model.chunk_len
    n_steps = len(actions)
    n_chunks = (n_steps - K) // H
    used_steps = K + n_chunks * H
    true_traj = np.concatenate([states[0:1], next_states], axis=0)[:used_steps + 1]

    pred_traj = np.zeros_like(true_traj)
    pred_traj[:K + 1] = true_traj[:K + 1]   # ground-truth seed (same role as eval_rollout.py's true_traj[0])

    with torch.no_grad():
        hist_states = torch.from_numpy(states[0:K]).float().unsqueeze(0)
        hist_actions = torch.from_numpy(actions[0:K]).float().unsqueeze(0)
        current_state = torch.from_numpy(next_states[K - 1:K]).float()

        for c in range(n_chunks):
            fut_a = torch.from_numpy(actions[K + c * H: K + (c + 1) * H]).float().unsqueeze(0)
            pred_future = model(hist_states, hist_actions, current_state, fut_a)[0]   # (H, S)
            pred_traj[K + c * H + 1: K + (c + 1) * H + 1] = pred_future.numpy()

            states_in_chunk = torch.cat([current_state, pred_future[:-1]], dim=0)   # (H, S)
            hist_states = states_in_chunk[-K:].unsqueeze(0)
            hist_actions = fut_a[:, -K:, :]
            current_state = pred_future[-1:, :]

    return true_traj, pred_traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunked-checkpoint", type=str, default="checkpoints/world_model_chunked.pt")
    ap.add_argument("--mlp-checkpoint", type=str, default="checkpoints/world_model.pt")
    ap.add_argument("--episode-seconds", type=float, default=10.0)
    ap.add_argument("--hold-ticks", type=int, default=100)
    ap.add_argument("--macro-k", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--base-seed", type=int, default=20000, help="distinct from train/val/eval_rollout.py's own seeds")
    args = ap.parse_args()

    chunked = load_chunked(args.chunked_checkpoint)
    mlp = load_mlp(args.mlp_checkpoint)

    chunked_rmses, mlp_rmses = [], []
    for i in range(args.seeds):
        seed = args.base_seed + i
        rng = np.random.default_rng(seed)
        env = MPCResidualEnv(episode_seconds=args.episode_seconds, seed=seed)
        states, actions, next_states = collect_episode(
            env, rng, args.episode_seconds, args.hold_ticks, args.macro_k)

        true_c, pred_c = rollout_chunked(chunked, states, actions, next_states)
        rmse_c = float(np.sqrt(np.mean((pred_c - true_c) ** 2)))
        true_m, pred_m = rollout_mlp(mlp, states, actions, next_states)
        rmse_m = float(np.sqrt(np.mean((pred_m - true_m) ** 2)))

        chunked_rmses.append(rmse_c)
        mlp_rmses.append(rmse_m)
        print(f"seed={seed}: chunked RMSE={rmse_c:.4f} ({len(true_c)-1} steps)  "
              f"mlp RMSE={rmse_m:.4f} ({len(true_m)-1} steps)")

    chunked_rmses, mlp_rmses = np.array(chunked_rmses), np.array(mlp_rmses)
    print()
    print(f"chunked model: mean RMSE={chunked_rmses.mean():.4f}  std={chunked_rmses.std():.4f}")
    print(f"mlp model:     mean RMSE={mlp_rmses.mean():.4f}  std={mlp_rmses.std():.4f}")


if __name__ == "__main__":
    main()
