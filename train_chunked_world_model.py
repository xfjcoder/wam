"""Stage 6: train the history-conditioned, action-chunked world model (wam/chunked_model.py)
on the SAME rollout dataset collect_data.py produces -- no new data collection needed, just
a different windowing of the same (state, action, next_state) transitions: each training
example is `history_len` past transitions + the current state + `chunk_len` future actions,
predicting all `chunk_len` future states in one non-autoregressive forward pass.

Usage (needs go1_rl_walk/mpc's own venv for torch):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 train_chunked_world_model.py \
        --data data/rollouts.npz --checkpoint checkpoints/world_model_chunked.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wam.chunked_model import ChunkedWorldModel                             # noqa: E402
from wam.state import STATE_DIM, ACTION_DIM                                  # noqa: E402
from train_world_model import build_window_starts                           # noqa: E402


def build_chunk_batch(states, actions, next_states, starts, history_len, chunk_len):
    """starts: (B,) window-start indices, each with history_len+chunk_len valid
    consecutive same-episode transitions ahead of it (see build_window_starts)."""
    hist_idx = starts[:, None] + np.arange(history_len)[None, :]        # (B, K)
    fut_idx = starts[:, None] + history_len + np.arange(chunk_len)[None, :]  # (B, H)
    hist_states = states[hist_idx]                                      # (B, K, S)
    hist_actions = actions[hist_idx]                                    # (B, K, A)
    current_state = next_states[starts + history_len - 1]                # (B, S)
    future_actions = actions[fut_idx]                                   # (B, H, A)
    future_states = next_states[fut_idx]                                # (B, H, S)
    return hist_states, hist_actions, current_state, future_actions, future_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/rollouts.npz")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/world_model_chunked.pt")
    ap.add_argument("--history-len", type=int, default=4)
    ap.add_argument("--chunk-len", type=int, default=10,
                     help="matches CEMPlanner's default horizon_steps so a trained "
                          "checkpoint can be dropped straight into the planner later")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    data = np.load(args.data)
    train_s, train_a = torch.from_numpy(data["train_states"]), torch.from_numpy(data["train_actions"])
    train_ns, train_eid = torch.from_numpy(data["train_next_states"]), data["train_episode_ids"]
    val_s, val_a = torch.from_numpy(data["val_states"]), torch.from_numpy(data["val_actions"])
    val_ns, val_eid = torch.from_numpy(data["val_next_states"]), data["val_episode_ids"]

    K, H = args.history_len, args.chunk_len
    train_starts = build_window_starts(train_eid, K + H)
    val_starts = build_window_starts(val_eid, K + H)
    print(f"Train: {len(train_s)} transitions, {len(train_starts)} chunk windows "
          f"(history={K}, chunk={H}). Val: {len(val_s)} transitions, {len(val_starts)} windows.")

    model = ChunkedWorldModel(STATE_DIM, ACTION_DIM, K, H, d_model=args.d_model,
                               n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff)

    train_starts_t = torch.from_numpy(train_starts)
    val_starts_t = torch.from_numpy(val_starts)

    # Fit normalization from a large random sample of training windows (cheap, no need
    # for every window).
    fit_idx = train_starts_t[torch.randperm(len(train_starts_t))[:min(20000, len(train_starts_t))]]
    hs, ha, cs, fa, fs = build_chunk_batch(train_s, train_a, train_ns, fit_idx.numpy(), K, H)
    model.fit_normalization(hs, ha, cs, fa, fs)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        perm = train_starts_t[torch.randperm(len(train_starts_t))]
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, len(perm), args.batch_size):
            batch_starts = perm[i:i + args.batch_size].numpy()
            hist_s, hist_a, cur_s, fut_a, fut_s = build_chunk_batch(
                train_s, train_a, train_ns, batch_starts, K, H)
            target_delta = fut_s - cur_s.unsqueeze(1)
            target_delta_n = (target_delta - model.chunk_delta_mean) / model.chunk_delta_std
            pred_delta_n = model.predict_chunk_normalized(hist_s, hist_a, cur_s, fut_a)
            loss = loss_fn(pred_delta_n, target_delta_n)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        epoch_loss /= max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            idx = val_starts_t[torch.randint(0, len(val_starts_t), (min(1024, len(val_starts_t)),))]
            hist_s, hist_a, cur_s, fut_a, fut_s = build_chunk_batch(
                val_s, val_a, val_ns, idx.numpy(), K, H)
            pred_fut_s = model(hist_s, hist_a, cur_s, fut_a)
            val_mae = (pred_fut_s - fut_s).abs().mean().item()
            val_rmse = torch.sqrt(((pred_fut_s - fut_s) ** 2).mean()).item()

        if val_rmse < best_val:
            best_val = val_rmse
            Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "history_len": K, "chunk_len": H,
                        "d_model": args.d_model, "n_heads": args.n_heads,
                        "n_layers": args.n_layers, "d_ff": args.d_ff,
                        "epoch": epoch, "val_rmse": val_rmse}, args.checkpoint)

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}: train_loss={epoch_loss:.5f} "
                  f"val_chunk_MAE={val_mae:.5f} val_chunk_RMSE={val_rmse:.5f}")

    print(f"Best val_chunk_RMSE={best_val:.5f}. Saved best checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
