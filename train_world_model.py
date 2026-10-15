"""Stage 1: train the MLP world model (wam/model.py) on the rollout dataset
collect_data.py produced -- supervised regression, (state, action) -> next_state, trained
as a normalized delta target (see wam/model.py's own docstring for why).

Trains on a COMBINATION of two losses, not just one-step prediction:
  - one-step: predict a single transition's delta (the easy, stable-gradient signal).
  - H-step rollout: unroll the model H steps autoregressively (feeding its OWN predictions
    back in as the next state, never peeking at ground truth mid-rollout) and compare
    against the true trajectory at every one of those H steps.
A model trained on one-step loss alone looks great on one-step metrics but its errors
compound catastrophically under the autoregressive rollout actually used at eval/planning
time -- confirmed directly here (see HISTORY/commit message for the exact blowup: accurate
to ~8s, then exponential divergence by 10s). The H-step rollout loss directly penalizes
that compounding, the standard fix in model-based RL (PETS/MBPO/Dreamer all train their
dynamics models this way, not on one-step error alone).

Usage (needs go1_rl_walk/mpc's own venv for torch):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 train_world_model.py \
        --data data/rollouts.npz --checkpoint checkpoints/world_model.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wam.model import WorldModel                                           # noqa: E402
from wam.state import STATE_DIM, ACTION_DIM                                 # noqa: E402


def build_window_starts(episode_ids: np.ndarray, horizon: int) -> np.ndarray:
    """Start indices `i` such that transitions i..i+horizon-1 all belong to the same
    episode -- i.e. a horizon-step autoregressive rollout from `i` never crosses into a
    different episode's (unrelated) trajectory."""
    starts = []
    n = len(episode_ids)
    i = 0
    while i < n:
        j = i
        while j < n and episode_ids[j] == episode_ids[i]:
            j += 1
        last_valid_start = j - horizon
        if last_valid_start >= i:
            starts.extend(range(i, last_valid_start + 1))
        i = j
    return np.array(starts, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/rollouts.npz")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/world_model.pt")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n-hidden", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--rollout-horizon", type=int, default=20,
                     help="steps unrolled autoregressively per rollout-loss batch (20 "
                          "steps * 50ms/step = 1.0s)")
    ap.add_argument("--rollout-batches-per-epoch", type=int, default=20)
    ap.add_argument("--rollout-loss-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    data = np.load(args.data)
    train_s = torch.from_numpy(data["train_states"])
    train_a = torch.from_numpy(data["train_actions"])
    train_ns = torch.from_numpy(data["train_next_states"])
    train_eid = data["train_episode_ids"]
    val_s = torch.from_numpy(data["val_states"])
    val_a = torch.from_numpy(data["val_actions"])
    val_ns = torch.from_numpy(data["val_next_states"])
    val_eid = data["val_episode_ids"]

    assert train_s.shape[1] == STATE_DIM and train_a.shape[1] == ACTION_DIM
    print(f"Train: {len(train_s)} transitions. Val: {len(val_s)} transitions.")

    H = args.rollout_horizon
    train_starts = build_window_starts(train_eid, H)
    val_starts = build_window_starts(val_eid, H)
    print(f"Rollout windows (horizon={H}): {len(train_starts)} train, {len(val_starts)} val.")

    model = WorldModel(STATE_DIM, ACTION_DIM, hidden=args.hidden, n_hidden=args.n_hidden)
    model.fit_normalization(train_s, train_a, train_ns - train_s)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    def normalized_delta_target(s, ns):
        delta = ns - s
        return (delta - model.delta_mean) / model.delta_std

    train_starts_tensor = torch.from_numpy(train_starts)
    val_starts_tensor = torch.from_numpy(val_starts)

    n_train = len(train_s)
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()

        # --- one-step loss pass (cheap, stable gradient signal) ---
        perm = torch.randperm(n_train)
        one_step_loss = 0.0
        for i in range(0, n_train, args.batch_size):
            idx = perm[i:i + args.batch_size]
            s, a, ns = train_s[idx], train_a[idx], train_ns[idx]
            pred_delta_n = model.predict_delta_normalized(s, a)
            target_delta_n = normalized_delta_target(s, ns)
            loss = loss_fn(pred_delta_n, target_delta_n)
            opt.zero_grad()
            loss.backward()
            opt.step()
            one_step_loss += loss.item() * len(idx)
        one_step_loss /= n_train

        # --- H-step rollout loss pass (penalizes autoregressive compounding directly) ---
        roll_loss_total = 0.0
        for _ in range(args.rollout_batches_per_epoch):
            idx = train_starts_tensor[torch.randint(0, len(train_starts_tensor), (args.batch_size,))]
            state = train_s[idx]
            loss = 0.0
            for h in range(H):
                state = model(state, train_a[idx + h])
                target_h = train_ns[idx + h]
                loss = loss + loss_fn(state / model.state_std, target_h / model.state_std)
            loss = (loss / H) * args.rollout_loss_weight
            opt.zero_grad()
            loss.backward()
            opt.step()
            roll_loss_total += loss.item()
        roll_loss_total /= args.rollout_batches_per_epoch

        model.eval()
        with torch.no_grad():
            val_pred_delta_n = model.predict_delta_normalized(val_s, val_a)
            val_target_delta_n = normalized_delta_target(val_s, val_ns)
            val_loss = loss_fn(val_pred_delta_n, val_target_delta_n).item()
            val_pred_next = model(val_s, val_a)
            val_mae = (val_pred_next - val_ns).abs().mean().item()

            # H-step rollout error on held-out validation windows -- the metric that
            # actually matters (what eval_rollout.py checks at full episode length).
            idx = val_starts_tensor[torch.randint(0, len(val_starts_tensor),
                                                   (min(256, len(val_starts_tensor)),))]
            state = val_s[idx]
            roll_err = 0.0
            for h in range(H):
                state = model(state, val_a[idx + h])
                roll_err += (state - val_ns[idx + h]).abs().mean().item()
            roll_err /= H

        if roll_err < best_val:
            best_val = roll_err
            Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "hidden": args.hidden,
                        "n_hidden": args.n_hidden, "epoch": epoch, "val_loss": val_loss,
                        "val_rollout_mae": roll_err, "rollout_horizon": H},
                       args.checkpoint)

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}: one_step_loss={one_step_loss:.5f} "
                  f"rollout_train_loss={roll_loss_total:.5f} val_one_step_loss={val_loss:.5f} "
                  f"val_next_state_MAE={val_mae:.5f} val_{H}step_rollout_MAE={roll_err:.5f}")

    print(f"Best val_{H}step_rollout_MAE={best_val:.5f}. Saved best checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
