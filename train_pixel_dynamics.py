"""Pixel-rendering mini-plan, step 6: train wam/pixel_dynamics.py's LatentDynamics on top of
the FROZEN encoder from step 5 (checkpoints/pixel_encoder.pt), and check whether the latent
space it learned for reconstruction ALSO supports one-step-ahead prediction -- the direct
pixel analogue of train_world_model.py's original (state, action) -> next_state regression.

Pixel frame i+1 is consecutive transition i's "next frame" whenever both belong to the same
episode (collect_data.py captures one frame per transition at the same macro-step instant as
state_t, back-to-back within an episode) -- no new data collection needed, just pairing up
indices that don't cross an episode boundary (train_world_model.py's build_window_starts does
the same thing for horizon-H windows; this is the horizon=1 case).

Two checks, not just one, mirroring this project's own Stage 1 lesson (one-step metrics alone
looked great and still diverged catastrophically under autoregressive rollout):
  1. One-step latent-delta MSE vs. a trivial "predict no change" (zero-delta) baseline.
  2. A short (5-step) autoregressive latent rollout, decoded back to images every step and
     compared against the real frames -- does it stay visually sane, or drift/collapse.

Usage (needs go1_rl_walk/mpc's own venv for torch):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 train_pixel_dynamics.py \
        --data data/rollouts_pixels.npz --encoder-checkpoint checkpoints/pixel_encoder.pt \
        --checkpoint checkpoints/pixel_dynamics.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wam.pixel_encoder import ConvAutoencoder  # noqa: E402
from wam.pixel_dynamics import LatentDynamics  # noqa: E402


def to_tensor_01(pixels_u8: np.ndarray) -> torch.Tensor:
    t = torch.as_tensor(pixels_u8, dtype=torch.float32) / 255.0
    return t.permute(0, 3, 1, 2).contiguous()


def consecutive_pairs(episode_ids: np.ndarray) -> np.ndarray:
    """Indices i such that (i, i+1) are consecutive transitions of the SAME episode --
    i.e. frame[i+1] really is frame[i]'s next frame, not some other episode's first frame."""
    same_episode = episode_ids[:-1] == episode_ids[1:]
    return np.nonzero(same_episode)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/rollouts_pixels.npz")
    ap.add_argument("--encoder-checkpoint", type=str, default="checkpoints/pixel_encoder.pt")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/pixel_dynamics.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout-steps", type=int, default=5)
    ap.add_argument("--rollout-grid", type=str,
                     default="media/pixel_dynamics_rollout.png",
                     help="saved ground-truth-vs-rollout decoded frame grid")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    data = np.load(args.data)
    resolution = int(data["pixel_resolution"])

    enc_ckpt = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=False)
    encoder = ConvAutoencoder(latent_dim=enc_ckpt["latent_dim"], resolution=enc_ckpt["resolution"])
    encoder.load_state_dict(enc_ckpt["model_state"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    def latents_of(pixels_u8):
        with torch.no_grad():
            return encoder.encode(to_tensor_01(pixels_u8))

    train_latent = latents_of(data["train_pixels_rgb"])
    val_latent = latents_of(data["val_pixels_rgb"])
    train_action = torch.as_tensor(data["train_actions"], dtype=torch.float32)
    val_action = torch.as_tensor(data["val_actions"], dtype=torch.float32)

    train_pairs = consecutive_pairs(data["train_episode_ids"])
    val_pairs = consecutive_pairs(data["val_episode_ids"])
    print(f"train pairs: {len(train_pairs)}, val pairs: {len(val_pairs)}, latent_dim: "
          f"{enc_ckpt['latent_dim']}")

    lat_t = train_latent[train_pairs]
    act_t = train_action[train_pairs]
    lat_next = train_latent[train_pairs + 1]
    delta_t = lat_next - lat_t

    val_lat_t = val_latent[val_pairs]
    val_act_t = val_action[val_pairs]
    val_lat_next = val_latent[val_pairs + 1]

    model = LatentDynamics(latent_dim=enc_ckpt["latent_dim"], action_dim=train_action.shape[1])
    model.fit_normalization(lat_t, act_t, delta_t)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Trivial baseline: predict NO CHANGE at all (next latent == current latent) -- the
    # standard "is the model doing more than nothing" bar for a delta-prediction model,
    # same role train_world_model.py's own comparisons serve for the state-space model.
    baseline_val_mse = loss_fn(val_lat_t, val_lat_next).item()
    print(f"trivial baseline (predict zero latent delta) val MSE: {baseline_val_mse:.5f}")

    n = len(lat_t)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            pred_next = model(lat_t[idx], act_t[idx])
            loss = loss_fn(pred_next, lat_next[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        train_mse = total_loss / n

        model.eval()
        with torch.no_grad():
            val_pred_next = model(val_lat_t, val_act_t)
            val_mse = loss_fn(val_pred_next, val_lat_next).item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}: train MSE {train_mse:.5f}  val MSE {val_mse:.5f}")

    print(f"Final: train MSE {train_mse:.5f}, val MSE {val_mse:.5f}, "
          f"trivial zero-delta baseline val MSE {baseline_val_mse:.5f} "
          f"({'BEATS' if val_mse < baseline_val_mse else 'DOES NOT beat'} the baseline).")

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "latent_dim": enc_ckpt["latent_dim"],
                "action_dim": train_action.shape[1]}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # Short autoregressive rollout, decoded back to images every step -- the actual test
    # this project always insists on (Stage 1's own lesson: one-step metrics alone can look
    # great while compounding catastrophically under autoregressive use).
    model.eval()
    H = args.rollout_steps
    val_episode_ids = data["val_episode_ids"]
    starts = [i for i in val_pairs if i + H < len(val_episode_ids)
              and np.all(val_episode_ids[i:i + H + 1] == val_episode_ids[i])]
    if not starts:
        print("No val window long enough for the rollout check -- skipping.")
        return
    start = starts[len(starts) // 2]  # a window from the middle of some episode, not episode 0's edge

    with torch.no_grad():
        true_frames = [data["val_pixels_rgb"][start]]
        pred_frames = [data["val_pixels_rgb"][start]]
        z = val_latent[start:start + 1].clone()
        for h in range(H):
            a = val_action[start + h:start + h + 1]
            z = model(z, a)
            true_frames.append(data["val_pixels_rgb"][start + h + 1])
            decoded = encoder.decode(z).clamp(0, 1)[0].permute(1, 2, 0).numpy()
            pred_frames.append((decoded * 255).astype(np.uint8))

    from PIL import Image
    top = np.concatenate(true_frames, axis=1)
    bottom = np.concatenate(pred_frames, axis=1)
    grid = np.concatenate([top, bottom], axis=0)
    out_path = Path(args.rollout_grid)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)
    print(f"Saved {H + 1}-step ground-truth(top)/rollout(bottom) frame grid to {out_path}")


if __name__ == "__main__":
    main()
