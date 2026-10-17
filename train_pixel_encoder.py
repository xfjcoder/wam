"""Pixel-rendering mini-plan, step 5: train wam/pixel_encoder.py's ConvAutoencoder on the
RGB frames collect_data.py --record-pixels produced, and check it actually learns something
-- reconstruction MSE on held-out val frames, compared against a trivial "predict the mean
training frame" baseline (a model that beats this isn't just curve-fitting a near-constant
image), plus a saved before/after grid to look at directly.

Deliberately not a dynamics model: doesn't touch actions/next_states at all, doesn't feed
anything into CEMPlanner. Just "does a compact latent of this specific camera's view exist
and is it learnable" -- the gate before deciding whether pixel-conditioned dynamics is even
worth attempting next.

Usage (needs go1_rl_walk/mpc's own venv for torch):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 train_pixel_encoder.py \
        --data data/rollouts_pixels.npz --checkpoint checkpoints/pixel_encoder.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wam.pixel_encoder import ConvAutoencoder  # noqa: E402


def to_tensor_01(pixels_u8: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float32 in [0, 1]."""
    t = torch.as_tensor(pixels_u8, dtype=torch.float32) / 255.0
    return t.permute(0, 3, 1, 2).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/rollouts_pixels.npz")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/pixel_encoder.pt")
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recon-grid", type=str, default="media/pixel_encoder_reconstructions.png",
                     help="saved original-vs-reconstructed grid for a handful of val frames")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    data = np.load(args.data)
    resolution = int(data["pixel_resolution"])
    train_x = to_tensor_01(data["train_pixels_rgb"])
    val_x = to_tensor_01(data["val_pixels_rgb"])
    print(f"train frames: {len(train_x)}, val frames: {len(val_x)}, resolution: {resolution}")

    model = ConvAutoencoder(latent_dim=args.latent_dim, resolution=resolution)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Trivial baseline: predict the per-pixel MEAN training frame for every input, regardless
    # of what it actually is -- a model that doesn't beat this isn't learning per-frame
    # structure at all, just the dataset's overall color/lighting.
    mean_frame = train_x.mean(dim=0, keepdim=True)
    baseline_val_mse = loss_fn(mean_frame.expand_as(val_x), val_x).item()
    print(f"trivial baseline (predict mean training frame) val MSE: {baseline_val_mse:.5f}")

    n = len(train_x)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            batch = train_x[idx]
            recon = model(batch)
            loss = loss_fn(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        train_mse = total_loss / n

        model.eval()
        with torch.no_grad():
            val_recon = model(val_x)
            val_mse = loss_fn(val_recon, val_x).item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}: train MSE {train_mse:.5f}  val MSE {val_mse:.5f}")

    print(f"Final: train MSE {train_mse:.5f}, val MSE {val_mse:.5f}, "
          f"trivial baseline val MSE {baseline_val_mse:.5f} "
          f"({'BEATS' if val_mse < baseline_val_mse else 'DOES NOT beat'} the baseline).")

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "latent_dim": args.latent_dim,
                "resolution": resolution}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    from PIL import Image
    model.eval()
    with torch.no_grad():
        n_show = min(8, len(val_x))
        idx = torch.linspace(0, len(val_x) - 1, n_show).long()
        originals = val_x[idx]
        recons = model(originals).clamp(0, 1)
    grid = torch.cat([originals, recons], dim=2)  # stack recon below original, per sample
    grid = grid.permute(0, 2, 3, 1).numpy()  # (n_show, 2*res, res, 3)
    grid = (grid * 255).astype(np.uint8)
    row = np.concatenate(list(grid), axis=1)  # (2*res, n_show*res, 3)
    out_path = Path(args.recon_grid)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(row).save(out_path)
    print(f"Saved {n_show} original(top)/reconstruction(bottom) pairs to {out_path}")


if __name__ == "__main__":
    main()
