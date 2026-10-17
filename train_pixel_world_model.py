"""Pixel-rendering mini-plan, step 7: joint training, the idea step 6's honest conclusion
flagged as the natural next move. Step 6 trained wam/pixel_dynamics.py's LatentDynamics on
top of a FROZEN, reconstruction-only encoder (step 5's checkpoint) and found a weak one-step
win that degraded fast under rollout -- a latent shaped purely by "what's easy to
reconstruct" isn't obviously the same latent that's easy to PREDICT forward in time.

Here the encoder, decoder, and dynamics model (same wam/pixel_encoder.py's ConvAutoencoder
and wam/pixel_dynamics.py's LatentDynamics classes, nothing new architecturally) train
TOGETHER from a fresh random init, on a combined loss over every (frame_t, action,
frame_{t+1}) triple:
  - reconstruction: decode(encode(frame_t)) vs frame_t, and the same for frame_{t+1} --
    keeps the latent anchored to something decodable at all; without this the encoder could
    collapse to whatever makes prediction trivial (e.g. a near-constant latent) rather than
    actually representing the frame.
  - prediction: decode(dynamics(encode(frame_t), action)) vs frame_{t+1} -- the actual
    forward-prediction task, supervised directly in PIXEL space (not just latent space) so
    the encoder gets a gradient signal that cares about what's predictable, not only what's
    reconstructible.
Equal weight on both by default -- deliberately not tuned against this specific dataset,
since the point of this check is whether the basic idea helps at all before investing in
weight-tuning.

`LatentDynamics`'s own input/output normalization buffers are refit periodically (every
`--refit-every` epochs, using a no-grad pass over the CURRENT encoder) rather than once --
the encoder's latent distribution isn't fixed here like it was in step 6, so a one-time fit
from a freshly-initialized (effectively random) encoder would be close to meaningless.

Usage (needs go1_rl_walk/mpc's own venv for torch):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 train_pixel_world_model.py \
        --data data/rollouts_pixels.npz --checkpoint checkpoints/pixel_world_model.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wam.pixel_encoder import ConvAutoencoder, to_tensor_01        # noqa: E402
from wam.pixel_dynamics import LatentDynamics, consecutive_pairs  # noqa: E402


def refit_normalization(encoder, dynamics, train_x, train_action, train_pairs):
    encoder.eval()
    with torch.no_grad():
        latents = encoder.encode(train_x)
        lat_t = latents[train_pairs]
        lat_next = latents[train_pairs + 1]
        dynamics.fit_normalization(lat_t, train_action[train_pairs], lat_next - lat_t)
    encoder.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/rollouts_pixels.npz")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/pixel_world_model.pt")
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-recon", type=float, default=1.0)
    ap.add_argument("--w-pred", type=float, default=1.0)
    ap.add_argument("--refit-every", type=int, default=10,
                     help="epochs between LatentDynamics normalization-buffer refits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout-steps", type=int, default=5)
    ap.add_argument("--rollout-grid", type=str, default="media/pixel_world_model_rollout.png")
    ap.add_argument("--recon-grid", type=str, default="media/pixel_world_model_reconstructions.png")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    data = np.load(args.data)
    resolution = int(data["pixel_resolution"])

    train_x = to_tensor_01(data["train_pixels_rgb"])
    val_x = to_tensor_01(data["val_pixels_rgb"])
    train_action = torch.as_tensor(data["train_actions"], dtype=torch.float32)
    val_action = torch.as_tensor(data["val_actions"], dtype=torch.float32)
    train_pairs = consecutive_pairs(data["train_episode_ids"])
    val_pairs = consecutive_pairs(data["val_episode_ids"])
    print(f"train frames: {len(train_x)} ({len(train_pairs)} pairs), "
          f"val frames: {len(val_x)} ({len(val_pairs)} pairs)")

    encoder = ConvAutoencoder(latent_dim=args.latent_dim, resolution=resolution)
    dynamics = LatentDynamics(latent_dim=args.latent_dim, action_dim=train_action.shape[1])
    params = list(encoder.parameters()) + list(dynamics.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    loss_fn = nn.MSELoss()

    # Pixel-space trivial baseline: predict "nothing changes" (frame_{t+1} == frame_t) --
    # the direct pixel-space analogue of step 6's latent-space zero-delta baseline.
    baseline_val_mse = loss_fn(val_x[val_pairs], val_x[val_pairs + 1]).item()
    print(f"trivial baseline (predict frame_t for frame_t+1) val MSE: {baseline_val_mse:.5f}")

    refit_normalization(encoder, dynamics, train_x, train_action, train_pairs)

    n = len(train_pairs)
    for epoch in range(1, args.epochs + 1):
        if epoch > 1 and (epoch - 1) % args.refit_every == 0:
            refit_normalization(encoder, dynamics, train_x, train_action, train_pairs)

        encoder.train()
        dynamics.train()
        perm = train_pairs[torch.randperm(n)]
        total_recon, total_pred = 0.0, 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            x_t = train_x[idx]
            x_next = train_x[idx + 1]
            a = train_action[idx]

            z_t = encoder.encode(x_t)
            z_next = encoder.encode(x_next)
            recon_t = encoder.decode(z_t)
            recon_next = encoder.decode(z_next)
            z_next_pred = dynamics(z_t, a)
            pred_next = encoder.decode(z_next_pred)

            recon_loss = loss_fn(recon_t, x_t) + loss_fn(recon_next, x_next)
            pred_loss = loss_fn(pred_next, x_next)
            loss = args.w_recon * recon_loss + args.w_pred * pred_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_recon += recon_loss.item() * len(idx)
            total_pred += pred_loss.item() * len(idx)

        if epoch % 10 == 0 or epoch == 1:
            encoder.eval()
            dynamics.eval()
            with torch.no_grad():
                val_z_t = encoder.encode(val_x[val_pairs])
                val_z_next_pred = dynamics(val_z_t, val_action[val_pairs])
                val_pred_next = encoder.decode(val_z_next_pred)
                val_pred_mse = loss_fn(val_pred_next, val_x[val_pairs + 1]).item()
                val_recon_mse = loss_fn(encoder(val_x), val_x).item()
            print(f"epoch {epoch:3d}: train recon {total_recon / n:.5f}  train pred "
                  f"{total_pred / n:.5f}  val recon {val_recon_mse:.5f}  val pred {val_pred_mse:.5f}")

    print(f"Final: val reconstruction MSE {val_recon_mse:.5f}, val one-step prediction MSE "
          f"{val_pred_mse:.5f}, trivial baseline val MSE {baseline_val_mse:.5f} "
          f"({'BEATS' if val_pred_mse < baseline_val_mse else 'DOES NOT beat'} the baseline).")

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder_state": encoder.state_dict(), "dynamics_state": dynamics.state_dict(),
                "latent_dim": args.latent_dim, "resolution": resolution,
                "action_dim": train_action.shape[1]}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    from PIL import Image
    encoder.eval()
    dynamics.eval()

    # Reconstruction grid -- same format as step 5's, to compare quality honestly against a
    # reconstruction-ONLY encoder (does sharing capacity with prediction cost reconstruction
    # fidelity?).
    with torch.no_grad():
        n_show = min(8, len(val_x))
        show_idx = torch.linspace(0, len(val_x) - 1, n_show).long()
        originals = val_x[show_idx]
        recons = encoder(originals).clamp(0, 1)
    grid = torch.cat([originals, recons], dim=2)
    grid = (grid.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    row = np.concatenate(list(grid), axis=1)
    Path(args.recon_grid).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(row).save(args.recon_grid)
    print(f"Saved {n_show} original(top)/reconstruction(bottom) pairs to {args.recon_grid}")

    # Same 5-step autoregressive rollout check as step 6, same format, for a direct
    # side-by-side comparison against media/pixel_dynamics_rollout.png.
    H = args.rollout_steps
    val_episode_ids = data["val_episode_ids"]
    starts = [i for i in val_pairs if i + H < len(val_episode_ids)
              and np.all(val_episode_ids[i:i + H + 1] == val_episode_ids[i])]
    if not starts:
        print("No val window long enough for the rollout check -- skipping.")
        return
    start = starts[len(starts) // 2]

    with torch.no_grad():
        true_frames = [data["val_pixels_rgb"][start]]
        pred_frames = [data["val_pixels_rgb"][start]]
        z = encoder.encode(val_x[start:start + 1])
        for h in range(H):
            a = val_action[start + h:start + h + 1]
            z = dynamics(z, a)
            true_frames.append(data["val_pixels_rgb"][start + h + 1])
            decoded = encoder.decode(z).clamp(0, 1)[0].permute(1, 2, 0).numpy()
            pred_frames.append((decoded * 255).astype(np.uint8))

    top = np.concatenate(true_frames, axis=1)
    bottom = np.concatenate(pred_frames, axis=1)
    rollout_grid = np.concatenate([top, bottom], axis=0)
    Path(args.rollout_grid).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rollout_grid).save(args.rollout_grid)
    print(f"Saved {H + 1}-step ground-truth(top)/rollout(bottom) frame grid to {args.rollout_grid}")


if __name__ == "__main__":
    main()
