"""Pixel-rendering mini-plan, step 15: trains wam/pixel_cost.py's LatentCostHead on top of a
FROZEN, already-trained depth-modality checkpoint (step 10/14's best lever), then runs the
actual test that matters for CEM usability -- not "can this reconstruct a frame", but "can a
candidate's cost be read straight off a PREDICTED (rolled-out) latent, with the decoder never
touched, and still track the real future danger level."

Target, `--target {min,far}`: step 15/16/17 used `min` (each frame's own MINIMUM depth value)
under the assumption it meant "distance to the nearest obstacle". Step 18 found this was
WRONG in an important way: on this camera's geometry (mounted low, pitched ~26deg below
horizontal), the per-frame minimum is almost always the near-field GROUND-PLANE intersection
point directly in front of the robot (~0.20-0.36m, confirmed near-constant across both the
flat-ground AND wall-approach datasets regardless of wall distance) -- it only dips below that
floor baseline in the last ~10cm before actual contact, making it a near-collision reflex, not
a several-meter "how far is the wall" signal.

`far` (each frame's own MAXIMUM depth value, clipped to `--depth-clip`) is the fix:
check_onboard_camera.py's ORIGINAL sanity check already confirmed this specific statistic (it
calls it "far-depth" in its own printed output) shrinks smoothly and monotonically from 3.62m
down to ~0.37m as the wall-demo robot approaches a wall, then back up on retreat -- a genuine
medium-range obstacle-distance signal, not a binary near-collision flag. `min` is kept as the
default for backward compatibility with already-reported step 15-17 results; new work should
use `--target far`.

Three checks, in increasing order of how much they matter for actual CEM use:
  1. one-step: does the cost head predict min-depth accurately from an ENCODED real frame's
     latent? (sanity check only -- doesn't yet test the no-decode rollout path)
  2. rollout: does it predict min-depth accurately from a PREDICTED latent, H steps into an
     autoregressive rollout, with the decoder never called? This is the actual CEM use case.
  3. speed: scoring N candidates over a horizon via (dynamics + cost head, no decode) vs.
     (dynamics + decode + numpy-style min()) every step -- the concrete speed claim this step
     exists to check.

Usage (needs go1_rl_walk/mpc's own venv for torch):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 train_pixel_cost.py \
        --checkpoint checkpoints/pixel_world_model_wall_demo_depth.pt \
        --data data/rollouts_wall_demo.npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wam.pixel_encoder import ConvAutoencoder, depth_to_tensor_01  # noqa: E402
from wam.pixel_dynamics import LatentDynamics, build_window_starts  # noqa: E402
from wam.pixel_cost import LatentCostHead                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str,
                     default="checkpoints/pixel_world_model_wall_demo_depth.pt")
    ap.add_argument("--data", type=str, default="data/rollouts_wall_demo.npz")
    ap.add_argument("--out", type=str, default="checkpoints/pixel_cost_head.pt")
    ap.add_argument("--depth-clip", type=float, default=3.0)
    ap.add_argument("--target", type=str, default="min", choices=["min", "far"],
                     help="min (step 15-17's original target) is a near-collision reflex, not "
                          "a medium-range signal -- see module docstring. far (step 18's fix) "
                          "is each frame's clipped MAXIMUM depth, a genuine multi-meter "
                          "obstacle-distance signal.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rollout-horizon", type=int, default=20)
    ap.add_argument("--n-bench-candidates", type=int, default=1000)
    ap.add_argument("--bench-horizon", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert ckpt.get("modality") == "depth", (
        "cost head targets are metric min-depth -- needs a --modality depth checkpoint "
        f"(got modality={ckpt.get('modality')!r})")

    encoder = ConvAutoencoder(latent_dim=ckpt["latent_dim"], resolution=ckpt["resolution"],
                               channels=ckpt["channels"])
    encoder.load_state_dict(ckpt["encoder_state"])
    dynamics = LatentDynamics(latent_dim=ckpt["latent_dim"], action_dim=ckpt["action_dim"])
    dynamics.load_state_dict(ckpt["dynamics_state"])
    encoder.eval()
    dynamics.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in dynamics.parameters():
        p.requires_grad_(False)

    def frame_target(depth_raw: np.ndarray) -> np.ndarray:
        flat = depth_raw.reshape(len(depth_raw), -1)
        if args.target == "min":
            return flat.min(axis=1)
        return np.minimum(flat.max(axis=1), args.depth_clip)

    data = np.load(args.data)
    train_depth_raw = data["train_pixels_depth"]
    val_depth_raw = data["val_pixels_depth"]
    train_x = depth_to_tensor_01(train_depth_raw, clip=args.depth_clip)
    val_x = depth_to_tensor_01(val_depth_raw, clip=args.depth_clip)
    train_target = torch.as_tensor(frame_target(train_depth_raw),
                                    dtype=torch.float32).unsqueeze(1)
    val_target = torch.as_tensor(frame_target(val_depth_raw),
                                  dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        train_latents = encoder.encode(train_x)
        val_latents = encoder.encode(val_x)

    cost_head = LatentCostHead(latent_dim=ckpt["latent_dim"])
    cost_head.fit_normalization(train_latents, train_target)
    opt = torch.optim.Adam(cost_head.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    baseline_val_mse = loss_fn(val_target, train_target.mean(0, keepdim=True).expand_as(
        val_target)).item()
    print(f"trivial baseline (predict training-mean {args.target}-depth, {train_target.mean().item():.3f}m) "
          f"val MSE: {baseline_val_mse:.5f}")

    n = len(train_latents)
    val_mse = val_corr = None
    for epoch in range(1, args.epochs + 1):
        cost_head.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            pred = cost_head(train_latents[idx])
            loss = loss_fn(pred, train_target[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        if epoch % 10 == 0 or epoch == 1:
            cost_head.eval()
            with torch.no_grad():
                val_pred = cost_head(val_latents)
                val_mse = loss_fn(val_pred, val_target).item()
                val_corr = np.corrcoef(val_pred.squeeze(1).numpy(),
                                        val_target.squeeze(1).numpy())[0, 1]
            print(f"epoch {epoch:3d}: train {total / n:.5f}  val {val_mse:.5f}  "
                  f"val_corr {val_corr:.3f}")

    print(f"Final (one-step, encoded real frame): val MSE {val_mse:.5f} (trivial baseline "
          f"{baseline_val_mse:.5f}, {'BEATS' if val_mse < baseline_val_mse else 'DOES NOT beat'} "
          f"it), val correlation {val_corr:.3f}")

    # --- Check 2, the one that actually matters: read the cost off a PREDICTED latent, H
    # steps into an autoregressive rollout, decoder never called -- the real CEM use case. ---
    H = args.rollout_horizon
    val_episode_ids = data["val_episode_ids"]
    val_action = torch.as_tensor(data["val_actions"], dtype=torch.float32)
    starts = build_window_starts(val_episode_ids, H)
    starts_t = torch.from_numpy(starts)
    per_frame_target = frame_target(val_depth_raw)
    true_future_target = np.stack(
        [per_frame_target[starts + h + 1] for h in range(H)], axis=1)  # (n_starts, H)

    cost_head.eval()
    with torch.no_grad():
        z = val_latents[starts_t]
        pred_costs = []
        for h in range(H):
            z = dynamics(z, val_action[starts_t + h])
            pred_costs.append(cost_head(z).squeeze(1).numpy())
    pred_costs = np.stack(pred_costs, axis=1)  # (n_starts, H)

    per_step_mse = ((pred_costs - true_future_target) ** 2).mean(axis=0)
    per_step_corr = [np.corrcoef(pred_costs[:, h], true_future_target[:, h])[0, 1]
                      for h in range(H)]
    print(f"\n{len(starts)} held-out {H}-step windows -- cost read off a PREDICTED latent, "
          f"no decode, vs. the TRUE future {args.target}-depth:")
    print(f"{'step':>4}  {'MSE':>10}  correlation")
    for h in range(H):
        print(f"{h + 1:4d}  {per_step_mse[h]:10.5f}  {per_step_corr[h]:.3f}")

    # --- Check 3: the actual speed claim -- scoring N candidates over a horizon via the
    # latent cost head (no decode) vs. decoding every step to depth and taking min(). ---
    N, HB = args.n_bench_candidates, args.bench_horizon
    z0 = val_latents[0:1].expand(N, -1).contiguous()
    rng = np.random.default_rng(args.seed)
    actions = torch.as_tensor(rng.normal(scale=0.1, size=(N, HB, ckpt["action_dim"])),
                               dtype=torch.float32)

    with torch.no_grad():
        z = z0
        for h in range(HB):
            z = dynamics(z, actions[:, h])
            _ = cost_head(z)
        t0 = time.time()
        z = z0
        for h in range(HB):
            z = dynamics(z, actions[:, h])
            _ = cost_head(z)
        t_latent = time.time() - t0

        reduce_fn = (lambda t: t.amin(dim=(1, 2, 3))) if args.target == "min" \
            else (lambda t: t.amax(dim=(1, 2, 3)))
        z = z0
        for h in range(HB):
            z = dynamics(z, actions[:, h])
            _ = reduce_fn(encoder.decode(z))
        t0 = time.time()
        z = z0
        for h in range(HB):
            z = dynamics(z, actions[:, h])
            _ = reduce_fn(encoder.decode(z))
        t_decode = time.time() - t0

    print(f"\nSpeed benchmark ({N} candidates x {HB} steps, one warmup pass each excluded): "
          f"latent-only cost head {t_latent * 1000:.1f}ms vs. decode-every-step "
          f"{t_decode * 1000:.1f}ms ({t_decode / t_latent:.1f}x slower)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cost_head_state": cost_head.state_dict(), "latent_dim": ckpt["latent_dim"],
                "source_checkpoint": args.checkpoint, "target": args.target,
                "depth_clip": args.depth_clip}, out_path)
    print(f"Saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
