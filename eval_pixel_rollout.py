"""Pixel-rendering mini-plan, step 13: a standalone checkpoint evaluator -- loads an
already-trained ConvAutoencoder+LatentDynamics checkpoint (no retraining) and reports
PER-STEP accuracy over an N-step autoregressive rollout on held-out validation windows,
instead of the single averaged-over-5-steps number every train_*.py script's own
end-of-training check reports. Every pixel-WAM result so far (steps 6-12) only checked a
5-step rollout -- the real CEM planner looks ahead much further (10-20 macro-steps), so "does
5-step accuracy hold" says nothing about whether a checkpoint is actually usable there. This
lets us ask "how far out does an existing model's accuracy hold" directly and cheaply, without
retraining anything.

Usage (needs go1_rl_walk/mpc's own venv for torch):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_pixel_rollout.py \
        --checkpoint checkpoints/pixel_world_model_wall_demo_depth.pt \
        --data data/rollouts_wall_demo.npz --horizon 20
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wam.pixel_encoder import ConvAutoencoder, to_tensor_01, depth_to_tensor_01  # noqa: E402
from wam.pixel_dynamics import LatentDynamics, build_window_starts              # noqa: E402


def frames_to_uint8(t: torch.Tensor) -> np.ndarray:
    arr = t.clamp(0, 1).detach().numpy()
    if arr.shape[1] == 1:
        arr = np.repeat(arr, 3, axis=1)
    return (arr.transpose(0, 2, 3, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--depth-clip", type=float, default=3.0)
    ap.add_argument("--n-windows", type=int, default=64,
                     help="held-out validation windows averaged per step")
    ap.add_argument("--rollout-grid", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    modality = ckpt.get("modality", "rgb")
    channels = ckpt.get("channels", 3)
    resolution = ckpt["resolution"]

    encoder = ConvAutoencoder(latent_dim=ckpt["latent_dim"], resolution=resolution,
                               channels=channels)
    encoder.load_state_dict(ckpt["encoder_state"])
    dynamics = LatentDynamics(latent_dim=ckpt["latent_dim"], action_dim=ckpt["action_dim"])
    dynamics.load_state_dict(ckpt["dynamics_state"])
    encoder.eval()
    dynamics.eval()

    data = np.load(args.data)
    if modality == "rgb":
        val_x = to_tensor_01(data["val_pixels_rgb"])
    else:
        val_x = depth_to_tensor_01(data["val_pixels_depth"], clip=args.depth_clip)
    val_action = torch.as_tensor(data["val_actions"], dtype=torch.float32)

    H = args.horizon
    starts = build_window_starts(data["val_episode_ids"], H)
    print(f"checkpoint modality={modality}, {len(starts)} held-out {H}-step windows available.")

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(starts, size=min(args.n_windows, len(starts)), replace=False)
    chosen_t = torch.from_numpy(chosen)

    loss_fn = torch.nn.MSELoss(reduction="none")
    per_step_mse, per_step_baseline = [], []
    with torch.no_grad():
        z = encoder.encode(val_x[chosen_t])
        x0 = val_x[chosen_t]
        for h in range(H):
            z = dynamics(z, val_action[chosen_t + h])
            decoded = encoder.decode(z).clamp(0, 1)
            target = val_x[chosen_t + h + 1]
            per_step_mse.append(loss_fn(decoded, target).mean(dim=(1, 2, 3)).mean().item())
            per_step_baseline.append(loss_fn(x0, target).mean(dim=(1, 2, 3)).mean().item())

    print(f"{'step':>4}  {'pred_MSE':>10}  {'baseline_MSE':>12}  beats_baseline")
    for h, (p, b) in enumerate(zip(per_step_mse, per_step_baseline), start=1):
        print(f"{h:4d}  {p:10.5f}  {b:12.5f}  {'YES' if p < b else 'no'}")

    if args.rollout_grid:
        from PIL import Image
        start = int(chosen[len(chosen) // 2])
        with torch.no_grad():
            true_frames = [frames_to_uint8(val_x[start:start + 1])[0]]
            pred_frames = [frames_to_uint8(val_x[start:start + 1])[0]]
            z = encoder.encode(val_x[start:start + 1])
            for h in range(H):
                z = dynamics(z, val_action[start + h:start + h + 1])
                true_frames.append(frames_to_uint8(val_x[start + h + 1:start + h + 2])[0])
                pred_frames.append(frames_to_uint8(encoder.decode(z))[0])
        top = np.concatenate(true_frames, axis=1)
        bottom = np.concatenate(pred_frames, axis=1)
        grid = np.concatenate([top, bottom], axis=0)
        Path(args.rollout_grid).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(grid).save(args.rollout_grid)
        print(f"Saved {H + 1}-step rollout grid to {args.rollout_grid}")


if __name__ == "__main__":
    main()
