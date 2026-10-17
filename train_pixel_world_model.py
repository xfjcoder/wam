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

Also adds the H-step rollout loss flagged (but not yet tried) since step 6: steps 6/7/8 only
ever trained on ONE-STEP prediction loss, exactly the gap train_world_model.py's own docstring
warns about for the state-based WorldModel -- one-step loss alone looked fine there too, but
errors compounded catastrophically under the autoregressive rollout actually used at eval
time, fixed by training on an explicit H-step rollout loss (PETS/MBPO/Dreamer-style), not
one-step error alone. Same fix here: unroll `dynamics` for `--rollout-train-horizon` steps in
LATENT space, decode at every step, and penalize divergence from the true frame at every one
of those steps (see wam/pixel_dynamics.py's build_window_starts for the frame-adjacency
windows this needs). Backprops through H repeated dynamics+decode applications (BPTT-style),
same gradient-spike risk train_world_model.py already guards against with --grad-clip.

Step 10: `--modality depth` trains on the SAME recording's depth channel instead of RGB (one
flag, zero new data collection -- collect_data.py/collect_wall_approach_demo.py already
capture both every episode). Depth directly encodes metric distance-to-obstacle -- the actual
quantity worth predicting for obstacle sensing -- instead of forcing the model to track an
incidental, subtle sub-pixel texture shift the way flat-ground RGB does. `ConvAutoencoder`
gained a `channels` arg (3 for RGB, 1 for depth) and `wam/pixel_encoder.py` gained
`depth_to_tensor_01` (clips metric depth to `--depth-clip` meters before normalizing -- raw
depth has sky/no-geometry pixels reading back at the renderer's far plane, ~200m on the
flat-ground dataset, which would otherwise swamp the loss).

Step 11: `--loss-weighting diff` reweights the PREDICTION and rollout losses (not
reconstruction, which isn't about change) by how much each pixel actually moved between
consecutive ground-truth frames (`1 + --diff-weight-scale * |frame_next - frame_t|`,
normalized to mean 1 per batch so the overall loss scale is comparable to the unweighted
case). On a mostly-static scene (flat-ground RGB's dominant failure mode) plain per-pixel MSE
is overwhelmingly dominated by the large, correctly-near-static majority of the frame,
diluting the gradient signal from the small informative region that's actually predictable.
This directly targets that dilution -- evaluation metrics (the baselines and final MSEs
printed/compared) stay the standard UNWEIGHTED pixel MSE throughout, only the training
objective changes, so "beats baseline" comparisons remain honest/apples-to-apples.

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
from wam.pixel_encoder import ConvAutoencoder, to_tensor_01, depth_to_tensor_01       # noqa: E402
from wam.pixel_dynamics import LatentDynamics, consecutive_pairs, build_window_starts  # noqa: E402


def frames_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(B, C, H, W) float in [0, 1] -> (B, H, W, 3) uint8 -- replicates a single depth
    channel to RGB so both modalities save through the same grid-building code below."""
    arr = t.clamp(0, 1).detach().numpy()
    if arr.shape[1] == 1:
        arr = np.repeat(arr, 3, axis=1)
    return (arr.transpose(0, 2, 3, 1) * 255).astype(np.uint8)


def diff_weighted_mse(pred: torch.Tensor, target: torch.Tensor, ref: torch.Tensor,
                       scale: float) -> torch.Tensor:
    """MSE(pred, target), with each pixel weighted by how much `target` actually changed
    from `ref` (the previous ground-truth frame) -- weight = 1 + scale * |target - ref|,
    normalized to mean 1 so the overall loss magnitude stays comparable to plain MSE. The
    weight is a stop-gradient target statistic, not a model output."""
    with torch.no_grad():
        weight = 1.0 + scale * (target - ref).abs()
        weight = weight / weight.mean()
    return (weight * (pred - target) ** 2).mean()


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
    ap.add_argument("--modality", type=str, default="rgb", choices=["rgb", "depth"],
                     help="train on RGB frames or the same recording's depth channel")
    ap.add_argument("--depth-clip", type=float, default=3.0,
                     help="meters -- depth is clipped to this before normalizing to [0, 1] "
                          "(only used when --modality depth)")
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-recon", type=float, default=1.0)
    ap.add_argument("--w-pred", type=float, default=1.0)
    ap.add_argument("--refit-every", type=int, default=10,
                     help="epochs between LatentDynamics normalization-buffer refits")
    ap.add_argument("--rollout-loss-weight", type=float, default=1.0,
                     help="weight on the H-step pixel-space rollout loss, added on top of "
                          "the one-step recon+pred losses above")
    ap.add_argument("--rollout-train-horizon", type=int, default=5,
                     help="steps unrolled autoregressively per rollout-loss batch, in LATENT "
                          "space (decoded to pixels for the loss) -- same role as "
                          "train_world_model.py's --rollout-horizon")
    ap.add_argument("--rollout-batches-per-epoch", type=int, default=10)
    ap.add_argument("--rollout-batch-size", type=int, default=16)
    ap.add_argument("--loss-weighting", type=str, default="none", choices=["none", "diff"],
                     help="'diff' reweights the PREDICTION and rollout losses toward pixels "
                          "that actually changed frame-to-frame (see module docstring); "
                          "reconstruction loss and all evaluation metrics stay unweighted")
    ap.add_argument("--diff-weight-scale", type=float, default=5.0,
                     help="only used when --loss-weighting diff")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                     help="max gradient norm, applied to both loss passes -- the H-step "
                          "rollout loss backprops through H repeated dynamics+decode "
                          "applications (BPTT-style), which can produce occasional much "
                          "larger gradients than the one-step losses ever do")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout-steps", type=int, default=5)
    ap.add_argument("--rollout-grid", type=str, default="media/pixel_world_model_rollout.png")
    ap.add_argument("--recon-grid", type=str, default="media/pixel_world_model_reconstructions.png")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    data = np.load(args.data)
    resolution = int(data["pixel_resolution"])

    if args.modality == "rgb":
        channels = 3
        train_x = to_tensor_01(data["train_pixels_rgb"])
        val_x = to_tensor_01(data["val_pixels_rgb"])
    else:
        channels = 1
        train_x = depth_to_tensor_01(data["train_pixels_depth"], clip=args.depth_clip)
        val_x = depth_to_tensor_01(data["val_pixels_depth"], clip=args.depth_clip)
    train_action = torch.as_tensor(data["train_actions"], dtype=torch.float32)
    val_action = torch.as_tensor(data["val_actions"], dtype=torch.float32)
    train_pairs = consecutive_pairs(data["train_episode_ids"])
    val_pairs = consecutive_pairs(data["val_episode_ids"])
    print(f"train frames: {len(train_x)} ({len(train_pairs)} pairs), "
          f"val frames: {len(val_x)} ({len(val_pairs)} pairs)")

    Htrain = args.rollout_train_horizon
    train_window_starts = torch.from_numpy(build_window_starts(data["train_episode_ids"], Htrain))
    val_window_starts = torch.from_numpy(build_window_starts(data["val_episode_ids"], Htrain))
    print(f"Rollout-loss windows (horizon={Htrain}): {len(train_window_starts)} train, "
          f"{len(val_window_starts)} val.")

    encoder = ConvAutoencoder(latent_dim=args.latent_dim, resolution=resolution, channels=channels)
    dynamics = LatentDynamics(latent_dim=args.latent_dim, action_dim=train_action.shape[1])
    params = list(encoder.parameters()) + list(dynamics.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    loss_fn = nn.MSELoss()

    # Pixel-space trivial baseline: predict "nothing changes" (frame_{t+1} == frame_t) --
    # the direct pixel-space analogue of step 6's latent-space zero-delta baseline.
    baseline_val_mse = loss_fn(val_x[val_pairs], val_x[val_pairs + 1]).item()
    print(f"trivial baseline (predict frame_t for frame_t+1) val MSE: {baseline_val_mse:.5f}")

    # Multi-step analogue of the baseline above -- "predict no change for H steps" -- the
    # honest comparison for the NEW rollout loss/metric below (a model that merely matches
    # the one-step baseline could still lose badly to this over H steps if its errors
    # compound, which is exactly the failure mode being targeted here).
    with torch.no_grad():
        baseline_roll = sum(
            loss_fn(val_x[val_window_starts], val_x[val_window_starts + h + 1]).item()
            for h in range(Htrain)) / Htrain
    print(f"trivial {Htrain}-step rollout baseline (predict frame_t unchanged for all steps) "
          f"val MSE: {baseline_roll:.5f}")

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
            if args.loss_weighting == "diff":
                pred_loss = diff_weighted_mse(pred_next, x_next, x_t, args.diff_weight_scale)
            else:
                pred_loss = loss_fn(pred_next, x_next)
            loss = args.w_recon * recon_loss + args.w_pred * pred_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            total_recon += recon_loss.item() * len(idx)
            total_pred += pred_loss.item() * len(idx)

        # --- H-step rollout loss pass (penalizes autoregressive compounding directly, in
        # PIXEL space) -- unroll in LATENT space for Htrain steps, decode at every step, and
        # penalize divergence from the true frame at every one of those steps, not just step
        # 1. This is the exact fix steps 6/7/8 never tried (see module docstring).
        roll_loss_total = 0.0
        for _ in range(args.rollout_batches_per_epoch):
            idx = train_window_starts[torch.randint(0, len(train_window_starts),
                                                      (args.rollout_batch_size,))]
            z = encoder.encode(train_x[idx])
            loss = 0.0
            for h in range(Htrain):
                z = dynamics(z, train_action[idx + h])
                decoded = encoder.decode(z)
                if args.loss_weighting == "diff":
                    loss = loss + diff_weighted_mse(decoded, train_x[idx + h + 1],
                                                      train_x[idx + h], args.diff_weight_scale)
                else:
                    loss = loss + loss_fn(decoded, train_x[idx + h + 1])
            loss = (loss / Htrain) * args.rollout_loss_weight
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            roll_loss_total += loss.item()
        roll_loss_total /= args.rollout_batches_per_epoch

        if epoch % 10 == 0 or epoch == 1:
            encoder.eval()
            dynamics.eval()
            with torch.no_grad():
                val_z_t = encoder.encode(val_x[val_pairs])
                val_z_next_pred = dynamics(val_z_t, val_action[val_pairs])
                val_pred_next = encoder.decode(val_z_next_pred)
                val_pred_mse = loss_fn(val_pred_next, val_x[val_pairs + 1]).item()
                val_recon_mse = loss_fn(encoder(val_x), val_x).item()

                val_z = encoder.encode(val_x[val_window_starts])
                val_roll_err = 0.0
                for h in range(Htrain):
                    val_z = dynamics(val_z, val_action[val_window_starts + h])
                    val_roll_err += loss_fn(encoder.decode(val_z).clamp(0, 1),
                                             val_x[val_window_starts + h + 1]).item()
                val_roll_err /= Htrain
            print(f"epoch {epoch:3d}: train recon {total_recon / n:.5f}  train pred "
                  f"{total_pred / n:.5f}  train rollout {roll_loss_total:.5f}  val recon "
                  f"{val_recon_mse:.5f}  val pred {val_pred_mse:.5f}  val {Htrain}step-rollout "
                  f"{val_roll_err:.5f}")

    print(f"Final: val reconstruction MSE {val_recon_mse:.5f}, val one-step prediction MSE "
          f"{val_pred_mse:.5f} (trivial baseline {baseline_val_mse:.5f}, "
          f"{'BEATS' if val_pred_mse < baseline_val_mse else 'DOES NOT beat'} it); "
          f"val {Htrain}-step rollout MSE {val_roll_err:.5f} (trivial {Htrain}-step baseline "
          f"{baseline_roll:.5f}, {'BEATS' if val_roll_err < baseline_roll else 'DOES NOT beat'} it).")

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder_state": encoder.state_dict(), "dynamics_state": dynamics.state_dict(),
                "latent_dim": args.latent_dim, "resolution": resolution, "channels": channels,
                "modality": args.modality, "action_dim": train_action.shape[1]}, ckpt_path)
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
    grid = np.concatenate([frames_to_uint8(originals), frames_to_uint8(recons)], axis=1)
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
        true_frames = [frames_to_uint8(val_x[start:start + 1])[0]]
        pred_frames = [frames_to_uint8(val_x[start:start + 1])[0]]
        z = encoder.encode(val_x[start:start + 1])
        for h in range(H):
            a = val_action[start + h:start + h + 1]
            z = dynamics(z, a)
            true_frames.append(frames_to_uint8(val_x[start + h + 1:start + h + 2])[0])
            pred_frames.append(frames_to_uint8(encoder.decode(z))[0])

    top = np.concatenate(true_frames, axis=1)
    bottom = np.concatenate(pred_frames, axis=1)
    rollout_grid = np.concatenate([top, bottom], axis=0)
    Path(args.rollout_grid).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rollout_grid).save(args.rollout_grid)
    print(f"Saved {H + 1}-step ground-truth(top)/rollout(bottom) frame grid to {args.rollout_grid}")


if __name__ == "__main__":
    main()
