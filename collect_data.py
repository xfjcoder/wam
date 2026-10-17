"""Stage 1: collect (state, action, next_state) rollout data from the real Go2 quadruped
(go1_rl_walk/mpc's MPCResidualEnv -- real MuJoCo dynamics + classical convex-MPC control,
imported unmodified) for training a world model.

Each episode: reset (which randomizes the initial velocity command per MPCResidualEnv's own
convention), then every `--hold-ticks` physics ticks (default 100 = 0.5s) resample a fresh
random command via `env.set_command()`, holding it constant in between. Every `--macro-k`
physics ticks (default 10 = 50ms) within that hold, record one world-model transition
`(state_t, action, state_{t+1})` -- see wam/state.py for exactly what "state" and "action"
mean here, and the README's "Scoping decisions" for why.

`action = zeros(12)` on every `env.step()` throughout -- the residual-torque RL action is
unused (every adopted demo across go1_rl_walk/slam_sim used zero residual, pure classical
MPC); the only control lever exercised is the velocity command.

An episode that falls or hits a QP failure is truncated there (no contaminated post-fall
transitions); data collection just moves on to the next episode.

Pixel-rendering mini-plan, step 4: `--record-pixels` additionally captures a paired RGB+depth
frame (wam/camera.py's `OnboardCamera`, the SAME onboard mount steps 1-3 already built and
verified) at the exact same macro-step cadence/alignment as `state_t` -- "what the camera saw
right when this transition's action was chosen", not next_state's moment. Saved as its own
dataset (`train_pixels_rgb`/`train_pixels_depth`, index-aligned with `train_states` etc.) in
the SAME .npz, not a separate file -- there is exactly one place this project's state/action
data gets collected, and the whole point of adding pixels now is to have them trivially
joinable with the data an encoder would eventually need to predict alongside, or instead of,
the existing low-dim state.

Usage (needs go1_rl_walk/mpc's own venv for mujoco/casadi/pinocchio):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 collect_data.py \
        --episodes 50 --val-episodes 10 --out data/rollouts.npz
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

GO1_RL_WALK = Path(__file__).resolve().parent.parent / "go1_rl_walk"
sys.path.insert(0, str(GO1_RL_WALK / "mpc"))
sys.path.insert(0, str(GO1_RL_WALK))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mpc_residual_env import MPCResidualEnv, DEFAULT_XML, _fix_meshdir_absolute  # noqa: E402
from wam.state import extract_state, extract_action, STATE_DIM, ACTION_DIM  # noqa: E402
from wam.camera import OnboardCamera, inject_onboard_camera                  # noqa: E402


def collect_episode(env, rng, episode_seconds, hold_ticks, macro_k,
                     max_accel=1.5, max_yaw_accel=3.0, cam=None):
    """Returns (states, actions, next_states), each (n_transitions, dim), for one episode --
    possibly shorter than the full episode_seconds if the robot fell or the QP failed.

    Target commands are resampled every `hold_ticks`, but the command actually SENT each
    tick ramps toward that target at a bounded rate (same `max_cmd_delta`-style clipping
    `demo_stage6/7/8_*.py` in the sibling slam_sim project use) -- an instantaneous jump to
    a very different random command destabilizes the classical MPC (confirmed directly:
    without this, episodes fell early far more often than with it)."""
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    max_ticks = int(episode_seconds * 200)
    dt = 1.0 / 200.0
    max_cmd_delta = max_accel * dt
    max_yaw_delta = max_yaw_accel * dt

    states, actions, next_states = [], [], []
    pixels_rgb, pixels_depth = [], []
    state_t = None
    action_t = None
    rgb_t = None
    depth_t = None
    ticks_in_macro = 0
    target_speed, target_lateral, target_yaw_rate = env.target_speed, env.target_lateral, env.target_yaw_rate
    cur_speed, cur_lateral, cur_yaw_rate = target_speed, target_lateral, target_yaw_rate

    for tick in range(max_ticks):
        if tick % hold_ticks == 0:
            target_speed = float(rng.uniform(*env.speed_range))
            target_lateral = float(rng.uniform(*env.lateral_range))
            target_yaw_rate = float(rng.uniform(*env.yaw_rate_range))

        cur_speed += np.clip(target_speed - cur_speed, -max_cmd_delta, max_cmd_delta)
        cur_lateral += np.clip(target_lateral - cur_lateral, -max_cmd_delta, max_cmd_delta)
        cur_yaw_rate += np.clip(target_yaw_rate - cur_yaw_rate, -max_yaw_delta, max_yaw_delta)
        env.set_command(cur_speed, cur_lateral, cur_yaw_rate)

        if ticks_in_macro == 0:
            info_for_state = {"base_height": float(env.mujoco_go2.data.qpos[2])}
            state_t = extract_state(obs, info_for_state)
            action_t = extract_action(env.target_speed, env.target_lateral, env.target_yaw_rate)
            if cam is not None:
                rgb_t = cam.rgb(env.mujoco_go2.data)
                depth_t = cam.depth(env.mujoco_go2.data)

        obs, _, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
        ticks_in_macro += 1

        if terminated:
            break   # fell / QP failed -- discard this incomplete macro-step, end episode

        if ticks_in_macro >= macro_k:
            state_next = extract_state(obs, info)
            states.append(state_t)
            actions.append(action_t)
            next_states.append(state_next)
            if cam is not None:
                pixels_rgb.append(rgb_t)
                pixels_depth.append(depth_t)
            ticks_in_macro = 0

        if truncated:
            break

    if not states:
        empty = (np.zeros((0, STATE_DIM), dtype=np.float32),
                 np.zeros((0, ACTION_DIM), dtype=np.float32),
                 np.zeros((0, STATE_DIM), dtype=np.float32))
        if cam is None:
            return empty
        res = cam.resolution
        return empty + (np.zeros((0, res, res, 3), dtype=np.uint8),
                         np.zeros((0, res, res), dtype=np.float32))

    result = (np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32),
              np.array(next_states, dtype=np.float32))
    if cam is None:
        return result
    return result + (np.array(pixels_rgb, dtype=np.uint8), np.array(pixels_depth, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50, help="training episodes")
    ap.add_argument("--val-episodes", type=int, default=10, help="held-out validation episodes")
    ap.add_argument("--episode-seconds", type=float, default=10.0)
    ap.add_argument("--hold-ticks", type=int, default=100, help="physics ticks a command is held for")
    ap.add_argument("--macro-k", type=int, default=10, help="physics ticks per world-model transition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/rollouts.npz")
    ap.add_argument("--terrain-amplitude-max", type=float, default=0.0,
                     help="m peak-to-peak random bumps -- MPCResidualEnv itself resamples a "
                          "fresh amplitude in [0, this] at every reset() (same mechanism it "
                          "already uses for speed_range/lateral_range/yaw_rate_range), so "
                          "episodes span flat ground through this amplitude's worth of bumps, "
                          "not a single fixed difficulty. 0.0 (default) reproduces Stage 1's "
                          "flat-only data collection exactly.")
    ap.add_argument("--record-pixels", action="store_true",
                     help="Pixel-rendering mini-plan step 4: also capture a paired RGB+depth "
                          "frame per transition from the onboard camera verified in steps 1-3 "
                          "(wam/camera.py), index-aligned with train/val_states. Off by "
                          "default -- adds real render time and .npz size for data nothing yet "
                          "consumes; every OTHER saved array is byte-identical whether this is "
                          "on or off.")
    ap.add_argument("--pixel-resolution", type=int, default=64,
                     help="square RGB/depth frame size when --record-pixels is set.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    terrain_range = (0.0, args.terrain_amplitude_max) if args.terrain_amplitude_max > 0 else None

    xml_path = None
    if args.record_pixels:
        xml_text = Path(DEFAULT_XML).read_text()
        xml_text = _fix_meshdir_absolute(xml_text, Path(DEFAULT_XML).resolve().parent)
        xml_text = inject_onboard_camera(xml_text)
        tmp_dir = Path(tempfile.mkdtemp(prefix="wam_collect_pixels_"))
        xml_path = tmp_dir / "go2_with_camera.xml"
        xml_path.write_text(xml_text)

    env = MPCResidualEnv(episode_seconds=args.episode_seconds, seed=args.seed,
                          terrain_amplitude_range=terrain_range,
                          **({"xml_path": str(xml_path)} if xml_path is not None else {}))
    cam = OnboardCamera(env.mujoco_go2.model, resolution=args.pixel_resolution) \
        if args.record_pixels else None

    def collect_n(n, tag):
        all_states, all_actions, all_next_states, episode_ids = [], [], [], []
        all_pixels_rgb, all_pixels_depth = [], []
        t0 = time.time()
        for ep in range(n):
            result = collect_episode(env, rng, args.episode_seconds, args.hold_ticks,
                                      args.macro_k, cam=cam)
            s, a, ns = result[:3]
            all_states.append(s)
            all_actions.append(a)
            all_next_states.append(ns)
            episode_ids.append(np.full(len(s), ep, dtype=np.int32))
            if cam is not None:
                all_pixels_rgb.append(result[3])
                all_pixels_depth.append(result[4])
            print(f"[{tag}] episode {ep + 1}/{n}: {len(s)} transitions "
                  f"({time.time() - t0:.1f}s elapsed)")
        out = [np.concatenate(all_states), np.concatenate(all_actions),
               np.concatenate(all_next_states), np.concatenate(episode_ids)]
        if cam is not None:
            out += [np.concatenate(all_pixels_rgb), np.concatenate(all_pixels_depth)]
        return out

    train_out = collect_n(args.episodes, "train")
    val_out = collect_n(args.val_episodes, "val")
    train_s, train_a, train_ns, train_ep = train_out[:4]
    val_s, val_a, val_ns, val_ep = val_out[:4]

    print(f"Collected {len(train_s)} training transitions ({args.episodes} episodes), "
          f"{len(val_s)} validation transitions ({args.val_episodes} episodes).")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = dict(
        train_states=train_s, train_actions=train_a, train_next_states=train_ns,
        train_episode_ids=train_ep,
        val_states=val_s, val_actions=val_a, val_next_states=val_ns,
        val_episode_ids=val_ep,
        hold_ticks=args.hold_ticks, macro_k=args.macro_k,
        episode_seconds=args.episode_seconds,
        terrain_amplitude_max=args.terrain_amplitude_max)
    if cam is not None:
        save_kwargs.update(
            train_pixels_rgb=train_out[4], train_pixels_depth=train_out[5],
            val_pixels_rgb=val_out[4], val_pixels_depth=val_out[5],
            pixel_resolution=args.pixel_resolution)
    np.savez(out_path, **save_kwargs)
    print(f"Saved dataset to {out_path}")
    if cam is not None:
        cam.close()


if __name__ == "__main__":
    main()
