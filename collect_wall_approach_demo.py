"""Pixel-rendering mini-plan, step 8: a deliberately EASY pixel-dynamics scenario, to isolate
whether steps 6/7's failure to predict was about the METHOD or about the flat-ground dataset
being a bad test case. That dataset's only real visual signal was a subtle, sub-pixel shift
in the floor's checker texture -- predicting it precisely means implicitly learning projective
geometry, a genuinely hard regression target for a tiny model/dataset.

Here the robot just drives straight at slam_sim's loop_corridor island wall, then straight
back, repeating -- a SCRIPTED (not randomly-sampled) forward-speed oscillation, not collect_
data.py's usual random-command exploration. The dominant visual change becomes "the gray wall
band in frame grows and shrinks", a large, simple, monotonic, directly action-correlated
signal (step 5/6/7's demo already confirmed this camera sees that band cleanly at these
ranges) -- about as easy as a pixel-dynamics prediction problem gets. If latent-dynamics
prediction still can't learn this, that's informative about the method itself; if it can,
that isolates the earlier failure to task difficulty/signal strength, not a broken approach.

Same per-transition recording cadence/format as collect_data.py's --record-pixels (one
(state, action, next_state, rgb, depth) tuple per macro-step) so train_pixel_encoder.py /
train_pixel_dynamics.py / train_pixel_world_model.py all work UNCHANGED on this dataset --
only how the ACTION is chosen differs (a scripted approach/retreat cycle instead of a
resampled-every-hold_ticks random command).

Usage (needs go1_rl_walk/mpc's own venv for mujoco/torch, and slam_sim importable as a
sibling repo):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 collect_wall_approach_demo.py \
        --episodes 10 --val-episodes 3 --out data/rollouts_wall_demo.npz
"""
import argparse
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

GO1_RL_WALK = Path(__file__).resolve().parent.parent / "go1_rl_walk"
SLAM_SIM = Path(__file__).resolve().parent.parent / "slam_sim"
sys.path.insert(0, str(GO1_RL_WALK / "mpc"))
sys.path.insert(0, str(GO1_RL_WALK))
sys.path.insert(0, str(SLAM_SIM))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mpc_residual_env import MPCResidualEnv, DEFAULT_XML, _fix_meshdir_absolute  # noqa: E402
from demo_stage6_go2_world import inject_walls                                   # noqa: E402
from slam.world import World                                                     # noqa: E402

from wam.state import extract_state, extract_action, STATE_DIM, ACTION_DIM  # noqa: E402
from wam.camera import OnboardCamera, inject_onboard_camera                 # noqa: E402

START_X, START_Y = 1.5, 4.0  # same proven-safe start check_onboard_camera.py uses


def build_xml(tmp_dir: Path) -> Path:
    world = World.loop_corridor()
    xml_text = Path(DEFAULT_XML).read_text()
    xml_text = _fix_meshdir_absolute(xml_text, Path(DEFAULT_XML).resolve().parent)
    xml_text = inject_walls(xml_text, world.walls)
    xml_text = inject_onboard_camera(xml_text)
    out_path = tmp_dir / "go2_wall_approach_demo.xml"
    out_path.write_text(xml_text)
    return out_path


def collect_episode(env, cam, rng, episode_seconds, approach_seconds, retreat_seconds,
                     speed, macro_k=10, max_accel=1.5):
    """Scripted forward-speed oscillation (approach then retreat, repeating) toward the
    island wall -- NOT collect_data.py's random-command resampling. Returns (states, actions,
    next_states, pixels_rgb, pixels_depth), index-aligned, same convention as collect_data.py's
    --record-pixels."""
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    env.mujoco_go2.data.qpos[0] = START_X
    env.mujoco_go2.data.qpos[1] = START_Y

    max_ticks = int(episode_seconds * 200)
    dt = 1.0 / 200.0
    max_cmd_delta = max_accel * dt
    cycle_seconds = approach_seconds + retreat_seconds

    states, actions, next_states, pixels_rgb, pixels_depth = [], [], [], [], []
    state_t = action_t = rgb_t = depth_t = None
    ticks_in_macro = 0
    cur_speed = 0.0

    for tick in range(max_ticks):
        phase_t = (tick * dt) % cycle_seconds
        target_speed = speed if phase_t < approach_seconds else -speed
        cur_speed += np.clip(target_speed - cur_speed, -max_cmd_delta, max_cmd_delta)
        env.set_command(cur_speed, 0.0, 0.0)

        if ticks_in_macro == 0:
            info_for_state = {"base_height": float(env.mujoco_go2.data.qpos[2])}
            state_t = extract_state(obs, info_for_state)
            action_t = extract_action(cur_speed, 0.0, 0.0)
            rgb_t = cam.rgb(env.mujoco_go2.data)
            depth_t = cam.depth(env.mujoco_go2.data)

        obs, _, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
        ticks_in_macro += 1

        if terminated:
            break

        if ticks_in_macro >= macro_k:
            states.append(state_t)
            actions.append(action_t)
            next_states.append(extract_state(obs, info))
            pixels_rgb.append(rgb_t)
            pixels_depth.append(depth_t)
            ticks_in_macro = 0

        if truncated:
            break

    if not states:
        res = cam.resolution
        return (np.zeros((0, STATE_DIM), dtype=np.float32),
                np.zeros((0, ACTION_DIM), dtype=np.float32),
                np.zeros((0, STATE_DIM), dtype=np.float32),
                np.zeros((0, res, res, 3), dtype=np.uint8),
                np.zeros((0, res, res), dtype=np.float32))
    return (np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32),
            np.array(next_states, dtype=np.float32), np.array(pixels_rgb, dtype=np.uint8),
            np.array(pixels_depth, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--val-episodes", type=int, default=3)
    ap.add_argument("--episode-seconds", type=float, default=10.0,
                     help="one full approach+retreat cycle by default")
    ap.add_argument("--approach-seconds", type=float, default=5.0)
    ap.add_argument("--retreat-seconds", type=float, default=5.0)
    ap.add_argument("--speed", type=float, default=0.35)
    ap.add_argument("--macro-k", type=int, default=10)
    ap.add_argument("--pixel-resolution", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/rollouts_wall_demo.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tmp_dir = Path(tempfile.mkdtemp(prefix="wam_wall_demo_"))
    xml_path = build_xml(tmp_dir)

    env = MPCResidualEnv(xml_path=str(xml_path), episode_seconds=args.episode_seconds,
                          speed_range=(0.0, 0.0), lateral_range=(0.0, 0.0),
                          yaw_rate_range=(0.0, 0.0), seed=args.seed)
    cam = OnboardCamera(env.mujoco_go2.model, resolution=args.pixel_resolution)

    def collect_n(n, tag):
        all_s, all_a, all_ns, all_rgb, all_depth, episode_ids = [], [], [], [], [], []
        t0 = time.time()
        for ep in range(n):
            s, a, ns, rgb, depth = collect_episode(
                env, cam, rng, args.episode_seconds, args.approach_seconds,
                args.retreat_seconds, args.speed, macro_k=args.macro_k)
            all_s.append(s); all_a.append(a); all_ns.append(ns)
            all_rgb.append(rgb); all_depth.append(depth)
            episode_ids.append(np.full(len(s), ep, dtype=np.int32))
            print(f"[{tag}] episode {ep + 1}/{n}: {len(s)} transitions "
                  f"({time.time() - t0:.1f}s elapsed)")
        return (np.concatenate(all_s), np.concatenate(all_a), np.concatenate(all_ns),
                np.concatenate(all_rgb), np.concatenate(all_depth), np.concatenate(episode_ids))

    train_s, train_a, train_ns, train_rgb, train_depth, train_ep = collect_n(args.episodes, "train")
    val_s, val_a, val_ns, val_rgb, val_depth, val_ep = collect_n(args.val_episodes, "val")
    cam.close()

    print(f"Collected {len(train_s)} training transitions ({args.episodes} episodes), "
          f"{len(val_s)} validation transitions ({args.val_episodes} episodes).")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
              train_states=train_s, train_actions=train_a, train_next_states=train_ns,
              train_episode_ids=train_ep, train_pixels_rgb=train_rgb, train_pixels_depth=train_depth,
              val_states=val_s, val_actions=val_a, val_next_states=val_ns,
              val_episode_ids=val_ep, val_pixels_rgb=val_rgb, val_pixels_depth=val_depth,
              pixel_resolution=args.pixel_resolution, macro_k=args.macro_k,
              episode_seconds=args.episode_seconds, approach_seconds=args.approach_seconds,
              retreat_seconds=args.retreat_seconds, speed=args.speed)
    print(f"Saved dataset to {out_path}")


if __name__ == "__main__":
    main()
