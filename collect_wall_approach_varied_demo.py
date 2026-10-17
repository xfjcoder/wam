"""Pixel-rendering mini-plan, step 12: a RICHER variant of step 8's wall-approach demo, to
check whether the method's win there was specific to one exact repeated trajectory or
generalizes across varied encounters with the same obstacle.

Step 8 used a single fixed start point (1.5, 4.0) approaching the SAME spot on the island's
west wall every single episode -- a strong, clean signal, but also the narrowest possible
dataset (one trajectory, replayed with only the usual physics/MPC-control noise). Here each
episode instead starts at a RANDOMLY SAMPLED y-position along that wall (still driving
straight at it, same approach/retreat oscillation, same safety guarantees) and a randomly
sampled speed -- so the wall band still grows/shrinks (same strong signal), but its exact
size-vs-time profile, approach angle to the corner, and rate of growth differ episode to
episode. If the model trained on step 8's data was secretly just memorizing one trajectory's
specific band-growth curve, this dataset would expose that; if it generalizes across the
varied start points the same way, that's a stronger result than step 8 alone.

y is sampled from [2.3, 5.7] -- strictly inside the island's y-span (2.0, 6.0) with a 0.3m
margin, so the approach always hits the island's west face squarely (not the gap beside it)
regardless of which y is drawn. x stays fixed at 1.5 (same 2.5m approach distance to the wall
for every episode, only the along-wall offset varies).

Same per-transition recording format as collect_wall_approach_demo.py / collect_data.py's
--record-pixels, so every existing training script runs UNCHANGED on this dataset.

Usage (needs go1_rl_walk/mpc's own venv for mujoco/torch, and slam_sim importable as a
sibling repo):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 collect_wall_approach_varied_demo.py \
        --episodes 10 --val-episodes 3 --out data/rollouts_wall_demo_varied.npz
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

START_X = 1.5
START_Y_RANGE = (2.3, 5.7)  # island's west wall spans y in (2.0, 6.0); 0.3m margin either side
SPEED_RANGE = (0.3, 0.4)


def build_xml(tmp_dir: Path) -> Path:
    world = World.loop_corridor()
    xml_text = Path(DEFAULT_XML).read_text()
    xml_text = _fix_meshdir_absolute(xml_text, Path(DEFAULT_XML).resolve().parent)
    xml_text = inject_walls(xml_text, world.walls)
    xml_text = inject_onboard_camera(xml_text)
    out_path = tmp_dir / "go2_wall_approach_varied_demo.xml"
    out_path.write_text(xml_text)
    return out_path


def collect_episode(env, cam, rng, episode_seconds, approach_seconds, retreat_seconds,
                     macro_k=10, max_accel=1.5):
    """Same scripted approach/retreat oscillation as collect_wall_approach_demo.py, but with
    a randomly sampled start_y (along the island's west wall) and speed each episode."""
    start_y = float(rng.uniform(*START_Y_RANGE))
    speed = float(rng.uniform(*SPEED_RANGE))

    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    env.mujoco_go2.data.qpos[0] = START_X
    env.mujoco_go2.data.qpos[1] = start_y

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
    ap.add_argument("--macro-k", type=int, default=10)
    ap.add_argument("--pixel-resolution", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/rollouts_wall_demo_varied.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tmp_dir = Path(tempfile.mkdtemp(prefix="wam_wall_demo_varied_"))
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
                args.retreat_seconds, macro_k=args.macro_k)
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
              retreat_seconds=args.retreat_seconds)
    print(f"Saved dataset to {out_path}")


if __name__ == "__main__":
    main()
