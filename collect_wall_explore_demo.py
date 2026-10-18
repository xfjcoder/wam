"""Pixel-rendering mini-plan, step 16: the data-diversity gap steps 10-14 explicitly left
open -- every pixel-WAM checkpoint so far was trained on ONE scripted scenario (straight-line
forward/backward approach toward a single wall, only start position/speed varied in step 12).
No turning, no lateral motion, no multiple walls/angles -- exactly the gap flagged before
wiring a cost head into an actual CEM planner, since CEM samples full (speed, lateral,
yaw_rate) candidates and a dynamics model that has never seen nonzero lateral/yaw_rate has no
business being trusted to roll them out.

This collects genuinely OPEN-ENDED exploration instead of a scripted approach/retreat: each
episode starts at a random free position AND HEADING inside slam_sim's own loop_corridor
world (same world navigate_wam_obstacles.py/collect_wall_approach_demo.py already use -- a
12x8m rectangle with a 4x4m island in the middle), then resamples a fresh random (speed,
lateral, yaw_rate) command every `--hold-ticks` physics ticks, same convention collect_data.py
already uses for its own (wall-less) flat-ground exploration -- just reusing that resampling
loop here, now with real walls in the scene and a start pose that puts the robot near one
often enough for the camera to actually see it. Command ranges match wam/planner.py's own
CEMPlanner defaults (speed (-0.4, 0.8), lateral (-0.3, 0.3), yaw_rate (-1.0, 1.0)) so the
learned dynamics model is asked to generalize only to the SAME action distribution a CEM
planner will actually query it with, not some separately-invented training range.

Start poses are rejection-sampled to be free (World.is_free, consistent with
navigate_wam_obstacles.py's own collision-free assumption) AND within
`--max-wall-distance` of some wall -- otherwise most of a 12x8m world is just open flat
ground with no wall in camera range, exactly the near-useless-signal case steps 1-7 already
diagnosed for the original flat-ground dataset. A real wall collision during an episode (the
injected walls are real MuJoCo geoms, not a non-colliding sim) just terminates that episode
early, same as a fall -- collect_episode discards the incomplete macro-step and moves on,
identical to collect_data.py's/collect_wall_approach_demo.py's own handling.

Step 18 adds `--corner-bias`: with that probability, an episode starts near one of the
island's own 4 corners (`island_corners`/`sample_start_pose_near_corner`) instead of anywhere
near-wall. Motivated directly by step 17's `navigate_wam_pixel.py` result -- a camera-only
local planner most plausibly stalls trying to coordinate an actual TURN around a corner, not
tracking a wall band growing/shrinking head-on the way the original approach/retreat scenario
(and most of step 16's general near-wall sampling) trained on. 0.0 (default) reproduces step
16's exact behavior.

Usage (needs go1_rl_walk/mpc's own venv for mujoco/torch, and slam_sim importable as a
sibling repo):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 collect_wall_explore_demo.py \
        --episodes 40 --val-episodes 10 --out data/rollouts_wall_explore.npz
    # step 18, corner-focused:
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 collect_wall_explore_demo.py \
        --episodes 50 --val-episodes 12 --corner-bias 0.6 \
        --out data/rollouts_wall_explore_corners.npz
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

import mujoco as mj                                                              # noqa: E402
from mpc_residual_env import MPCResidualEnv, DEFAULT_XML, _fix_meshdir_absolute  # noqa: E402
from demo_stage6_go2_world import inject_walls                                   # noqa: E402
from slam.world import World                                                     # noqa: E402

from wam.state import extract_state, extract_action, STATE_DIM, ACTION_DIM  # noqa: E402
from wam.camera import OnboardCamera, inject_onboard_camera                 # noqa: E402

SPEED_RANGE = (-0.4, 0.8)
LATERAL_RANGE = (-0.3, 0.3)
YAW_RATE_RANGE = (-1.0, 1.0)


def build_xml(tmp_dir: Path, world: World) -> Path:
    xml_text = Path(DEFAULT_XML).read_text()
    xml_text = _fix_meshdir_absolute(xml_text, Path(DEFAULT_XML).resolve().parent)
    xml_text = inject_walls(xml_text, world.walls)
    xml_text = inject_onboard_camera(xml_text)
    out_path = tmp_dir / "go2_wall_explore_demo.xml"
    out_path.write_text(xml_text)
    return out_path


def distance_to_nearest_wall(x: float, y: float, world: World) -> float:
    """Box-distance to the outer boundary or the island's rectangle -- same "one specific
    axis-aligned layout, not a general polygon" assumption World.is_free's own docstring
    already makes, reused here rather than re-deriving a general point-to-polygon distance."""
    xmin, xmax, ymin, ymax = world.bounds
    dist_outer = min(x - xmin, xmax - x, y - ymin, ymax - y)
    island = world.walls[4:8]
    ix1, iy1 = island[0, 0], island[0, 1]
    ix2, iy2 = island[1, 2], island[2, 3]
    dx = max(ix1 - x, 0.0, x - ix2)
    dy = max(iy1 - y, 0.0, y - iy2)
    dist_island = float(np.hypot(dx, dy)) if (dx > 0 or dy > 0) else 0.0
    return min(dist_outer, dist_island)


def sample_start_pose(world: World, rng, margin: float, max_wall_distance: float):
    xmin, xmax, ymin, ymax = world.bounds
    for _ in range(200):
        x = float(rng.uniform(xmin + margin, xmax - margin))
        y = float(rng.uniform(ymin + margin, ymax - margin))
        if not world.is_free(x, y, margin=margin):
            continue
        if distance_to_nearest_wall(x, y, world) > max_wall_distance:
            continue
        yaw = float(rng.uniform(-np.pi, np.pi))
        return x, y, yaw
    raise RuntimeError("could not sample a valid near-wall start pose in 200 tries")


def island_corners(world: World):
    """The island's own 4 corners -- same axis-aligned-layout assumption
    distance_to_nearest_wall/World.is_free already make. These are where
    navigate_wam_pixel.py's step 17 eval most likely stalled: a camera-only local planner
    has to coordinate a real TURN around one of these, not just track a shrinking/growing
    wall band head-on the way the original wall-approach scenario trained on."""
    island = world.walls[4:8]
    ix1, iy1 = island[0, 0], island[0, 1]
    ix2, iy2 = island[1, 2], island[2, 3]
    return [(ix1, iy1), (ix2, iy1), (ix2, iy2), (ix1, iy2)]


def sample_start_pose_near_corner(world: World, rng, margin: float, corner_radius: float):
    corners = island_corners(world)
    cx, cy = corners[int(rng.integers(0, len(corners)))]
    for _ in range(200):
        x = cx + float(rng.uniform(-corner_radius, corner_radius))
        y = cy + float(rng.uniform(-corner_radius, corner_radius))
        if world.is_free(x, y, margin=margin):
            yaw = float(rng.uniform(-np.pi, np.pi))
            return x, y, yaw
    raise RuntimeError("could not sample a valid near-corner start pose in 200 tries")


def teleport(env, x: float, y: float, yaw: float):
    data = env.mujoco_go2.data
    data.qpos[0] = x
    data.qpos[1] = y
    data.qpos[3] = np.cos(yaw / 2.0)
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = np.sin(yaw / 2.0)
    mj.mj_forward(env.mujoco_go2.model, data)


def collect_episode(env, cam, world, rng, episode_seconds, hold_ticks, macro_k,
                     start_margin, max_wall_distance, corner_bias=0.0, corner_radius=1.5,
                     max_accel=1.5, max_yaw_accel=3.0):
    """Same resampling-loop shape as collect_data.py's collect_episode, PLUS a random
    near-wall start pose/heading each episode -- the actual data-diversity fix. With
    probability `corner_bias`, starts near one of the island's own corners instead of
    anywhere near-wall (step 18: the camera-only planner most likely stalls coordinating a
    real turn around a corner, not tracking a wall band head-on -- more data of exactly
    that situation is the cheap thing to try before a bigger global-planner fix)."""
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    if corner_bias > 0 and rng.random() < corner_bias:
        x0, y0, yaw0 = sample_start_pose_near_corner(world, rng, start_margin, corner_radius)
    else:
        x0, y0, yaw0 = sample_start_pose(world, rng, start_margin, max_wall_distance)
    teleport(env, x0, y0, yaw0)

    max_ticks = int(episode_seconds * 200)
    dt = 1.0 / 200.0
    max_cmd_delta = max_accel * dt
    max_yaw_delta = max_yaw_accel * dt

    states, actions, next_states, pixels_rgb, pixels_depth = [], [], [], [], []
    state_t = action_t = rgb_t = depth_t = None
    ticks_in_macro = 0
    target_speed = float(rng.uniform(*SPEED_RANGE))
    target_lateral = float(rng.uniform(*LATERAL_RANGE))
    target_yaw_rate = float(rng.uniform(*YAW_RATE_RANGE))
    cur_speed, cur_lateral, cur_yaw_rate = target_speed, target_lateral, target_yaw_rate

    for tick in range(max_ticks):
        if tick % hold_ticks == 0 and tick > 0:
            target_speed = float(rng.uniform(*SPEED_RANGE))
            target_lateral = float(rng.uniform(*LATERAL_RANGE))
            target_yaw_rate = float(rng.uniform(*YAW_RATE_RANGE))

        cur_speed += np.clip(target_speed - cur_speed, -max_cmd_delta, max_cmd_delta)
        cur_lateral += np.clip(target_lateral - cur_lateral, -max_cmd_delta, max_cmd_delta)
        cur_yaw_rate += np.clip(target_yaw_rate - cur_yaw_rate, -max_yaw_delta, max_yaw_delta)
        env.set_command(cur_speed, cur_lateral, cur_yaw_rate)

        if ticks_in_macro == 0:
            info_for_state = {"base_height": float(env.mujoco_go2.data.qpos[2])}
            state_t = extract_state(obs, info_for_state)
            action_t = extract_action(target_speed, target_lateral, target_yaw_rate)
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
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--val-episodes", type=int, default=10)
    ap.add_argument("--episode-seconds", type=float, default=8.0)
    ap.add_argument("--hold-ticks", type=int, default=100)
    ap.add_argument("--macro-k", type=int, default=10)
    ap.add_argument("--start-margin", type=float, default=0.4)
    ap.add_argument("--max-wall-distance", type=float, default=4.0,
                     help="reject start poses farther than this from every wall -- keeps "
                          "the camera's depth channel informative (steps 1-7 found open flat "
                          "ground gives it almost nothing to predict).")
    ap.add_argument("--pixel-resolution", type=int, default=64)
    ap.add_argument("--corner-bias", type=float, default=0.0,
                     help="probability an episode starts near one of the island's own "
                          "corners instead of anywhere near-wall (step 18).")
    ap.add_argument("--corner-radius", type=float, default=1.5,
                     help="episode start is sampled within this many meters of the chosen "
                          "corner (box-uniform, then rejected if not free).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/rollouts_wall_explore.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    world = World.loop_corridor()
    tmp_dir = Path(tempfile.mkdtemp(prefix="wam_wall_explore_"))
    xml_path = build_xml(tmp_dir, world)

    env = MPCResidualEnv(xml_path=str(xml_path), episode_seconds=args.episode_seconds,
                          speed_range=SPEED_RANGE, lateral_range=LATERAL_RANGE,
                          yaw_rate_range=YAW_RATE_RANGE, seed=args.seed)
    cam = OnboardCamera(env.mujoco_go2.model, resolution=args.pixel_resolution)

    def collect_n(n, tag):
        all_s, all_a, all_ns, all_rgb, all_depth, episode_ids = [], [], [], [], [], []
        t0 = time.time()
        n_fell = 0
        for ep in range(n):
            s, a, ns, rgb, depth = collect_episode(
                env, cam, world, rng, args.episode_seconds, args.hold_ticks, args.macro_k,
                args.start_margin, args.max_wall_distance,
                corner_bias=args.corner_bias, corner_radius=args.corner_radius)
            if len(s) * args.macro_k < 0.9 * args.episode_seconds * 200:
                n_fell += 1
            all_s.append(s); all_a.append(a); all_ns.append(ns)
            all_rgb.append(rgb); all_depth.append(depth)
            episode_ids.append(np.full(len(s), ep, dtype=np.int32))
            print(f"[{tag}] episode {ep + 1}/{n}: {len(s)} transitions "
                  f"({time.time() - t0:.1f}s elapsed)")
        print(f"[{tag}] {n_fell}/{n} episodes ended early (fall/QP failure/wall collision)")
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
              episode_seconds=args.episode_seconds, hold_ticks=args.hold_ticks,
              speed_range=SPEED_RANGE, lateral_range=LATERAL_RANGE, yaw_rate_range=YAW_RATE_RANGE)
    print(f"Saved dataset to {out_path}")


if __name__ == "__main__":
    main()
