"""Stage 3: obstacle-aware planning -- the CEM planner from Stage 2 (wam/planner.py), now
scoring candidate rollouts against a lidar-built occupancy map, not just the goal. Stage 3b
(wam/global_planner.py) adds an A* global planner over that same map, feeding the CEM a
nearby waypoint instead of the (possibly far-away, behind-a-corner) true goal directly --
this is what fixes the original Stage 3's corner-stall limitation, see README.

Everything sensing/mapping-related is reused COMPLETELY UNMODIFIED from the sibling
`slam_sim` project (its own Stage 1/2/6 work, not re-derived here): `slam.world.World` for
the ground-truth wall layout, `demo_stage6_go2_world.build_env_xml` for injecting those
walls into the real Go2's MuJoCo scene, `slam.mj_lidar.mj_lidar_scan` for the real ray-cast
lidar, and `slam.occupancy_grid.OccupancyGrid` for the log-odds map those scans build up
over time. This script adds the planning side: feeding `grid.occupied_points()` into
`CEMPlanner.plan()`'s `obstacle_points` argument every replan, and (new) running
`wam/global_planner.py`'s A* over that same grid to pick WHICH goal to hand the CEM.

Deliberately uses the REAL ground-truth pose (from MPCResidualEnv's own info) to build the
occupancy map, not slam_sim's ICP-corrected belief pose -- localization-under-drift is a
separate, already-solved problem over in slam_sim's own Stage 4/6; this stage's new ground
is PLANNING around what's been sensed, not re-demonstrating localization. One consequence
worth knowing: the map only contains what lidar has actually scanned so far, so the planner
can only avoid obstacles it has already sensed -- an obstacle just beyond lidar range or
behind the robot is invisible to it until a scan covers that direction, same limitation any
real lidar-mapping-then-planning stack has.

The default world is slam_sim's own `World.loop_corridor()`: a 12x8m rectangle with a 4x4m
island in the middle, open on all sides -- a straight line from one corner to the opposite
corner is blocked by the island, so a goal across the diagonal forces genuine routing around
it, not just goal-seeking in open space (Stage 2's eval_navigate.py trials).

Usage (needs go1_rl_walk/mpc's own venv for mujoco/torch, and slam_sim importable as a
sibling repo):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 navigate_wam_obstacles.py \
        --goal-x 10.5 --goal-y 6.5 --checkpoint checkpoints/world_model.pt \
        --record-robot media/navigate_wam_obstacles_robot.gif \
        --record-map media/navigate_wam_obstacles_map.gif
"""
import argparse
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import tempfile
from pathlib import Path

import mujoco as mj
import numpy as np
import torch

GO1_RL_WALK = Path(__file__).resolve().parent.parent / "go1_rl_walk"
SLAM_SIM = Path(__file__).resolve().parent.parent / "slam_sim"
sys.path.insert(0, str(GO1_RL_WALK / "mpc"))
sys.path.insert(0, str(GO1_RL_WALK))
sys.path.insert(0, str(SLAM_SIM))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mpc_residual_env import MPCResidualEnv                 # noqa: E402

from demo_stage6_go2_world import build_env_xml              # noqa: E402
from slam.world import World                                 # noqa: E402
from slam.occupancy_grid import OccupancyGrid                 # noqa: E402
from slam.mj_lidar import mj_lidar_scan                        # noqa: E402
from slam.viz import render_frame, save_gif                     # noqa: E402

from wam.model import WorldModel                            # noqa: E402
from wam.planner import CEMPlanner                            # noqa: E402
from wam.state import STATE_DIM, ACTION_DIM, extract_state    # noqa: E402
from wam.global_planner import plan_path                      # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-x", type=float, default=1.5)
    parser.add_argument("--start-y", type=float, default=1.5)
    parser.add_argument("--goal-x", type=float, default=10.5)
    parser.add_argument("--goal-y", type=float, default=6.5)
    parser.add_argument("--tolerance", type=float, default=0.3)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/world_model.pt")
    parser.add_argument("--horizon-steps", type=int, default=10)
    parser.add_argument("--replan-every", type=int, default=5, help="macro-steps between replans")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-elites", type=int, default=20)
    parser.add_argument("--n-iters", type=int, default=3)
    parser.add_argument("--w-obstacle", type=float, default=40.0)
    parser.add_argument("--w-collision", type=float, default=200.0)
    parser.add_argument("--safety-margin", type=float, default=0.5)
    parser.add_argument("--collision-margin", type=float, default=0.35)
    parser.add_argument("--disable-obstacle-cost", action="store_true",
                         help="plan with Stage 2's goal-only cost (ignores the occupancy "
                              "map entirely) -- for an intentional before/after comparison "
                              "against the obstacle-aware planner on the SAME island world.")
    parser.add_argument("--disable-global-planner", action="store_true",
                         help="keep the local CEM obstacle cost but drop the A* global "
                              "waypoint path -- reproduces Stage 3's original corner-stall "
                              "behavior, for an intentional before/after comparison.")
    parser.add_argument("--robot-radius", type=float, default=0.35,
                         help="inflation radius (m) used when A* decides which grid cells "
                              "are passable -- should roughly match CEMPlanner's own "
                              "collision-margin so the global path and local cost agree.")
    parser.add_argument("--lidar-beams", type=int, default=180)
    parser.add_argument("--lidar-every", type=int, default=10, help="control ticks between lidar scans")
    parser.add_argument("--occ-threshold", type=float, default=0.7)
    parser.add_argument("--terrain-amplitude-max", type=float, default=0.0,
                         help="Stage 4/5: (0, max) m peak-to-peak random bumps added to the "
                              "island world's ground, resampled at env reset -- 0.0 (default) "
                              "reproduces Stage 3/3b's flat-ground behavior exactly.")
    parser.add_argument("--terrain-footstep-planning", action="store_true",
                         help="make the classical MPC's OWN footstep planner terrain-aware "
                              "(go1_rl_walk's existing MPCResidualEnv flag, reused unmodified) "
                              "instead of assuming flat touchdown height -- independent of "
                              "the learned world model's own flat-vs-terrain checkpoints.")
    parser.add_argument("--max-accel", type=float, default=1.5)
    parser.add_argument("--max-yaw-accel", type=float, default=3.0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--record-robot", type=str, default=None)
    parser.add_argument("--record-map", type=str, default=None)
    parser.add_argument("--map-render-every", type=int, default=2, help="lidar scans between map-GIF frames")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WorldModel(STATE_DIM, ACTION_DIM, hidden=ckpt["hidden"], n_hidden=ckpt["n_hidden"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    world = World.loop_corridor()
    tmp_dir = Path(tempfile.mkdtemp(prefix="wam_obstacles_"))
    xml_path = build_env_xml(world.walls, tmp_dir)

    terrain_range = ((0.0, args.terrain_amplitude_max)
                      if args.terrain_amplitude_max > 0 else None)
    env = MPCResidualEnv(
        xml_path=str(xml_path), episode_seconds=args.seconds,
        speed_range=(0.0, 0.0), lateral_range=(0.0, 0.0), yaw_rate_range=(0.0, 0.0),
        terrain_amplitude_range=terrain_range,
        terrain_footstep_planning=args.terrain_footstep_planning, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    # reset() spawns the robot at its default (near-origin) position, where terrain is a
    # flat start pad by construction -- teleporting x/y to an arbitrary --start-x/-y leaves
    # qpos[2] (base height) still set for THAT pad's terrain height, not the new spot's. On
    # flat ground (terrain disabled) both are 0 so this is a no-op; with terrain enabled and
    # a start point off the pad, skipping this compensation spawns the robot either floating
    # above or embedded inside a terrain bump -- confirmed directly: caused an instant
    # fall/QP-failure within ~0.1m of every teleport, not a real planning or dynamics result.
    terrain_h_before = env._terrain_height(env.mujoco_go2.data.qpos[0], env.mujoco_go2.data.qpos[1])
    env.mujoco_go2.data.qpos[0:2] = [args.start_x, args.start_y]
    terrain_h_after = env._terrain_height(args.start_x, args.start_y)
    env.mujoco_go2.data.qpos[2] += float(terrain_h_after - terrain_h_before)
    mj.mj_forward(env.mujoco_go2.model, env.mujoco_go2.data)

    planner = CEMPlanner(
        model, speed_range=(-0.4, 0.8), lateral_range=(-0.3, 0.3), yaw_rate_range=(-1.0, 1.0),
        horizon_steps=args.horizon_steps, macro_dt=10 / 200.0,
        n_samples=args.n_samples, n_elites=args.n_elites, n_iters=args.n_iters,
        w_obstacle=args.w_obstacle, w_collision=args.w_collision,
        safety_margin=args.safety_margin, collision_margin=args.collision_margin)

    grid = OccupancyGrid(world.bounds, resolution=0.1)
    goal = np.array([args.goal_x, args.goal_y])

    max_cmd_delta = args.max_accel / 200.0
    max_yaw_delta = args.max_yaw_accel / 200.0
    max_steps = int(args.seconds * 200)
    x, y, yaw = args.start_x, args.start_y, 0.0
    cur_fwd, cur_lat, cur_yaw = 0.0, 0.0, 0.0
    target_action = np.zeros(3, dtype=np.float32)
    true_trail = []
    n_lidar_scans = 0
    reached, fell = False, False
    t_reached = None

    robot_renderer, robot_cam = None, None
    if args.record_robot:
        robot_cam = mj.MjvCamera()
        robot_cam.type = mj.mjtCamera.mjCAMERA_FREE
        robot_cam.lookat[:] = [world.bounds[1] / 2.0, world.bounds[3] / 2.0, 0.0]
        robot_cam.distance = 17.0
        robot_cam.elevation = -60
        robot_cam.azimuth = 90
        robot_renderer = mj.Renderer(env.mujoco_go2.model, height=480, width=640)

    map_frames = []
    robot_frames = []
    last_waypoints = None

    for step in range(1, max_steps + 1):
        dist = float(np.hypot(goal[0] - x, goal[1] - y))
        if dist <= args.tolerance:
            if not reached:
                reached, t_reached = True, step / 200.0

        if reached:
            target_action[:] = 0.0
        elif (step - 1) % (args.replan_every * 10) == 0:
            info_for_state = {"base_height": float(env.mujoco_go2.data.qpos[2])}
            state = extract_state(obs, info_for_state)
            obstacle_pts = (None if args.disable_obstacle_cost
                             else grid.occupied_points(threshold=args.occ_threshold))

            nav_goal = tuple(goal)
            last_waypoints = None
            if not args.disable_obstacle_cost and not args.disable_global_planner:
                waypoints = plan_path(grid, (x, y), tuple(goal),
                                       robot_radius=args.robot_radius,
                                       occ_threshold=args.occ_threshold)
                if waypoints:
                    nav_goal = waypoints[0]
                    last_waypoints = waypoints

            result = planner.plan(state, (x, y, yaw), nav_goal, warm_start=target_action,
                                   obstacle_points=obstacle_pts)
            target_action = result.action.numpy()
            if args.verbose:
                n_obs = 0 if obstacle_pts is None else len(obstacle_pts)
                n_wp = 0 if last_waypoints is None else len(last_waypoints)
                print(f"  t={step/200.0:5.1f}s pos=({x:+.2f},{y:+.2f}) yaw={yaw:+.2f} "
                      f"dist={dist:.2f} nav_goal=({nav_goal[0]:+.2f},{nav_goal[1]:+.2f}) "
                      f"n_waypoints={n_wp} action={target_action} n_obs={n_obs} "
                      f"cost={result.cost:.2f}")

        cur_fwd += np.clip(target_action[0] - cur_fwd, -max_cmd_delta, max_cmd_delta)
        cur_lat += np.clip(target_action[1] - cur_lat, -max_cmd_delta, max_cmd_delta)
        cur_yaw += np.clip(target_action[2] - cur_yaw, -max_yaw_delta, max_yaw_delta)
        env.set_command(cur_fwd, cur_lat, cur_yaw)

        obs, _, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
        x, y = float(info["base_pos"][0]), float(info["base_pos"][1])
        yaw = float(info["yaw"])
        true_trail.append([x, y])

        if step % args.lidar_every == 0:
            ranges, hit, angles = mj_lidar_scan(
                env.mujoco_go2.model, env.mujoco_go2.data, (x, y), yaw, n_beams=args.lidar_beams)
            grid.update((x, y, yaw), ranges, hit, angles)
            n_lidar_scans += 1
            if args.record_map and n_lidar_scans % args.map_render_every == 0:
                path_points = (np.array(last_waypoints) if last_waypoints is not None
                               else None)
                map_frames.append(render_frame(
                    world, (x, y, yaw), occupancy_grid=grid, scan_points=path_points,
                    true_trail=np.array(true_trail) if len(true_trail) > 1 else None,
                    status_text=f"goal ({goal[0]:.1f},{goal[1]:.1f})"))

        if args.record_robot and step % 16 == 0:
            robot_renderer.update_scene(env.mujoco_go2.data, camera=robot_cam)
            robot_frames.append(robot_renderer.render().copy())

        if terminated or truncated:
            fell = terminated
            break
        if reached and step > (t_reached * 200 + 200):
            break

    final_dist = float(np.hypot(goal[0] - x, goal[1] - y))
    if fell:
        print(f"FELL/QP-FAILED before reaching the goal (final distance {final_dist:.2f} m).")
    elif reached:
        print(f"Reached goal ({goal[0]:.2f}, {goal[1]:.2f}) in {t_reached:.1f}s, "
              f"final distance {final_dist:.2f} m, {n_lidar_scans} lidar scans.")
    else:
        print(f"Did NOT reach the goal within {args.seconds:.0f}s "
              f"(final distance {final_dist:.2f} m).")

    if robot_renderer is not None:
        robot_renderer.close()
    if args.record_map and map_frames:
        Path(args.record_map).parent.mkdir(parents=True, exist_ok=True)
        save_gif(map_frames, args.record_map, fps=6.0)
        print(f"Saved {len(map_frames)} frames to {args.record_map}")
    if args.record_robot and robot_frames:
        Path(args.record_robot).parent.mkdir(parents=True, exist_ok=True)
        save_gif(robot_frames, args.record_robot, fps=200.0 / 16)
        print(f"Saved {len(robot_frames)} frames to {args.record_robot}")


if __name__ == "__main__":
    main()
