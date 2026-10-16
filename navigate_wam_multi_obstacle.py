"""Stage 10: TWO independently-moving obstacles, not just one -- stress-tests
`wam/obstacle_tracker.py`'s `DynamicObstacleTracker`, which through Stage 9 only ever had to
track a single dynamic blob. With two obstacles sensed at once, a naive "average all dynamic
points into one centroid" tracker would blend two separate obstacles into one meaningless
phantom point between them; this checks the clustering + per-track data association added
for this stage actually keeps them apart and gives the planner two separate velocity-
extrapolated hazards instead.

Default scenario: the same (1.5, 1.5) -> (1.5, 6.5) straight line Stage 8/9 used, but now
with TWO obstacles at different heights along it (one encountered earlier in the traversal,
one later), each sweeping along x independently with its own phase -- so the robot generally
has to react to each one separately, not just once.

Reuses everything else from navigate_wam_moving_obstacle.py unchanged: the same lidar scan ->
occupancy grid -> A* waypoint -> CEM obstacle-cost pipeline, the same static island/outer
walls, the same recording/marker conventions, the same `DynamicObstacleTracker` class (now
generalized to however many tracks are live, not duplicated for this script).

Usage (needs go1_rl_walk/mpc's own venv, and slam_sim importable as a sibling repo):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 navigate_wam_multi_obstacle.py \
        --checkpoint checkpoints/world_model.pt \
        --record-robot media/navigate_wam_multi_robot.gif \
        --record-map media/navigate_wam_multi_map.gif
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
from wam.moving_obstacle import MovingObstacle, inject_moving_obstacle  # noqa: E402
from wam.obstacle_tracker import DynamicObstacleTracker, build_static_reference_points  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-x", type=float, default=1.5)
    parser.add_argument("--start-y", type=float, default=1.5)
    parser.add_argument("--goal-x", type=float, default=1.5)
    parser.add_argument("--goal-y", type=float, default=6.5)
    parser.add_argument("--tolerance", type=float, default=0.3)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/world_model.pt")
    parser.add_argument("--horizon-steps", type=int, default=10)
    parser.add_argument("--replan-every", type=int, default=2)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-elites", type=int, default=20)
    parser.add_argument("--n-iters", type=int, default=3)
    parser.add_argument("--w-obstacle", type=float, default=40.0)
    parser.add_argument("--w-collision", type=float, default=200.0)
    parser.add_argument("--w-smooth", type=float, default=0.0,
                         help="Stage 11's command-smoothness/hysteresis penalty -- penalizes "
                              "a candidate command for differing from the PREVIOUS plan()'s "
                              "chosen action. Investigated as a fix for indecisive lingering "
                              "observed with two dynamic obstacles, but repeated multi-seed "
                              "evals found no reliable net benefit (noise-level either way) "
                              "and a too-high weight (5.0) caused an actual timeout -- kept "
                              "available but OFF by default; see README Stage 11.")
    parser.add_argument("--safety-margin", type=float, default=0.6)
    parser.add_argument("--collision-margin", type=float, default=0.35)
    parser.add_argument("--disable-obstacle-cost", action="store_true")
    parser.add_argument("--disable-global-planner", action="store_true")
    parser.add_argument("--disable-extrapolation", action="store_true",
                         help="falls back to treating BOTH obstacles' last-sensed positions "
                              "as frozen for the whole rollout horizon (Stage 8's behavior), "
                              "instead of Stage 9/10's per-track velocity extrapolation.")
    parser.add_argument("--robot-radius", type=float, default=0.35)
    parser.add_argument("--lidar-beams", type=int, default=180)
    parser.add_argument("--lidar-every", type=int, default=10)
    parser.add_argument("--occ-threshold", type=float, default=0.7)

    parser.add_argument("--obstacle1-center-x", type=float, default=None)
    parser.add_argument("--obstacle1-center-y", type=float, default=3.0)
    parser.add_argument("--obstacle1-axis", choices=["x", "y"], default="x")
    parser.add_argument("--obstacle1-amplitude", type=float, default=0.8)
    parser.add_argument("--obstacle1-period", type=float, default=10.0)
    parser.add_argument("--obstacle1-phase", type=float, default=0.0)
    parser.add_argument("--obstacle1-size", type=float, default=0.3)

    parser.add_argument("--obstacle2-center-x", type=float, default=None)
    parser.add_argument("--obstacle2-center-y", type=float, default=5.5)
    parser.add_argument("--obstacle2-axis", choices=["x", "y"], default="x")
    parser.add_argument("--obstacle2-amplitude", type=float, default=0.8)
    parser.add_argument("--obstacle2-period", type=float, default=10.0)
    parser.add_argument("--obstacle2-phase", type=float, default=np.pi)
    parser.add_argument("--obstacle2-size", type=float, default=0.3)

    parser.add_argument("--max-accel", type=float, default=1.5)
    parser.add_argument("--max-yaw-accel", type=float, default=3.0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--record-robot", type=str, default=None)
    parser.add_argument("--record-map", type=str, default=None)
    parser.add_argument("--map-render-every", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WorldModel(STATE_DIM, ACTION_DIM, hidden=ckpt["hidden"], n_hidden=ckpt["n_hidden"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    world = World.loop_corridor()
    default_cx = (args.start_x + args.goal_x) / 2.0
    obstacles = [
        MovingObstacle(
            "moving_obstacle_1",
            (args.obstacle1_center_x if args.obstacle1_center_x is not None else default_cx,
             args.obstacle1_center_y),
            axis=args.obstacle1_axis, amplitude=args.obstacle1_amplitude,
            period=args.obstacle1_period, phase=args.obstacle1_phase, size=args.obstacle1_size),
        MovingObstacle(
            "moving_obstacle_2",
            (args.obstacle2_center_x if args.obstacle2_center_x is not None else default_cx,
             args.obstacle2_center_y),
            axis=args.obstacle2_axis, amplitude=args.obstacle2_amplitude,
            period=args.obstacle2_period, phase=args.obstacle2_phase, size=args.obstacle2_size),
    ]

    tmp_dir = Path(tempfile.mkdtemp(prefix="wam_multi_obstacle_"))
    xml_path = build_env_xml(world.walls, tmp_dir)
    static_xml_text = xml_path.read_text()
    xml_text = static_xml_text
    for obstacle in obstacles:
        xml_text = inject_moving_obstacle(xml_text, obstacle)
    xml_path.write_text(xml_text)

    tracker = None
    if not args.disable_obstacle_cost and not args.disable_extrapolation:
        static_model = mj.MjModel.from_xml_string(static_xml_text)
        static_data = mj.MjData(static_model)
        mj.mj_forward(static_model, static_data)
        static_points = build_static_reference_points(static_model, static_data, world)
        tracker = DynamicObstacleTracker(static_points, match_radius=0.25)

    env = MPCResidualEnv(
        xml_path=str(xml_path), episode_seconds=args.seconds,
        speed_range=(0.0, 0.0), lateral_range=(0.0, 0.0), yaw_rate_range=(0.0, 0.0),
        seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    env.mujoco_go2.data.qpos[0:2] = [args.start_x, args.start_y]
    mj.mj_forward(env.mujoco_go2.model, env.mujoco_go2.data)

    mocap_ids = []
    for obstacle in obstacles:
        body_id = mj.mj_name2id(env.mujoco_go2.model, mj.mjtObj.mjOBJ_BODY, obstacle.name)
        mocap_id = int(env.mujoco_go2.model.body_mocapid[body_id])
        assert mocap_id >= 0, f"{obstacle.name} body did not come out as a mocap body"
        mocap_ids.append(mocap_id)

    # Shared obstacle-footprint radius for the dynamic cost term -- both obstacles are the
    # same size by default; if they differ, use the larger as a conservative shared margin.
    shared_obstacle_radius = max(args.obstacle1_size, args.obstacle2_size)

    planner = CEMPlanner(
        model, speed_range=(-0.4, 0.8), lateral_range=(-0.3, 0.3), yaw_rate_range=(-1.0, 1.0),
        horizon_steps=args.horizon_steps, macro_dt=10 / 200.0,
        n_samples=args.n_samples, n_elites=args.n_elites, n_iters=args.n_iters,
        w_obstacle=args.w_obstacle, w_collision=args.w_collision,
        safety_margin=args.safety_margin, collision_margin=args.collision_margin,
        w_smooth=args.w_smooth)

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
    min_dist_to_obstacle = float("inf")

    def _add_marker(scene, pos_xy, rgba, size=0.15):
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(
            g, mj.mjtGeom.mjGEOM_SPHERE, np.array([size, 0, 0]),
            np.array([pos_xy[0], pos_xy[1], 0.05]), np.eye(3).flatten(),
            np.array(rgba, dtype=np.float32))
        scene.ngeom += 1

    robot_renderer, robot_cam = None, None
    if args.record_robot:
        robot_cam = mj.MjvCamera()
        robot_cam.type = mj.mjtCamera.mjCAMERA_FREE
        robot_cam.lookat[:] = [world.bounds[1] / 2.0, world.bounds[3] / 2.0, 0.0]
        robot_cam.distance = 17.0
        robot_cam.elevation = -60
        robot_cam.azimuth = 90
        robot_renderer = mj.Renderer(env.mujoco_go2.model, height=480, width=640)
        START_RGBA = (0.1, 0.9, 0.1, 1.0)
        GOAL_RGBA = (0.2, 0.3, 0.9, 1.0)  # blue, distinct from the (red) moving obstacles
        WAYPOINT_RGBA = (0.9, 0.9, 0.1, 1.0)

    map_frames = []
    robot_frames = []
    last_waypoints = None
    last_dynamic_estimates = []

    for step in range(1, max_steps + 1):
        t_now = step / 200.0
        obs_positions = [obstacle.position_at(t_now) for obstacle in obstacles]
        for mocap_id, (ox, oy) in zip(mocap_ids, obs_positions):
            env.mujoco_go2.data.mocap_pos[mocap_id, 0:2] = [ox, oy]

        dist = float(np.hypot(goal[0] - x, goal[1] - y))
        if dist <= args.tolerance:
            if not reached:
                reached, t_reached = True, step / 200.0

        if reached:
            target_action[:] = 0.0
        elif (step - 1) % (args.replan_every * 10) == 0:
            info_for_state = {"base_height": float(env.mujoco_go2.data.qpos[2])}
            state = extract_state(obs, info_for_state)
            all_occ_pts = (None if args.disable_obstacle_cost
                            else grid.occupied_points(threshold=args.occ_threshold))
            dynamic_obstacles = []
            if all_occ_pts is None:
                obstacle_pts = None
            elif tracker is not None:
                obstacle_pts = tracker.static_subset(all_occ_pts)
                dynamic_obstacles = last_dynamic_estimates
            else:
                obstacle_pts = all_occ_pts

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
                                   obstacle_points=obstacle_pts, dynamic_obstacles=dynamic_obstacles,
                                   dynamic_obstacle_radius=shared_obstacle_radius)
            target_action = result.action.numpy()
            if args.verbose:
                n_obs = 0 if obstacle_pts is None else len(obstacle_pts)
                n_wp = 0 if last_waypoints is None else len(last_waypoints)
                dyn_str = "none" if not dynamic_obstacles else " ".join(
                    f"[pos=({p[0]:+.2f},{p[1]:+.2f}) vel=({v[0]:+.2f},{v[1]:+.2f})]"
                    for p, v in dynamic_obstacles)
                obs_str = " ".join(f"({ox:+.2f},{oy:+.2f})" for ox, oy in obs_positions)
                print(f"  t={t_now:5.1f}s pos=({x:+.2f},{y:+.2f}) yaw={yaw:+.2f} "
                      f"dist={dist:.2f} obstacles={obs_str} n_waypoints={n_wp} "
                      f"action={target_action} n_obs={n_obs} n_tracks={len(dynamic_obstacles)} "
                      f"dynamic={dyn_str} cost={result.cost:.2f}")

        cur_fwd += np.clip(target_action[0] - cur_fwd, -max_cmd_delta, max_cmd_delta)
        cur_lat += np.clip(target_action[1] - cur_lat, -max_cmd_delta, max_cmd_delta)
        cur_yaw += np.clip(target_action[2] - cur_yaw, -max_yaw_delta, max_yaw_delta)
        env.set_command(cur_fwd, cur_lat, cur_yaw)

        obs, _, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
        x, y = float(info["base_pos"][0]), float(info["base_pos"][1])
        yaw = float(info["yaw"])
        true_trail.append([x, y])
        for ox, oy in obs_positions:
            min_dist_to_obstacle = min(min_dist_to_obstacle, float(np.hypot(x - ox, y - oy)))

        if step % args.lidar_every == 0:
            ranges, hit, angles = mj_lidar_scan(
                env.mujoco_go2.model, env.mujoco_go2.data, (x, y), yaw, n_beams=args.lidar_beams)
            grid.update((x, y, yaw), ranges, hit, angles)
            n_lidar_scans += 1
            if tracker is not None:
                last_dynamic_estimates = tracker.update(
                    grid.occupied_points(threshold=args.occ_threshold), t_now)
            if args.record_map and n_lidar_scans % args.map_render_every == 0:
                path_points = (np.array(last_waypoints) if last_waypoints is not None
                               else None)
                obs_str = "  ".join(f"({ox:.1f},{oy:.1f})" for ox, oy in obs_positions)
                map_frames.append(render_frame(
                    world, (x, y, yaw), occupancy_grid=grid, scan_points=path_points,
                    true_trail=np.array(true_trail) if len(true_trail) > 1 else None,
                    status_text=f"goal ({goal[0]:.1f},{goal[1]:.1f})  obstacles {obs_str}"))

        if args.record_robot and step % 16 == 0:
            robot_renderer.update_scene(env.mujoco_go2.data, camera=robot_cam)
            _add_marker(robot_renderer.scene, (args.start_x, args.start_y), START_RGBA)
            _add_marker(robot_renderer.scene, goal, GOAL_RGBA)
            if last_waypoints is not None:
                for wp in last_waypoints[:-1]:
                    _add_marker(robot_renderer.scene, wp, WAYPOINT_RGBA, size=0.12)
            robot_frames.append(robot_renderer.render().copy())

        if terminated or truncated:
            fell = terminated
            break
        if reached and step > (t_reached * 200 + 200):
            break

    final_dist = float(np.hypot(goal[0] - x, goal[1] - y))
    if fell:
        print(f"FELL/QP-FAILED before reaching the goal (final distance {final_dist:.2f} m, "
              f"min distance to moving obstacles {min_dist_to_obstacle:.2f} m).")
    elif reached:
        print(f"Reached goal ({goal[0]:.2f}, {goal[1]:.2f}) in {t_reached:.1f}s, "
              f"final distance {final_dist:.2f} m, {n_lidar_scans} lidar scans, "
              f"min distance to moving obstacles {min_dist_to_obstacle:.2f} m.")
    else:
        print(f"Did NOT reach the goal within {args.seconds:.0f}s "
              f"(final distance {final_dist:.2f} m, "
              f"min distance to moving obstacles {min_dist_to_obstacle:.2f} m).")

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
