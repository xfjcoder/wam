"""Stage 8: obstacle-aware planning (Stage 3/3b) against a MOVING obstacle, not just the
static island -- the planner still only ever reacts to what `slam.occupancy_grid.OccupancyGrid`
has sensed via lidar (same "only avoid what's been scanned" philosophy Stage 3 already
established), so this tests how well that sense-then-plan pipeline tracks something that
isn't where the last scan saw it, not an omniscient oracle avoidance.

Default scenario reuses eval_obstacles.py's own "no obstacle (sanity check)" start/goal pair
-- (1.5, 1.5) -> (1.5, 6.5), a straight line through open space with nothing static in the
way -- and adds exactly ONE moving obstacle (wam/moving_obstacle.py's `MovingObstacle`,
injected as a MuJoCo mocap body) centered on that line's midpoint, sweeping back and forth
across it. What was a trivial, obstacle-free sanity check becomes a genuine dynamic-avoidance
test: the straight line is clear most of the time, but periodically blocked.

Reuses everything else from navigate_wam_obstacles.py unchanged: the same lidar scan ->
occupancy grid -> A* waypoint -> CEM obstacle-cost pipeline, the same static island/outer
walls (`slam.world.World.loop_corridor()`), the same recording/marker conventions.

Usage (needs go1_rl_walk/mpc's own venv, and slam_sim importable as a sibling repo):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 navigate_wam_moving_obstacle.py \
        --checkpoint checkpoints/world_model.pt \
        --record-robot media/navigate_wam_moving_robot.gif \
        --record-map media/navigate_wam_moving_map.gif
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
    parser.add_argument("--replan-every", type=int, default=2,
                         help="macro-steps between replans -- shorter than Stage 3/3b's static-"
                              "obstacle default (5) so the planner reacts to the moving "
                              "obstacle's latest sensed position sooner; this is the replan "
                              "CADENCE, not the rollout horizon Stage 7 found shouldn't be "
                              "shortened")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-elites", type=int, default=20)
    parser.add_argument("--n-iters", type=int, default=3)
    parser.add_argument("--w-obstacle", type=float, default=40.0)
    parser.add_argument("--w-collision", type=float, default=200.0)
    parser.add_argument("--safety-margin", type=float, default=0.6)
    parser.add_argument("--collision-margin", type=float, default=0.35)
    parser.add_argument("--disable-obstacle-cost", action="store_true",
                         help="plan with Stage 2's goal-only cost (ignores the occupancy "
                              "map entirely) -- isolates whether the obstacle cost is what's "
                              "actually keeping the robot away from the moving obstacle, "
                              "same before/after discipline Stage 3 used for the static one.")
    parser.add_argument("--disable-global-planner", action="store_true")
    parser.add_argument("--disable-extrapolation", action="store_true",
                         help="Stage 9's velocity-extrapolated prediction of the moving "
                              "obstacle's future position, off -- falls back to Stage 8's "
                              "behavior of treating its last-sensed position as frozen for "
                              "the whole rollout horizon. Isolates whether extrapolation "
                              "itself helps, same before/after discipline as "
                              "--disable-obstacle-cost.")
    parser.add_argument("--robot-radius", type=float, default=0.35)
    parser.add_argument("--lidar-beams", type=int, default=180)
    parser.add_argument("--lidar-every", type=int, default=10, help="control ticks between lidar scans")
    parser.add_argument("--occ-threshold", type=float, default=0.7)
    parser.add_argument("--obstacle-center-x", type=float, default=None,
                         help="defaults to the midpoint of (start-x,start-y)/(goal-x,goal-y)")
    parser.add_argument("--obstacle-center-y", type=float, default=None)
    parser.add_argument("--obstacle-axis", choices=["x", "y"], default="x",
                         help="which world axis the obstacle sweeps along")
    parser.add_argument("--obstacle-amplitude", type=float, default=0.8)
    parser.add_argument("--obstacle-period", type=float, default=10.0, help="seconds per full sweep cycle")
    parser.add_argument("--obstacle-phase", type=float, default=0.0, help="radians, varies WHEN the "
                         "sweep crosses the robot's path relative to episode start")
    parser.add_argument("--obstacle-size", type=float, default=0.3, help="box half-width (m)")
    parser.add_argument("--max-accel", type=float, default=1.5)
    parser.add_argument("--max-yaw-accel", type=float, default=3.0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--record-robot", type=str, default=None)
    parser.add_argument("--record-map", type=str, default=None)
    parser.add_argument("--map-render-every", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Same bug class Stage 12 step 18 and Stage 13's own seeding-bug-fix commit found
    # independently (CEMPlanner's own torch.randn candidate sampling is never seeded by
    # anything else) -- found a third time while building navigate_wam_moving_obstacle_slam.py
    # and confirmed it also affects THIS script (and navigate_wam_multi_obstacle.py), never
    # ported back here when first fixed elsewhere. Means every Stage 8/9/10/11 multi-seed/phase
    # eval ever run through this script partly measured CEM sampling noise alongside whatever
    # was actually being tested -- see README Stage 8/9/10/11 re-verification note.
    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WorldModel(STATE_DIM, ACTION_DIM, hidden=ckpt["hidden"], n_hidden=ckpt["n_hidden"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    world = World.loop_corridor()
    center_x = (args.obstacle_center_x if args.obstacle_center_x is not None
                else (args.start_x + args.goal_x) / 2.0)
    center_y = (args.obstacle_center_y if args.obstacle_center_y is not None
                else (args.start_y + args.goal_y) / 2.0)
    obstacle = MovingObstacle(
        "moving_obstacle", (center_x, center_y), axis=args.obstacle_axis,
        amplitude=args.obstacle_amplitude, period=args.obstacle_period, phase=args.obstacle_phase,
        size=args.obstacle_size)

    tmp_dir = Path(tempfile.mkdtemp(prefix="wam_moving_obstacle_"))
    xml_path = build_env_xml(world.walls, tmp_dir)
    static_xml_text = xml_path.read_text()
    xml_text = inject_moving_obstacle(static_xml_text, obstacle)
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

    mocap_body_id = mj.mj_name2id(env.mujoco_go2.model, mj.mjtObj.mjOBJ_BODY, obstacle.name)
    mocap_id = int(env.mujoco_go2.model.body_mocapid[mocap_body_id])
    assert mocap_id >= 0, "moving_obstacle body did not come out as a mocap body"

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
        GOAL_RGBA = (0.2, 0.3, 0.9, 1.0)  # blue, distinct from the (red) moving obstacle geom
        WAYPOINT_RGBA = (0.9, 0.9, 0.1, 1.0)

    map_frames = []
    robot_frames = []
    last_waypoints = None
    last_dynamic_estimates = []

    for step in range(1, max_steps + 1):
        t_now = step / 200.0
        obs_x, obs_y = obstacle.position_at(t_now)
        env.mujoco_go2.data.mocap_pos[mocap_id, 0:2] = [obs_x, obs_y]

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
                # Stage 9: static safety net (walls/island) stays frozen as before; the
                # moving obstacle itself is pulled OUT of that frozen set and handed to the
                # planner separately so it can be velocity-extrapolated per rollout step.
                obstacle_pts = tracker.static_subset(all_occ_pts)
                dynamic_obstacles = last_dynamic_estimates
            else:
                # Stage 8's original behavior: every sensed point (including the moving
                # obstacle's) frozen at its last-sensed position for the whole horizon.
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
                                   dynamic_obstacle_radius=args.obstacle_size)
            target_action = result.action.numpy()
            if args.verbose:
                n_obs = 0 if obstacle_pts is None else len(obstacle_pts)
                n_wp = 0 if last_waypoints is None else len(last_waypoints)
                dyn_str = "none" if not dynamic_obstacles else " ".join(
                    f"[pos=({p[0]:+.2f},{p[1]:+.2f}) vel=({v[0]:+.2f},{v[1]:+.2f})]"
                    for p, v in dynamic_obstacles)
                print(f"  t={t_now:5.1f}s pos=({x:+.2f},{y:+.2f}) yaw={yaw:+.2f} "
                      f"dist={dist:.2f} obstacle=({obs_x:+.2f},{obs_y:+.2f}) "
                      f"n_waypoints={n_wp} action={target_action} n_obs={n_obs} "
                      f"dynamic={dyn_str} cost={result.cost:.2f}")

        cur_fwd += np.clip(target_action[0] - cur_fwd, -max_cmd_delta, max_cmd_delta)
        cur_lat += np.clip(target_action[1] - cur_lat, -max_cmd_delta, max_cmd_delta)
        cur_yaw += np.clip(target_action[2] - cur_yaw, -max_yaw_delta, max_yaw_delta)
        env.set_command(cur_fwd, cur_lat, cur_yaw)

        obs, _, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
        x, y = float(info["base_pos"][0]), float(info["base_pos"][1])
        yaw = float(info["yaw"])
        true_trail.append([x, y])
        min_dist_to_obstacle = min(min_dist_to_obstacle, float(np.hypot(x - obs_x, y - obs_y)))

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
                map_frames.append(render_frame(
                    world, (x, y, yaw), occupancy_grid=grid, scan_points=path_points,
                    true_trail=np.array(true_trail) if len(true_trail) > 1 else None,
                    status_text=f"goal ({goal[0]:.1f},{goal[1]:.1f})  obstacle "
                                f"({obs_x:.1f},{obs_y:.1f})"))

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
              f"min distance to moving obstacle {min_dist_to_obstacle:.2f} m).")
    elif reached:
        print(f"Reached goal ({goal[0]:.2f}, {goal[1]:.2f}) in {t_reached:.1f}s, "
              f"final distance {final_dist:.2f} m, {n_lidar_scans} lidar scans, "
              f"min distance to moving obstacle {min_dist_to_obstacle:.2f} m.")
    else:
        print(f"Did NOT reach the goal within {args.seconds:.0f}s "
              f"(final distance {final_dist:.2f} m, "
              f"min distance to moving obstacle {min_dist_to_obstacle:.2f} m).")

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
