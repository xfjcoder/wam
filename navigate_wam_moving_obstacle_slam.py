"""A new combination: Stage 8/9's moving-obstacle tracker (`wam/obstacle_tracker.py`'s
`DynamicObstacleTracker`) driven by Stage 13's noisy/drifting SLAM belief pose
(`navigate_wam_obstacles.py`'s `--noisy-localization`), instead of ground truth. Neither track
had combined with the other before: Stage 8/9/10 always used perfect pose (the question was
whether the tracker could tell a moving obstacle apart from static walls at all); Stage 13
always ran on a world with no moving obstacles (the question was whether lidar+ICP could
localize at all).

The motivating hypothesis, specific to this combination (not a generic "does noise hurt"
question): `DynamicObstacleTracker` classifies a live occupied cell as static-vs-dynamic by
checking its distance to a ONE-TIME reference map built by scanning the environment's KNOWN
geometry from ground truth (`build_static_reference_points`, world-frame, unaffected by
localization). But under `--noisy-localization`, the LIVE occupancy grid is built from
`grid.update(belief_pose, ...)` -- if belief_pose drifts even a little from true pose, a
genuinely static wall cell gets written into the live grid at a slightly WRONG world-frame
location, possibly far enough from its true position to fall outside `match_radius` (0.25m) of
the static reference built from ground truth. That would misclassify a real wall cell as "new,
unexplained, must be the obstacle" -- a spurious "ghost" dynamic track, the exact failure mode
Stage 10 found and fixed for an UNRELATED cause (sparse static-reference vantage sampling).
This script tests whether BELIEF-DRIFT-induced ghosting is also a real failure mode, and (since
Stage 13 already confirmed `--icp-degeneracy-filter` reduces belief drift on degenerate
geometries) whether that same fix also helps here if it is.

Single lidar scan, single occupancy grid serves BOTH roles here (same as
`navigate_wam_obstacles.py`'s original single-grid design) -- localization (ICP matching
against `grid.occupied_points()`) and obstacle tracking/costing (tracker + CEM both read the
same grid), unlike Stage 14's pixel+SLAM combination which needed two separate grids for two
different SENSORS. Here it's one sensor (lidar) serving two roles, same sensor Stage 13 itself
already used for localization.

Everything else reused completely unmodified: `wam/moving_obstacle.py`'s `MovingObstacle`
(Stage 8), `wam/obstacle_tracker.py`'s `DynamicObstacleTracker`/`build_static_reference_points`
(Stage 9/10), the odometry-noise + ICP-correction block (Stage 13, including its confirmed
`--icp-degeneracy-filter`/`--icp-reject-below-correspondences` fixes), `wam/global_planner.py`'s
A* (Stage 3b).

New diagnostic this script adds, not present upstream: `max_tracks_seen`, the largest number of
SIMULTANEOUS dynamic tracks the tracker ever reported in one episode. There is only ever ONE
real moving obstacle in this scenario -- any value above 1 is a DIRECT, unambiguous signal of a
spurious ghost track, not something that has to be inferred from task outcome alone.

Usage (needs go1_rl_walk/mpc's own venv, and slam_sim importable as a sibling repo):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 navigate_wam_moving_obstacle_slam.py \
        --noisy-localization --icp-degeneracy-filter --verbose \
        --record-robot media/navigate_wam_moving_slam_robot.gif
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
from scipy.spatial import cKDTree

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
from slam.scan_match import icp_2d, scan_to_local_points, transform_points         # noqa: E402
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
    parser.add_argument("--safety-margin", type=float, default=0.6)
    parser.add_argument("--collision-margin", type=float, default=0.35)
    parser.add_argument("--disable-obstacle-cost", action="store_true")
    parser.add_argument("--disable-global-planner", action="store_true")
    parser.add_argument("--disable-extrapolation", action="store_true")
    parser.add_argument("--robot-radius", type=float, default=0.35)
    parser.add_argument("--lidar-beams", type=int, default=180)
    parser.add_argument("--lidar-every", type=int, default=10, help="control ticks between lidar scans")
    parser.add_argument("--occ-threshold", type=float, default=0.7)
    parser.add_argument("--obstacle-center-x", type=float, default=None)
    parser.add_argument("--obstacle-center-y", type=float, default=None)
    parser.add_argument("--obstacle-axis", choices=["x", "y"], default="x")
    parser.add_argument("--obstacle-amplitude", type=float, default=0.8)
    parser.add_argument("--obstacle-period", type=float, default=10.0)
    parser.add_argument("--obstacle-phase", type=float, default=0.0)
    parser.add_argument("--obstacle-size", type=float, default=0.3)
    parser.add_argument("--noisy-localization", action="store_true",
                         help="Stage 13 track feeding Stage 8/9's track: replace ground-truth "
                              "pose with a belief pose maintained via lidar + ICP, exactly as "
                              "navigate_wam_obstacles.py does -- the SAME single occupancy grid "
                              "(used for BOTH localization and obstacle tracking/costing here) "
                              "is built from this belief pose, and the tracker's live-scan side "
                              "sees only it; the static reference map stays ground-truth (an "
                              "assumed known floor plan, same as Stage 9/10). True pose is kept "
                              "solely for scoring and as the lidar's physical sensing origin. "
                              "Default off reproduces this script's exact original ground-truth "
                              "-pose behavior.")
    parser.add_argument("--odom-noise-v-frac", type=float, default=0.02)
    parser.add_argument("--odom-noise-omega-frac", type=float, default=0.02)
    parser.add_argument("--odom-noise-cross", type=float, default=0.01)
    parser.add_argument("--icp-max-correspondence-dist", type=float, default=0.5)
    parser.add_argument("--icp-min-correspondences", type=int, default=20)
    parser.add_argument("--icp-degeneracy-filter", action="store_true")
    parser.add_argument("--icp-confidence-margin", type=float, default=0.0)
    parser.add_argument("--icp-reject-below-correspondences", type=int, default=0)
    parser.add_argument("--max-accel", type=float, default=1.5)
    parser.add_argument("--max-yaw-accel", type=float, default=3.0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--record-robot", type=str, default=None)
    parser.add_argument("--record-map", type=str, default=None)
    parser.add_argument("--map-render-every", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--world-outer", type=float, nargs=2, default=(12.0, 8.0),
                         metavar=("W", "H"),
                         help="(width, height) of the outer rectangle passed to "
                              "World.loop_corridor() -- default reproduces the original "
                              "12x8m island world exactly. Used to escalate this combination "
                              "onto Stage 13's sparse/corridor stress worlds.")
    parser.add_argument("--world-island", type=float, nargs=4, default=(4.0, 2.0, 8.0, 6.0),
                         metavar=("X1", "Y1", "X2", "Y2"),
                         help="(x1, y1, x2, y2) of the island rectangle passed to "
                              "World.loop_corridor() -- default reproduces the original "
                              "4x4m island exactly.")
    args = parser.parse_args()

    # Same bug class Stage 12 step 18 / Stage 13's seeding-bug-fix commit both found
    # independently -- CEMPlanner's own torch.randn candidate sampling needs its own seed.
    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WorldModel(STATE_DIM, ACTION_DIM, hidden=ckpt["hidden"], n_hidden=ckpt["n_hidden"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    world = World.loop_corridor(outer=tuple(args.world_outer), island=tuple(args.world_island))
    center_x = (args.obstacle_center_x if args.obstacle_center_x is not None
                else (args.start_x + args.goal_x) / 2.0)
    center_y = (args.obstacle_center_y if args.obstacle_center_y is not None
                else (args.start_y + args.goal_y) / 2.0)
    obstacle = MovingObstacle(
        "moving_obstacle", (center_x, center_y), axis=args.obstacle_axis,
        amplitude=args.obstacle_amplitude, period=args.obstacle_period, phase=args.obstacle_phase,
        size=args.obstacle_size)

    tmp_dir = Path(tempfile.mkdtemp(prefix="wam_moving_obstacle_slam_"))
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
    max_tracks_seen = 0

    loc_rng = np.random.default_rng(args.seed)
    belief_x, belief_y, belief_yaw = args.start_x, args.start_y, 0.0
    odom_x, odom_y, odom_yaw = args.start_x, args.start_y, 0.0
    belief_trail, odom_trail = [], []
    n_icp_attempts, n_icp_converged = 0, 0

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
        GOAL_RGBA = (0.2, 0.3, 0.9, 1.0)
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
                obstacle_pts = tracker.static_subset(all_occ_pts)
                dynamic_obstacles = last_dynamic_estimates
            else:
                obstacle_pts = all_occ_pts

            # Stage 13 fed into Stage 8/9: the planner/A* get whatever pose the robot itself
            # BELIEVES it's at -- true (x, y, yaw) only when localization is ground-truth.
            plan_pose = ((belief_x, belief_y, belief_yaw) if args.noisy_localization
                         else (x, y, yaw))

            nav_goal = tuple(goal)
            last_waypoints = None
            if not args.disable_obstacle_cost and not args.disable_global_planner:
                waypoints = plan_path(grid, plan_pose[:2], tuple(goal),
                                       robot_radius=args.robot_radius,
                                       occ_threshold=args.occ_threshold)
                if waypoints:
                    nav_goal = waypoints[0]
                    last_waypoints = waypoints

            result = planner.plan(state, plan_pose, nav_goal, warm_start=target_action,
                                   obstacle_points=obstacle_pts, dynamic_obstacles=dynamic_obstacles,
                                   dynamic_obstacle_radius=args.obstacle_size)
            target_action = result.action.numpy()
            if args.verbose:
                n_obs = 0 if obstacle_pts is None else len(obstacle_pts)
                n_wp = 0 if last_waypoints is None else len(last_waypoints)
                dyn_str = "none" if not dynamic_obstacles else " ".join(
                    f"[pos=({p[0]:+.2f},{p[1]:+.2f}) vel=({v[0]:+.2f},{v[1]:+.2f})]"
                    for p, v in dynamic_obstacles)
                loc_note = ""
                if args.noisy_localization:
                    drift = float(np.hypot(belief_x - x, belief_y - y))
                    loc_note = f" belief=({belief_x:+.2f},{belief_y:+.2f}) drift={drift:.2f}"
                print(f"  t={t_now:5.1f}s pos=({x:+.2f},{y:+.2f}) yaw={yaw:+.2f} "
                      f"dist={dist:.2f}{loc_note} obstacle=({obs_x:+.2f},{obs_y:+.2f}) "
                      f"n_waypoints={n_wp} action={target_action} n_obs={n_obs} "
                      f"n_tracks={len(dynamic_obstacles)} dynamic={dyn_str} cost={result.cost:.2f}")

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
            # Physically mounted on the real robot -- always cast from TRUE pose, localization
            # uncertainty or not.
            ranges, hit, angles = mj_lidar_scan(
                env.mujoco_go2.model, env.mujoco_go2.data, (x, y), yaw, n_beams=args.lidar_beams)

            if args.noisy_localization:
                dt_loc = args.lidar_every / 200.0
                v_fwd, v_lat, yaw_rate = float(obs[0]), float(obs[1]), float(obs[5])
                v_fwd_n = v_fwd + loc_rng.normal(0.0, args.odom_noise_v_frac * abs(v_fwd) + 1e-4)
                v_lat_n = v_lat + loc_rng.normal(0.0, args.odom_noise_v_frac * abs(v_lat) + 1e-4)
                yaw_rate_n = yaw_rate + loc_rng.normal(
                    0.0, args.odom_noise_omega_frac * abs(yaw_rate)
                    + args.odom_noise_cross * (abs(v_fwd) + abs(v_lat)) + 1e-4)

                def _integrate(px, py, pyaw):
                    c, s = np.cos(pyaw), np.sin(pyaw)
                    px_next = px + (v_fwd_n * c - v_lat_n * s) * dt_loc
                    py_next = py + (v_fwd_n * s + v_lat_n * c) * dt_loc
                    pyaw_next = (pyaw + yaw_rate_n * dt_loc + np.pi) % (2 * np.pi) - np.pi
                    return px_next, py_next, pyaw_next

                odom_x, odom_y, odom_yaw = _integrate(odom_x, odom_y, odom_yaw)
                pred_x, pred_y, pred_yaw = _integrate(belief_x, belief_y, belief_yaw)

                local_pts = scan_to_local_points(ranges, hit, angles)
                map_pts = grid.occupied_points(threshold=args.occ_threshold)
                corrected, converged, n_valid_icp = icp_2d(
                    local_pts, map_pts, (pred_x, pred_y, pred_yaw),
                    max_correspondence_dist=args.icp_max_correspondence_dist,
                    min_correspondences=args.icp_min_correspondences)

                if (converged and args.icp_reject_below_correspondences > 0
                        and n_valid_icp < args.icp_reject_below_correspondences):
                    converged = False

                if converged and args.icp_degeneracy_filter:
                    world_pts = transform_points(local_pts, corrected)
                    dists, idx = cKDTree(map_pts).query(world_pts)
                    valid = dists < args.icp_max_correspondence_dist
                    matched = map_pts[idx[valid]]
                    centered = matched - matched.mean(axis=0)
                    eigvals, eigvecs = np.linalg.eigh(np.cov(centered.T))
                    eccentricity = eigvals[0] / (eigvals[1] + 1e-9)
                    v_minor, v_major = eigvecs[:, 0], eigvecs[:, 1]
                    delta = np.array([corrected[0] - pred_x, corrected[1] - pred_y])
                    delta_filtered = ((delta @ v_minor) * v_minor
                                      + eccentricity * (delta @ v_major) * v_major)
                    if args.icp_confidence_margin > 0:
                        count_confidence = float(np.clip(
                            (n_valid_icp - args.icp_min_correspondences)
                            / args.icp_confidence_margin, 0.0, 1.0))
                        delta_filtered = delta_filtered * count_confidence
                    belief_x = pred_x + delta_filtered[0]
                    belief_y = pred_y + delta_filtered[1]
                    belief_yaw = corrected[2]
                else:
                    belief_x, belief_y, belief_yaw = (
                        corrected if converged else (pred_x, pred_y, pred_yaw))
                n_icp_attempts += 1
                n_icp_converged += int(converged)
                belief_trail.append([belief_x, belief_y])
                odom_trail.append([odom_x, odom_y])

            map_pose = (belief_x, belief_y, belief_yaw) if args.noisy_localization else (x, y, yaw)
            grid.update(map_pose, ranges, hit, angles)
            n_lidar_scans += 1
            if tracker is not None:
                last_dynamic_estimates = tracker.update(
                    grid.occupied_points(threshold=args.occ_threshold), t_now)
                max_tracks_seen = max(max_tracks_seen, len(last_dynamic_estimates))
            if args.record_map and n_lidar_scans % args.map_render_every == 0:
                path_points = (np.array(last_waypoints) if last_waypoints is not None
                               else None)
                map_frames.append(render_frame(
                    world, (x, y, yaw), occupancy_grid=grid, scan_points=path_points,
                    true_trail=np.array(true_trail) if len(true_trail) > 1 else None,
                    belief_trail=(np.array(belief_trail) if args.noisy_localization
                                  and len(belief_trail) > 1 else None),
                    odom_trail=(np.array(odom_trail) if args.noisy_localization
                                and len(odom_trail) > 1 else None),
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

    if tracker is not None:
        print(f"Max simultaneous dynamic tracks seen: {max_tracks_seen} "
              f"(1 real obstacle -- anything above 1 is a spurious/ghost track).")

    if args.noisy_localization:
        belief_drift = float(np.hypot(belief_x - x, belief_y - y))
        odom_drift = float(np.hypot(odom_x - x, odom_y - y))
        pct = 100.0 * n_icp_converged / n_icp_attempts if n_icp_attempts else 0.0
        print(f"Final localization drift: belief (ICP-corrected) = {belief_drift:.3f}m, "
              f"raw odometry = {odom_drift:.3f}m. ICP converged on {n_icp_converged}/"
              f"{n_icp_attempts} scans ({pct:.0f}%).")

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
