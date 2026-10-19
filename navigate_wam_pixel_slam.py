"""Combining the two tracks Stage 12 and Stage 13 built independently: `navigate_wam_pixel.py`'s
camera-only obstacle sensing/A*/CEM planning, now driven by `navigate_wam_obstacles.py`'s
`--noisy-localization` belief pose instead of ground truth. The two tracks never touched before
this: Stage 12 always assumed perfect pose (obstacle-sensing was the question, not
localization); Stage 13 always used the lidar-built occupancy grid for both localization AND
obstacle/A* costing. This script keeps the two sensing roles split the way they'd realistically
be split on a robot carrying both a lidar and a camera: lidar + ICP for localization ONLY (a
private `lidar_grid`, reused completely unmodified from `navigate_wam_obstacles.py`, including
the confirmed `--icp-degeneracy-filter`/`--icp-reject-below-correspondences` fixes), and the
camera for obstacle-sensing/A*/CEM cost ONLY (a separate `cam_grid`, reused completely
unmodified from `navigate_wam_pixel.py`) -- but now updated and planned against `belief_pose`
instead of true pose when `--noisy-localization` is set, same as `navigate_wam_obstacles.py`'s
own `plan_pose` convention.

Both sensors are physically mounted on the real robot, so both are cast from the TRUE physical
pose regardless of localization uncertainty (same "never fed to any mapping/localization
algorithm, which would be cheating" rule `navigate_wam_obstacles.py` documents) -- only WHERE
each resulting scan/frame gets written into its grid, and what pose the planner/A* are told to
start from, switches to the belief pose.

The motivating question: Stage 12's mini-plan closed with closed-loop navigation never reaching
the goal on the island world's corner-stall task even under PERFECT (ground-truth) pose, with
the likely remaining cause narrowed to camera-derived map ACCURACY near corners (two planner-
tuning hypotheses already ruled out). That's a floor effect for this combination test on that
same task -- there's no room to measure "worse" when the baseline is already 0/5. So this script
is tested on an EASIER goal first (a straight line, no routing required -- the same "no
obstacle" pair `eval_localization.py` uses) where ground-truth-pose camera navigation is
expected to actually succeed, to see whether adding realistic SLAM noise on top of an
already-fragile camera-perception chain costs anything measurable even without the corner-stall
confound.

Usage (needs go1_rl_walk/mpc's own venv for mujoco/torch, and slam_sim importable as a sibling
repo):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 navigate_wam_pixel_slam.py \
        --goal-x 1.5 --goal-y 6.5 --noisy-localization --icp-degeneracy-filter \
        --record-robot media/navigate_wam_pixel_slam_robot.gif
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

from mpc_residual_env import MPCResidualEnv, DEFAULT_XML, _fix_meshdir_absolute  # noqa: E402
from demo_stage6_go2_world import inject_walls                                   # noqa: E402
from slam.world import World                                                     # noqa: E402
from slam.viz import render_frame, save_gif                                      # noqa: E402
from slam.occupancy_grid import OccupancyGrid                                    # noqa: E402
from slam.mj_lidar import mj_lidar_scan                                          # noqa: E402
from slam.scan_match import icp_2d, scan_to_local_points, transform_points       # noqa: E402

from wam.model import WorldModel                                   # noqa: E402
from wam.planner import CEMPlanner                                 # noqa: E402
from wam.state import STATE_DIM, ACTION_DIM, extract_state         # noqa: E402
from wam.camera import OnboardCamera, inject_onboard_camera        # noqa: E402
from wam.pixel_encoder import ConvAutoencoder, depth_to_tensor_01  # noqa: E402
from wam.pixel_dynamics import LatentDynamics                      # noqa: E402
from wam.pixel_cost import LatentCostHead                          # noqa: E402
from wam.global_planner import plan_path                           # noqa: E402


def build_xml(tmp_dir: Path, world: World) -> Path:
    xml_text = Path(DEFAULT_XML).read_text()
    xml_text = _fix_meshdir_absolute(xml_text, Path(DEFAULT_XML).resolve().parent)
    xml_text = inject_walls(xml_text, world.walls)
    xml_text = inject_onboard_camera(xml_text)
    out_path = tmp_dir / "go2_navigate_pixel_slam.xml"
    out_path.write_text(xml_text)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-x", type=float, default=1.5)
    parser.add_argument("--start-y", type=float, default=1.5)
    parser.add_argument("--goal-x", type=float, default=10.5)
    parser.add_argument("--goal-y", type=float, default=6.5)
    parser.add_argument("--tolerance", type=float, default=0.3)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/world_model.pt")
    parser.add_argument("--pixel-checkpoint", type=str,
                         default="checkpoints/pixel_world_model_wall_explore_depth.pt")
    parser.add_argument("--cost-checkpoint", type=str,
                         default="checkpoints/pixel_cost_head_explore_far.pt")
    parser.add_argument("--depth-clip", type=float, default=3.0)
    parser.add_argument("--horizon-steps", type=int, default=10)
    parser.add_argument("--pixel-horizon-steps", type=int, default=3)
    parser.add_argument("--replan-every", type=int, default=5, help="macro-steps between replans")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-elites", type=int, default=20)
    parser.add_argument("--n-iters", type=int, default=3)
    parser.add_argument("--w-pixel-obstacle", type=float, default=40.0)
    parser.add_argument("--w-pixel-collision", type=float, default=200.0)
    parser.add_argument("--pixel-safety-margin", type=float, default=0.6)
    parser.add_argument("--pixel-collision-margin", type=float, default=0.35)
    parser.add_argument("--disable-pixel-cost", action="store_true")
    parser.add_argument("--disable-global-planner", action="store_true")
    parser.add_argument("--grid-resolution", type=float, default=0.1)
    parser.add_argument("--robot-radius", type=float, default=0.35)
    parser.add_argument("--occ-threshold", type=float, default=0.7)
    parser.add_argument("--far-depth-hit-margin", type=float, default=0.1)
    parser.add_argument("--n-beams", type=int, default=7)
    parser.add_argument("--lidar-beams", type=int, default=180)
    parser.add_argument("--lidar-every", type=int, default=10, help="control ticks between lidar scans")
    parser.add_argument("--noisy-localization", action="store_true",
                         help="Stage 13 track feeding Stage 12's track: replace ground-truth "
                              "pose with a belief pose maintained via lidar + ICP, exactly as "
                              "navigate_wam_obstacles.py does -- the camera's occupancy grid, "
                              "the CEM planner, and A* all see ONLY this belief pose; true pose "
                              "is kept solely for scoring and as both sensors' physical origin. "
                              "Default off reproduces navigate_wam_pixel.py's exact original "
                              "ground-truth-pose behavior, given the SAME --pixel-checkpoint/"
                              "--cost-checkpoint (this script's OWN default --cost-checkpoint "
                              "is pixel_cost_head_explore_far.pt, the README's recommended best "
                              "checkpoint if this track is revisited, not navigate_wam_pixel.py's "
                              "own older default -- verified byte-identical output when both "
                              "scripts are pointed at the same checkpoint).")
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
    parser.add_argument("--map-render-every", type=int, default=2, help="lidar scans between map-GIF frames")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Same bug class Stage 12 step 18 and Stage 13's seeding-bug-fix commit both found
    # independently in the two scripts this one combines -- CEMPlanner's own torch.randn
    # candidate sampling needs its own seed, args.seed alone only controls env/numpy.
    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WorldModel(STATE_DIM, ACTION_DIM, hidden=ckpt["hidden"], n_hidden=ckpt["n_hidden"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pixel_ckpt = torch.load(args.pixel_checkpoint, map_location="cpu", weights_only=False)
    assert pixel_ckpt.get("modality", "rgb") == "depth", (
        "this script's cost head targets metric min/far-depth -- needs a depth-modality "
        f"pixel checkpoint (got modality={pixel_ckpt.get('modality')!r})")
    encoder = ConvAutoencoder(latent_dim=pixel_ckpt["latent_dim"], resolution=pixel_ckpt["resolution"],
                               channels=pixel_ckpt["channels"])
    encoder.load_state_dict(pixel_ckpt["encoder_state"])
    encoder.eval()
    pixel_dynamics = LatentDynamics(latent_dim=pixel_ckpt["latent_dim"],
                                     action_dim=pixel_ckpt["action_dim"])
    pixel_dynamics.load_state_dict(pixel_ckpt["dynamics_state"])
    pixel_dynamics.eval()

    cost_ckpt = torch.load(args.cost_checkpoint, map_location="cpu", weights_only=False)
    cost_head = LatentCostHead(latent_dim=cost_ckpt["latent_dim"])
    cost_head.load_state_dict(cost_ckpt["cost_head_state"])
    cost_head.eval()

    world = World.loop_corridor()
    tmp_dir = Path(tempfile.mkdtemp(prefix="wam_navigate_pixel_slam_"))
    xml_path = build_xml(tmp_dir, world)

    env = MPCResidualEnv(
        xml_path=str(xml_path), episode_seconds=args.seconds,
        speed_range=(0.0, 0.0), lateral_range=(0.0, 0.0), yaw_rate_range=(0.0, 0.0),
        seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    env.mujoco_go2.data.qpos[0:2] = [args.start_x, args.start_y]
    mj.mj_forward(env.mujoco_go2.model, env.mujoco_go2.data)

    cam = OnboardCamera(env.mujoco_go2.model, resolution=pixel_ckpt["resolution"])

    depth_cam_id = mj.mj_name2id(env.mujoco_go2.model, mj.mjtObj.mjOBJ_CAMERA, "depth_cam")
    half_fov_rad = np.radians(float(env.mujoco_go2.model.cam_fovy[depth_cam_id])) / 2.0
    frame_w = pixel_ckpt["resolution"]
    beam_col_groups = np.array_split(np.arange(frame_w), args.n_beams)
    beam_centers = np.array([g.mean() for g in beam_col_groups])
    beam_angles = (1.0 - 2.0 * beam_centers / (frame_w - 1)) * half_fov_rad

    planner = CEMPlanner(
        model, speed_range=(-0.4, 0.8), lateral_range=(-0.3, 0.3), yaw_rate_range=(-1.0, 1.0),
        horizon_steps=args.horizon_steps, macro_dt=10 / 200.0,
        n_samples=args.n_samples, n_elites=args.n_elites, n_iters=args.n_iters,
        w_obstacle=0.0, w_collision=0.0,
        pixel_dynamics=None if args.disable_pixel_cost else pixel_dynamics,
        pixel_cost_head=None if args.disable_pixel_cost else cost_head,
        w_pixel_obstacle=args.w_pixel_obstacle, w_pixel_collision=args.w_pixel_collision,
        pixel_safety_margin=args.pixel_safety_margin,
        pixel_collision_margin=args.pixel_collision_margin,
        pixel_horizon_steps=args.pixel_horizon_steps)

    # Two grids, two jobs -- the whole point of this script: a private lidar-built grid used
    # ONLY for ICP localization (navigate_wam_obstacles.py's role for its single grid), and a
    # separate camera-built grid used ONLY for obstacle-sensing/A*/CEM cost
    # (navigate_wam_pixel.py's role for its single grid). Neither grid is shared across roles.
    lidar_grid = OccupancyGrid(world.bounds, resolution=args.grid_resolution)
    cam_grid = OccupancyGrid(world.bounds, resolution=args.grid_resolution)
    goal = np.array([args.goal_x, args.goal_y])
    max_cmd_delta = args.max_accel / 200.0
    max_yaw_delta = args.max_yaw_accel / 200.0
    max_steps = int(args.seconds * 200)
    x, y, yaw = args.start_x, args.start_y, 0.0
    cur_fwd, cur_lat, cur_yaw = 0.0, 0.0, 0.0
    target_action = np.zeros(3, dtype=np.float32)
    true_trail = []
    reached, fell = False, False
    t_reached = None
    last_waypoints = None
    n_replans_mapped = 0
    n_lidar_scans = 0

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
        GOAL_RGBA = (0.9, 0.1, 0.1, 1.0)
        WAYPOINT_RGBA = (0.9, 0.9, 0.1, 1.0)
    robot_frames = []
    map_frames = []

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

            depth_frame = cam.depth(env.mujoco_go2.data)

            pixel_latent0 = None
            if not args.disable_pixel_cost:
                depth_t = depth_to_tensor_01(depth_frame[None, ...], clip=args.depth_clip)
                with torch.no_grad():
                    pixel_latent0 = encoder.encode(depth_t)[0]

            # Stage 13 fed into Stage 12: the camera's own occupancy map and A*/CEM start
            # point see whatever pose the robot BELIEVES it's at -- true (x, y, yaw) only when
            # localization is ground-truth (the original navigate_wam_pixel.py behavior).
            plan_pose = ((belief_x, belief_y, belief_yaw) if args.noisy_localization
                         else (x, y, yaw))

            nav_goal = tuple(goal)
            last_waypoints = None
            if not args.disable_global_planner:
                col_max = depth_frame.max(axis=0)
                beam_ranges = np.minimum(
                    np.array([col_max[g].max() for g in beam_col_groups]), args.depth_clip)
                beam_hits = beam_ranges < (args.depth_clip - args.far_depth_hit_margin)
                cam_grid.update(plan_pose, ranges=beam_ranges, hit=beam_hits, angles=beam_angles)
                n_replans_mapped += 1
                waypoints = plan_path(cam_grid, plan_pose[:2], tuple(goal),
                                       robot_radius=args.robot_radius,
                                       occ_threshold=args.occ_threshold)
                if waypoints:
                    nav_goal = waypoints[0]
                    last_waypoints = waypoints

            result = planner.plan(state, plan_pose, nav_goal, warm_start=target_action,
                                   pixel_latent0=pixel_latent0)
            target_action = result.action.numpy()
            if args.verbose:
                n_wp = 0 if last_waypoints is None else len(last_waypoints)
                loc_note = ""
                if args.noisy_localization:
                    drift = float(np.hypot(belief_x - x, belief_y - y))
                    loc_note = f" belief=({belief_x:+.2f},{belief_y:+.2f}) drift={drift:.2f}"
                print(f"  t={step/200.0:5.1f}s pos=({x:+.2f},{y:+.2f}) yaw={yaw:+.2f} "
                      f"dist={dist:.2f}{loc_note} nav_goal=({nav_goal[0]:+.2f},{nav_goal[1]:+.2f}) "
                      f"n_waypoints={n_wp} action={target_action} cost={result.cost:.2f} "
                      f"pixel_cost={result.pixel_cost:.2f}")

            if args.record_map:
                path_points = (np.array(last_waypoints) if last_waypoints is not None
                               else None)
                map_frames.append(render_frame(
                    world, (x, y, yaw), occupancy_grid=cam_grid, scan_points=path_points,
                    true_trail=np.array(true_trail) if len(true_trail) > 1 else None,
                    belief_trail=(np.array(belief_trail) if args.noisy_localization
                                  and len(belief_trail) > 1 else None),
                    odom_trail=(np.array(odom_trail) if args.noisy_localization
                                and len(odom_trail) > 1 else None),
                    status_text=f"goal ({goal[0]:.1f},{goal[1]:.1f})"))

        cur_fwd += np.clip(target_action[0] - cur_fwd, -max_cmd_delta, max_cmd_delta)
        cur_lat += np.clip(target_action[1] - cur_lat, -max_cmd_delta, max_cmd_delta)
        cur_yaw += np.clip(target_action[2] - cur_yaw, -max_yaw_delta, max_yaw_delta)
        env.set_command(cur_fwd, cur_lat, cur_yaw)

        obs, _, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
        x, y = float(info["base_pos"][0]), float(info["base_pos"][1])
        yaw = float(info["yaw"])
        true_trail.append([x, y])

        if args.noisy_localization and step % args.lidar_every == 0:
            # Physically mounted on the real robot -- always cast from TRUE pose, same rule
            # the camera capture above already follows.
            ranges, hit, angles = mj_lidar_scan(
                env.mujoco_go2.model, env.mujoco_go2.data, (x, y), yaw, n_beams=args.lidar_beams)

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
            map_pts = lidar_grid.occupied_points(threshold=args.occ_threshold)
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

            lidar_grid.update((belief_x, belief_y, belief_yaw), ranges, hit, angles)
            n_lidar_scans += 1

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

    cam.close()
    final_dist = float(np.hypot(goal[0] - x, goal[1] - y))
    if fell:
        print(f"FELL/QP-FAILED before reaching the goal (final distance {final_dist:.2f} m).")
    elif reached:
        print(f"Reached goal ({goal[0]:.2f}, {goal[1]:.2f}) in {t_reached:.1f}s, "
              f"final distance {final_dist:.2f} m.")
    else:
        print(f"Did NOT reach the goal within {args.seconds:.0f}s "
              f"(final distance {final_dist:.2f} m).")

    if args.noisy_localization:
        belief_drift = float(np.hypot(belief_x - x, belief_y - y))
        odom_drift = float(np.hypot(odom_x - x, odom_y - y))
        pct = 100.0 * n_icp_converged / n_icp_attempts if n_icp_attempts else 0.0
        print(f"Final localization drift: belief (ICP-corrected) = {belief_drift:.3f}m, "
              f"raw odometry = {odom_drift:.3f}m. ICP converged on {n_icp_converged}/"
              f"{n_icp_attempts} scans ({pct:.0f}%).")

    if robot_renderer is not None:
        robot_renderer.close()
    if args.record_robot and robot_frames:
        Path(args.record_robot).parent.mkdir(parents=True, exist_ok=True)
        save_gif(robot_frames, args.record_robot, fps=200.0 / 16)
        print(f"Saved {len(robot_frames)} frames to {args.record_robot}")
    if args.record_map and map_frames:
        Path(args.record_map).parent.mkdir(parents=True, exist_ok=True)
        save_gif(map_frames, args.record_map, fps=6.0)
        print(f"Saved {len(map_frames)} frames to {args.record_map}")
    if not args.disable_global_planner:
        print(f"Camera-derived map: {n_replans_mapped} replans x {args.n_beams} beams each "
              f"({n_replans_mapped * args.n_beams} total beam updates over the episode).")


if __name__ == "__main__":
    main()
