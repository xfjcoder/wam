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

By default still uses the REAL ground-truth pose (from MPCResidualEnv's own info) to build
the occupancy map -- the original scope here was PLANNING around what's been sensed, not
localization, so this deliberately deferred to slam_sim's own already-solved Stage 4/6
localization work rather than re-demonstrating it.

Stage 13 closes that gap: `--noisy-localization` replaces ground truth with a `belief_pose`
maintained the SAME way slam_sim's Stage 4 does for its own toy diff-drive robot, now wired
onto the real MuJoCo Go2 -- odometry noise injected into the robot's OWN executed body-frame
velocity (`obs`'s v_body/w_body, not a command setpoint) via the exact Thrun/Burgard/Fox
noise model `slam.robot.DiffDriveRobot` already uses (reused verbatim, not re-tuned, extended
with the SAME v-noise formula for lateral velocity too -- Go2 can strafe, unlike a pure
diff-drive robot), corrected every lidar scan by `slam.scan_match.icp_2d` against the map
built so far (also reused completely unmodified). `belief_pose` -- not true pose -- is what
feeds the occupancy grid, the CEM planner's current-state pose, and A*'s start point; true
pose is kept ONLY for scoring (goal-reached distance, final_dist) and as the lidar's own
physical sensing origin -- the same true/belief separation slam_sim's own docs insist on
("never fed to any mapping/localization algorithm, which would be cheating"). One consequence
worth knowing (independent of localization): the map only contains what lidar has actually
scanned so far, so the planner can only avoid obstacles it has already sensed -- an obstacle
just beyond lidar range or behind the robot is invisible to it until a scan covers that
direction, same limitation any real lidar-mapping-then-planning stack has.

The default world is slam_sim's own `World.loop_corridor()`: a 12x8m rectangle with a 4x4m
island in the middle, open on all sides -- a straight line from one corner to the opposite
corner is blocked by the island, so a goal across the diagonal forces genuine routing around
it, not just goal-seeking in open space (Stage 2's eval_navigate.py trials). `--world-outer`/
`--world-island` (default unchanged) build other `World.loop_corridor()` sizes, used by
`eval_localization_sparse.py` to stress-test Stage 13 on sparser/more degenerate geometries.

That stress test confirmed a real failure mode: on a long straight corridor, icp_2d can
report converged=True with a low, stable residual while still applying a WRONG along-corridor
correction, because a straight wall's matched points constrain the perpendicular direction but
not the direction parallel to the wall (the textbook point-cloud-registration aliasing case).
`--icp-degeneracy-filter` is the fix: it re-examines the correspondence set ICP's returned
pose implies, and down-weights the correction's component along the matched points' dominant
(long) axis by that axis's own eigenvalue ratio -- full trust when the match is isotropic
(corners, the default island world), near-zero trust along a collinear wall. Default off
reproduces the original `--noisy-localization` behavior exactly.

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
    parser.add_argument("--noisy-localization", action="store_true",
                         help="Stage 13: replace ground-truth pose with a belief pose "
                              "maintained via noisy odometry + ICP scan-matching against the "
                              "map built so far (slam_sim's own Stage 4 approach, ported onto "
                              "the real Go2) -- the occupancy grid, CEM planner, and A* all "
                              "see ONLY this belief pose; true pose is kept solely for scoring "
                              "(goal-reached distance, final_dist) and as the lidar's physical "
                              "sensing origin. Default off reproduces this script's exact "
                              "original ground-truth-pose behavior.")
    parser.add_argument("--odom-noise-v-frac", type=float, default=0.02,
                         help="odometry noise model (Thrun/Burgard/Fox, same as "
                              "slam.robot.DiffDriveRobot's default): linear-velocity noise "
                              "std scales with |v| by this fraction (applied to both forward "
                              "and lateral body velocity).")
    parser.add_argument("--odom-noise-omega-frac", type=float, default=0.02,
                         help="angular-velocity noise std scales with |yaw_rate| by this "
                              "fraction.")
    parser.add_argument("--odom-noise-cross", type=float, default=0.01,
                         help="angular-velocity noise ALSO scales with |v| by this fraction "
                              "(straight-line driving still accumulates heading drift, "
                              "matching real wheel/leg-odometry behavior).")
    parser.add_argument("--icp-max-correspondence-dist", type=float, default=0.5)
    parser.add_argument("--icp-min-correspondences", type=int, default=20)
    parser.add_argument("--icp-degeneracy-filter", action="store_true",
                         help="Stage 13 addendum's confirmed fix target: icp_2d can report "
                              "converged=True with a near-zero residual while still being "
                              "wrong, when the matched map points are nearly collinear (e.g. "
                              "a long straight corridor wall) -- the correction is genuinely "
                              "unconstrained along that line's direction, not just noisy. "
                              "When set, re-examines the correspondence set ICP's returned "
                              "pose implies, and down-weights the component of the position "
                              "correction along the matched points' dominant (long) axis by "
                              "its eigenvalue ratio to the perpendicular axis -- full trust "
                              "when the match is isotropic (corners, varied wall angles, the "
                              "default island world), near-zero trust along a collinear "
                              "wall's own direction (falls back to the odometry prediction "
                              "for THAT axis only, same as an unconverged scan already does "
                              "for the whole correction). Default off reproduces the original "
                              "--noisy-localization behavior exactly.")
    parser.add_argument("--icp-confidence-margin", type=float, default=0.0,
                         help="Only used together with --icp-degeneracy-filter (0, the "
                              "default, disables it): confirmed directly that per-scan "
                              "correction magnitude grows noticeably as the correspondence "
                              "count (n_valid) shrinks toward --icp-min-correspondences's own "
                              "floor, in a sparse map independent of the geometric-degeneracy "
                              "check above (a small but non-collinear correspondence set can "
                              "still be statistically unreliable). Scales the WHOLE correction "
                              "by how far n_valid is above that floor, ramping to full trust "
                              "this many correspondences above it -- e.g. 50 means a 70-point "
                              "match (20+50) is fully trusted, a 20-point match is not trusted "
                              "at all (falls back to the odometry prediction).")
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
    parser.add_argument("--world-outer", type=float, nargs=2, default=(12.0, 8.0),
                         metavar=("W", "H"),
                         help="(width, height) of the outer rectangle passed to "
                              "World.loop_corridor() -- default reproduces the original "
                              "12x8m island world exactly. Used to build sparser/more "
                              "elongated (aliasing-prone) worlds for Stage 13 stress tests "
                              "without touching the default scenario.")
    parser.add_argument("--world-island", type=float, nargs=4, default=(4.0, 2.0, 8.0, 6.0),
                         metavar=("X1", "Y1", "X2", "Y2"),
                         help="(x1, y1, x2, y2) of the island rectangle passed to "
                              "World.loop_corridor() -- default reproduces the original "
                              "4x4m island exactly.")
    args = parser.parse_args()

    # Stage 12 step 18 found and fixed this exact bug in navigate_wam_pixel.py (CEMPlanner's
    # own torch.randn candidate sampling was never seeded anywhere -- confirmed there by
    # re-running an identical --seed and getting a DIFFERENT final distance) but it was never
    # ported back here, even though CEMPlanner (wam/planner.py) is the SAME shared module both
    # scripts use. Found again independently while cross-checking two --icp-degeneracy-filter
    # eval runs that should have been identical for a corridor pair whose n_valid never drops
    # near --icp-confidence-margin's threshold (so the margin couldn't explain the difference)
    # and weren't -- confirming every multi-seed eval ever run through this script (Stage 3b
    # onward) partly measured CEM sampling noise alongside whatever was actually being tested.
    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WorldModel(STATE_DIM, ACTION_DIM, hidden=ckpt["hidden"], n_hidden=ckpt["n_hidden"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    world = World.loop_corridor(outer=tuple(args.world_outer), island=tuple(args.world_island))
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

    # Stage 13: belief_pose is what the planner/map actually see when --noisy-localization is
    # on; odom_pose is the UNCORRECTED noisy trail, kept only to show ICP's correction is
    # doing something (same 3-pose convention slam_sim's own demo_stage4_slam.py uses). Both
    # start at the true start pose -- "known start pose" is the standard SLAM assumption, not
    # a shortcut (a robot deployed at a known charging-dock/start marker knows its own pose
    # exactly at t=0; drift only accumulates from there).
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

            # Stage 13: the planner/A* get whatever pose the robot itself BELIEVES it's at --
            # true (x, y, yaw) only when localization is ground-truth (the original behavior).
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
                                   obstacle_points=obstacle_pts)
            target_action = result.action.numpy()
            if args.verbose:
                n_obs = 0 if obstacle_pts is None else len(obstacle_pts)
                n_wp = 0 if last_waypoints is None else len(last_waypoints)
                loc_note = ""
                if args.noisy_localization:
                    drift = float(np.hypot(belief_x - x, belief_y - y))
                    loc_note = f" belief=({belief_x:+.2f},{belief_y:+.2f}) drift={drift:.2f}"
                print(f"  t={step/200.0:5.1f}s pos=({x:+.2f},{y:+.2f}) yaw={yaw:+.2f} "
                      f"dist={dist:.2f}{loc_note} nav_goal=({nav_goal[0]:+.2f},{nav_goal[1]:+.2f}) "
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
            # Sensor is physically mounted on the real robot -- always cast from TRUE pose,
            # localization uncertainty or not (this is ground truth about where the lidar
            # actually is, not a belief the algorithm gets to use for anything else).
            ranges, hit, angles = mj_lidar_scan(
                env.mujoco_go2.model, env.mujoco_go2.data, (x, y), yaw, n_beams=args.lidar_beams)

            if args.noisy_localization:
                dt_loc = args.lidar_every / 200.0
                v_fwd, v_lat, yaw_rate = float(obs[0]), float(obs[1]), float(obs[5])
                # Thrun/Burgard/Fox odometry noise model, identical to
                # slam.robot.DiffDriveRobot's default (0.02, 0.02, 0.01) -- extended with the
                # SAME v-noise formula for lateral velocity (Go2 can strafe; a pure
                # diff-drive robot can't, so DiffDriveRobot never needed this term).
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

                if converged and args.icp_degeneracy_filter:
                    # icp_2d's own residual/convergence check has no way to tell "confidently
                    # right" from "confidently wrong along an unconstrained axis" -- a long
                    # straight wall gives a low, stable residual regardless of where along the
                    # wall the match lands. Re-examine what it actually matched: a quick,
                    # cheap one-shot re-query at the returned pose (not a second ICP run) finds
                    # the same correspondence set its last iteration converged on.
                    world_pts = transform_points(local_pts, corrected)
                    dists, idx = cKDTree(map_pts).query(world_pts)
                    valid = dists < args.icp_max_correspondence_dist
                    matched = map_pts[idx[valid]]
                    centered = matched - matched.mean(axis=0)
                    eigvals, eigvecs = np.linalg.eigh(np.cov(centered.T))
                    # ascending order: eigvals[0] <= eigvals[1]. A corner/varied-angle match
                    # (the default island world) spreads matched points roughly isotropically
                    # (ratio near 1, trust the correction fully). A straight wall spreads them
                    # almost entirely along its own length (ratio near 0): the correction
                    # component along THAT axis is genuinely unconstrained by this scan, not
                    # just noisy, so it's down-weighted toward the odometry prediction instead
                    # of trusted -- the perpendicular (cross-wall) component, which IS
                    # constrained, is kept at full confidence either way.
                    eccentricity = eigvals[0] / (eigvals[1] + 1e-9)
                    v_minor, v_major = eigvecs[:, 0], eigvecs[:, 1]
                    delta = np.array([corrected[0] - pred_x, corrected[1] - pred_y])
                    delta_filtered = ((delta @ v_minor) * v_minor
                                      + eccentricity * (delta @ v_major) * v_major)
                    if args.icp_confidence_margin > 0:
                        # The OTHER half of the confidence check: a sparse map can hand back
                        # a small, isotropic-LOOKING correspondence set (e.g. a short wall
                        # segment just reacquired after a long blind stretch) that isn't
                        # geometrically degenerate, just statistically unreliable from too few
                        # points -- confirmed directly, per-scan correction magnitude grows
                        # noticeably as n_valid shrinks toward --icp-min-correspondences's own
                        # floor. Scale the WHOLE correction (both axes) by how far n_valid is
                        # above that floor, ramping to full trust `icp_confidence_margin`
                        # correspondences above it.
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
                    status_text=f"goal ({goal[0]:.1f},{goal[1]:.1f})"))

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
        print(f"FELL/QP-FAILED before reaching the goal (final distance {final_dist:.2f} m).")
    elif reached:
        print(f"Reached goal ({goal[0]:.2f}, {goal[1]:.2f}) in {t_reached:.1f}s, "
              f"final distance {final_dist:.2f} m, {n_lidar_scans} lidar scans.")
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
