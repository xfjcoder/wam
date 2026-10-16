"""Stage 6: drive the real Go2 to a goal using the chunked, history-conditioned world
model (wam/chunked_model.py) + ChunkedCEMPlanner (wam/chunked_planner.py), instead of
navigate_wam.py's single-step MLP + CEMPlanner. Same task/control lever/rate-limiting
conventions as navigate_wam.py -- the only thing that changes is which model/planner
chooses the velocity command.

Needs a real, maintained history of the last `history_len` (state, action) pairs (see
wam/chunked_planner.py's ChunkedHistoryBuffer) -- updated once per MACRO-STEP (every 10
physics ticks), using the TARGET command that was in effect during that macro-step (not
the rate-limited, actually-sent command), matching collect_data.py's own action-recording
convention exactly.

Usage:
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 navigate_wam_chunked.py \
        --goal-x 3.0 --goal-y 1.0 --checkpoint checkpoints/world_model_chunked_large.pt
"""
import argparse
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import time
from pathlib import Path

import mujoco as mj
import numpy as np
import torch

GO1_RL_WALK = Path(__file__).resolve().parent.parent / "go1_rl_walk"
sys.path.insert(0, str(GO1_RL_WALK / "mpc"))
sys.path.insert(0, str(GO1_RL_WALK))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mpc_residual_env import MPCResidualEnv                 # noqa: E402
from wam.chunked_model import ChunkedWorldModel              # noqa: E402
from wam.chunked_planner import ChunkedCEMPlanner, ChunkedHistoryBuffer  # noqa: E402
from wam.state import STATE_DIM, ACTION_DIM, extract_state   # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal-x", type=float, default=None)
    parser.add_argument("--goal-y", type=float, default=None)
    parser.add_argument("--waypoint", type=float, nargs=2, action="append", default=None,
                         metavar=("X", "Y"))
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tolerance", type=float, default=0.15)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/world_model_chunked_large.pt")
    parser.add_argument("--replan-every", type=int, default=5,
                         help="macro-steps between replans (same convention as navigate_wam.py)")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-elites", type=int, default=20)
    parser.add_argument("--n-iters", type=int, default=3)
    parser.add_argument("--max-accel", type=float, default=1.5)
    parser.add_argument("--max-yaw-accel", type=float, default=3.0)
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--terrain-amplitude", type=float, default=None)
    parser.add_argument("--terrain-footstep-planning", action="store_true")
    parser.add_argument("--record", type=str, default=None)
    parser.add_argument("--slowmo", type=float, default=1.0)
    parser.add_argument("--print-replan-time", action="store_true",
                         help="print mean wall-clock time per planner.plan() call at the end")
    args = parser.parse_args()

    if args.waypoint:
        route = [np.array(wp) for wp in args.waypoint]
    elif args.goal_x is not None and args.goal_y is not None:
        route = [np.array([args.goal_x, args.goal_y])]
    else:
        parser.error("give either --goal-x/--goal-y or at least one --waypoint")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ChunkedWorldModel(STATE_DIM, ACTION_DIM, ckpt["history_len"], ckpt["chunk_len"],
                              d_model=ckpt["d_model"], n_heads=ckpt["n_heads"],
                              n_layers=ckpt["n_layers"], d_ff=ckpt["d_ff"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    env = MPCResidualEnv(
        episode_seconds=args.seconds,
        speed_range=(0.0, 0.0), lateral_range=(0.0, 0.0), yaw_rate_range=(0.0, 0.0),
        terrain_amplitude_range=(args.terrain_amplitude, args.terrain_amplitude)
            if args.terrain_amplitude is not None else None,
        terrain_footstep_planning=args.terrain_footstep_planning,
        seed=args.seed,
    )
    obs, _ = env.reset(seed=args.seed)

    planner = ChunkedCEMPlanner(
        model, speed_range=(-0.4, 0.8), lateral_range=(-0.3, 0.3), yaw_rate_range=(-1.0, 1.0),
        macro_dt=10 / 200.0, n_samples=args.n_samples, n_elites=args.n_elites,
        n_iters=args.n_iters)

    max_cmd_delta = args.max_accel / 200.0
    max_yaw_delta = args.max_yaw_accel / 200.0
    max_steps = int(args.seconds * 200)
    n_wp = len(route)
    current_idx = 0
    goal = route[current_idx]
    x, y, yaw = 0.0, 0.0, 0.0
    cur_fwd, cur_lat, cur_yaw = 0.0, 0.0, 0.0
    target_action = np.zeros(3, dtype=np.float32)
    frames = []
    reached, fell = False, False
    t_reached = None
    visits = []
    just_arrived_idx = None
    n_replans = 0
    replan_times = []

    history = ChunkedHistoryBuffer(model.history_len, STATE_DIM, ACTION_DIM)
    info_for_state = {"base_height": float(env.mujoco_go2.data.qpos[2])}
    history.reset(extract_state(obs, info_for_state))
    pending_macro_state = history.states[-1].numpy()
    pending_macro_action = target_action.copy()

    def _add_marker(scene, pos_xy, rgba, size=0.08):
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(
            g, mj.mjtGeom.mjGEOM_SPHERE, np.array([size, 0, 0]),
            np.array([pos_xy[0], pos_xy[1], 0.05]), np.eye(3).flatten(),
            np.array(rgba, dtype=np.float32))
        scene.ngeom += 1

    renderer = None
    if args.record:
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.0, 0.0, 0.0]
        cam.distance = max(6.0, 1.3 * max(float(np.hypot(*wp)) for wp in route) + 2.0)
        cam.elevation = -70
        cam.azimuth = 90
        renderer = mj.Renderer(env.mujoco_go2.model, height=480, width=640)
        START_RGBA = (0.1, 0.9, 0.1, 1.0)
        GOAL_RGBA = (0.9, 0.1, 0.1, 1.0)
        WAYPOINT_RGBA = (0.9, 0.9, 0.1, 1.0)

    for step in range(1, max_steps + 1):
        dist = float(np.hypot(goal[0] - x, goal[1] - y))
        at_final_stop = (current_idx == n_wp - 1) and (not args.loop or n_wp == 1)
        if dist <= args.tolerance:
            if just_arrived_idx != current_idx:
                visits.append((current_idx, step / 200.0))
                just_arrived_idx = current_idx
            if at_final_stop:
                if not reached:
                    reached, t_reached = True, step / 200.0
            else:
                current_idx = (current_idx + 1) % n_wp
                goal = route[current_idx]
                just_arrived_idx = None

        # Macro-step boundary (every 10 physics ticks): the PREVIOUS macro-step's real
        # transition is now complete -- push it to history before planning with it.
        if (step - 1) % 10 == 0 and step > 1:
            history.push(pending_macro_state, pending_macro_action)
            info_for_state = {"base_height": float(env.mujoco_go2.data.qpos[2])}
            pending_macro_state = extract_state(obs, info_for_state)

        if reached:
            target_action[:] = 0.0
        elif (step - 1) % (args.replan_every * 10) == 0:
            t0 = time.perf_counter()
            result = planner.plan(history, (x, y, yaw), tuple(goal), warm_start=target_action)
            replan_times.append(time.perf_counter() - t0)
            target_action = result.action.numpy()
            n_replans += 1
        pending_macro_action = target_action.copy()

        cur_fwd += np.clip(target_action[0] - cur_fwd, -max_cmd_delta, max_cmd_delta)
        cur_lat += np.clip(target_action[1] - cur_lat, -max_cmd_delta, max_cmd_delta)
        cur_yaw += np.clip(target_action[2] - cur_yaw, -max_yaw_delta, max_yaw_delta)
        env.set_command(cur_fwd, cur_lat, cur_yaw)

        obs, _, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
        x, y = float(info["base_pos"][0]), float(info["base_pos"][1])
        yaw = float(info["yaw"])

        if args.record:
            if step % 8 == 0:
                renderer.update_scene(env.mujoco_go2.data, camera=cam)
                _add_marker(renderer.scene, (0.0, 0.0), START_RGBA)
                for i, wp in enumerate(route):
                    _add_marker(renderer.scene, wp,
                                GOAL_RGBA if i == n_wp - 1 else WAYPOINT_RGBA)
                frames.append(renderer.render().copy())
        else:
            time.sleep(0.005)

        if terminated or truncated:
            fell = terminated
            break
        if reached and step > (t_reached * 200 + 200):
            break

    final_dist = float(np.hypot(goal[0] - x, goal[1] - y))
    print(f"({n_replans} replans over {step} ticks)")
    if n_wp == 1:
        if fell:
            print(f"FELL/QP-FAILED before reaching the goal (final distance {final_dist:.2f} m).")
        elif reached:
            print(f"Reached goal ({goal[0]:.2f}, {goal[1]:.2f}) in {t_reached:.1f}s, "
                  f"final distance {final_dist:.2f} m.")
        else:
            print(f"Did NOT reach the goal within {args.seconds:.0f}s "
                  f"(final distance {final_dist:.2f} m, started {np.hypot(*route[0]):.2f} m away).")
    else:
        loop_s = " (looping)" if args.loop else ""
        print(f"Route{loop_s}: {len(visits)} waypoint arrival(s) across {n_wp}-waypoint route:")
        for idx, t in visits:
            print(f"  waypoint {idx} ({route[idx][0]:+.2f}, {route[idx][1]:+.2f}) reached at t={t:.1f}s")
        if fell:
            print(f"FELL/QP-FAILED (final distance {final_dist:.2f} m from waypoint {current_idx}).")
        elif reached:
            print(f"Completed the route, held the final waypoint for 1s after t={t_reached:.1f}s.")
        else:
            print(f"Did NOT complete the route within {args.seconds:.0f}s "
                  f"(currently targeting waypoint {current_idx}, {final_dist:.2f} m away).")

    if args.print_replan_time and replan_times:
        print(f"mean planner.plan() wall-clock time: {1000 * np.mean(replan_times):.2f} ms "
              f"({len(replan_times)} calls)")

    if renderer is not None:
        renderer.close()
    if args.record and frames:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        base_fps = 200.0 / 8
        frame_duration_ms = (1000.0 / base_fps) / args.slowmo
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=frame_duration_ms, loop=0)
        print(f"Saved {len(imgs)} frames to {args.record}")


if __name__ == "__main__":
    main()
