"""Stage 2: drive the real Go2 (classical MPC, zero RL residual, same convention as
collect_data.py/eval_rollout.py and the sibling go1_rl_walk/mpc/navigate_mpc.py) toward a
world-frame (x, y) goal -- but choose the velocity command with the Stage 1 learned world
model via CEM shooting (wam/planner.py), not a hand-written steering law. A direct
comparison point against navigate_mpc.py's `compute_nav_command`/`compute_nav_command_turn`:
same task, same real simulator, same zero-residual control lever (the 3-dim velocity
command) -- the only thing that changes is whether the command comes from an analytic
formula or from a learned model being asked "which command gets me closest to the goal".

Every tick still reads the REAL ground-truth pose (info["base_pos"]/info["yaw"]) for "where
am I now" and feeds it to the planner -- only the planner's internal look-ahead rollout
(wam/planner.py + wam/pose.py) uses the learned model and its own dead-reckoned pose
estimate to score candidate commands. The chosen command is rate-limited the same way
navigate_mpc.py's is before being sent to env.set_command() (same reason: this env's QP
goes infeasible under raw step changes in a continuously-recomputed command).

Usage (needs go1_rl_walk/mpc's own venv):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 navigate_wam.py \
        --goal-x 3.0 --goal-y 1.0 --checkpoint checkpoints/world_model.pt --record out.gif
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
from wam.model import WorldModel                            # noqa: E402
from wam.planner import CEMPlanner                           # noqa: E402
from wam.state import STATE_DIM, ACTION_DIM, extract_state   # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal-x", type=float, default=None)
    parser.add_argument("--goal-y", type=float, default=None)
    parser.add_argument("--waypoint", type=float, nargs=2, action="append", default=None,
                         metavar=("X", "Y"))
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tolerance", type=float, default=0.15)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/world_model.pt")
    parser.add_argument("--horizon-steps", type=int, default=10,
                         help="macro-steps (50ms each) the planner looks ahead per replan.")
    parser.add_argument("--replan-every", type=int, default=5,
                         help="macro-steps between replans -- holding a command this many "
                              "macro-steps at a time matches Stage 1's own data-collection "
                              "hold period scale, see wam/planner.py's module docstring.")
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
    args = parser.parse_args()

    if args.waypoint:
        route = [np.array(wp) for wp in args.waypoint]
    elif args.goal_x is not None and args.goal_y is not None:
        route = [np.array([args.goal_x, args.goal_y])]
    else:
        parser.error("give either --goal-x/--goal-y or at least one --waypoint")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WorldModel(STATE_DIM, ACTION_DIM, hidden=ckpt["hidden"], n_hidden=ckpt["n_hidden"])
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

    # The planner's search space is the env's REAL command ranges (what Stage 1's training
    # data was collected over), not the zeroed ranges passed to MPCResidualEnv above (those
    # only suppress the env's own random initial command at reset).
    planner = CEMPlanner(
        model, speed_range=(-0.4, 0.8), lateral_range=(-0.3, 0.3), yaw_rate_range=(-1.0, 1.0),
        horizon_steps=args.horizon_steps, macro_dt=10 / 200.0,
        n_samples=args.n_samples, n_elites=args.n_elites, n_iters=args.n_iters)

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

    renderer = None
    if args.record:
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.0, 0.0, 0.0]
        cam.distance = max(6.0, 1.3 * max(float(np.hypot(*wp)) for wp in route) + 2.0)
        cam.elevation = -70
        cam.azimuth = 90
        renderer = mj.Renderer(env.mujoco_go2.model, height=480, width=640)

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

        if reached:
            target_action[:] = 0.0
        elif (step - 1) % (args.replan_every * 10) == 0:
            info_for_state = {"base_height": float(env.mujoco_go2.data.qpos[2])}
            state = extract_state(obs, info_for_state)
            result = planner.plan(state, (x, y, yaw), tuple(goal), warm_start=target_action)
            target_action = result.action.numpy()
            n_replans += 1

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
