"""Diagnostic: how much of the CEM planner's rollout position error comes from the pose-
INTEGRATION scheme itself (wam/pose.py's integrate_pose -- a forward-Euler unicycle update
that freezes yaw at its START-of-step value for the whole 50ms macro-step) versus the learned
world model's own compounding autoregressive prediction error? Stage 3's corner-stall writeup
flagged "small pose-integration error" as a plausible contributor without ever isolating it
from model error -- this measures the two separately, across many held-out hold-blocks from
several seeds (never a single episode/block -- see memory "go1-eval-method").

Each held-out episode's command is resampled and held constant every `--hold-ticks` (default
100 ticks = 0.5s = 10 macro-steps) -- exactly planner.py's own horizon_steps/macro_dt (0.5s
CEM rollout under one constant candidate command). So every hold-block in a held-out episode
is a natural, already-labeled instance of "what the planner's rollout is trying to predict",
with real ground-truth start/end world pose available for free.

For each hold-block, four horizon-matched final-position estimates are compared against the
REAL ground-truth end pose:
  1. integration-only, Euler      -- real/ground-truth velocities each macro-step, integrate_pose
  2. integration-only, exact-arc  -- SAME real velocities, integrate_pose_exact
  3. full rollout, Euler          -- model-PREDICTED velocities (autoregressive, exactly what
                                      planner.plan() does today), integrate_pose
  4. full rollout, exact-arc      -- same model predictions, integrate_pose_exact

(1) vs (2) isolates integration-scheme error ALONE (zero model error -- both given the real
velocities). (1) vs (3) isolates how much the model's own autoregressive prediction error
adds on top of integration-scheme error, holding the integration scheme fixed. (3) vs (4)
is the one that actually matters for a fix: does swapping in exact-arc integration reduce the
TOTAL error the real planner would see, given its own (imperfect) model predictions?

Usage (needs go1_rl_walk/mpc's own venv):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_pose_integration.py \
        --checkpoint checkpoints/world_model.pt --seeds 6
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

GO1_RL_WALK = Path(__file__).resolve().parent.parent / "go1_rl_walk"
sys.path.insert(0, str(GO1_RL_WALK / "mpc"))
sys.path.insert(0, str(GO1_RL_WALK))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mpc_residual_env import MPCResidualEnv                                # noqa: E402
from wam.model import WorldModel                                           # noqa: E402
from wam.state import extract_state, extract_action, STATE_DIM, ACTION_DIM  # noqa: E402
from wam.pose import integrate_pose, integrate_pose_exact                  # noqa: E402
from wam.planner import _V_FWD, _V_LAT, _YAW_RATE                          # noqa: E402


def collect_episode_with_pose(env, rng, episode_seconds, hold_ticks, macro_k,
                               max_accel=1.5, max_yaw_accel=3.0):
    """Same procedure as collect_data.py's own collect_episode(), plus the real ground-truth
    world pose (x, y, yaw) recorded at every macro-step boundary -- needed here only, not for
    training data, so kept local to this diagnostic rather than changing collect_data.py's
    saved schema."""
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    max_ticks = int(episode_seconds * 200)
    dt = 1.0 / 200.0
    max_cmd_delta = max_accel * dt
    max_yaw_delta = max_yaw_accel * dt

    states, actions, next_states, poses = [], [], [], []
    state_t = None
    action_t = None
    pose_t = None
    # Treat the reset pose as the origin/zero-yaw reference frame, same convention
    # navigate_wam.py and friends already use ("x, y, yaw = 0.0, 0.0, 0.0" right after
    # reset()) -- the standing qpos's initial orientation is yaw=0 by construction.
    last_pose = (float(env.mujoco_go2.data.qpos[0]), float(env.mujoco_go2.data.qpos[1]), 0.0)
    ticks_in_macro = 0
    target_speed, target_lateral, target_yaw_rate = env.target_speed, env.target_lateral, env.target_yaw_rate
    cur_speed, cur_lateral, cur_yaw_rate = target_speed, target_lateral, target_yaw_rate

    for tick in range(max_ticks):
        if tick % hold_ticks == 0:
            target_speed = float(rng.uniform(*env.speed_range))
            target_lateral = float(rng.uniform(*env.lateral_range))
            target_yaw_rate = float(rng.uniform(*env.yaw_rate_range))

        cur_speed += np.clip(target_speed - cur_speed, -max_cmd_delta, max_cmd_delta)
        cur_lateral += np.clip(target_lateral - cur_lateral, -max_cmd_delta, max_cmd_delta)
        cur_yaw_rate += np.clip(target_yaw_rate - cur_yaw_rate, -max_yaw_delta, max_yaw_delta)
        env.set_command(cur_speed, cur_lateral, cur_yaw_rate)

        if ticks_in_macro == 0:
            info_for_state = {"base_height": float(env.mujoco_go2.data.qpos[2])}
            state_t = extract_state(obs, info_for_state)
            action_t = extract_action(env.target_speed, env.target_lateral, env.target_yaw_rate)
            pose_t = last_pose

        obs, _, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
        ticks_in_macro += 1
        last_pose = (float(info["base_pos"][0]), float(info["base_pos"][1]), float(info["yaw"]))

        if terminated:
            break

        if ticks_in_macro >= macro_k:
            state_next = extract_state(obs, info)
            states.append(state_t)
            actions.append(action_t)
            next_states.append(state_next)
            poses.append(pose_t)
            ticks_in_macro = 0

        if truncated:
            break

    # One extra trailing entry so poses[t] is always "start of macro-step t" for t in
    # [0, len(states)), AND poses[len(states)] is "end of the last recorded macro-step" --
    # avoids any off-by-one when a caller looks up a block's end pose as poses[hi].
    if states:
        poses.append(last_pose)

    return (np.array(states, dtype=np.float32) if states else np.zeros((0, STATE_DIM), dtype=np.float32),
            np.array(actions, dtype=np.float32) if actions else np.zeros((0, ACTION_DIM), dtype=np.float32),
            np.array(next_states, dtype=np.float32) if next_states else np.zeros((0, STATE_DIM), dtype=np.float32),
            np.array(poses, dtype=np.float32) if poses else np.zeros((0, 3), dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/world_model.pt")
    ap.add_argument("--episode-seconds", type=float, default=20.0)
    ap.add_argument("--hold-ticks", type=int, default=100)
    ap.add_argument("--macro-k", type=int, default=10)
    ap.add_argument("--horizon-steps", type=int, default=10,
                     help="macro-steps per block, matching planner.py's own horizon_steps")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--base-seed", type=int, default=9000,
                     help="distinct from training/validation/other-eval seed ranges")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WorldModel(STATE_DIM, ACTION_DIM, hidden=ckpt["hidden"], n_hidden=ckpt["n_hidden"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    macro_dt = args.macro_k / 200.0
    assert args.hold_ticks == args.horizon_steps * args.macro_k, (
        "hold-ticks must equal horizon_steps * macro_k so each hold-block is exactly one "
        "planner-horizon-equivalent interval under one constant command")

    errs = {"euler_real": [], "exact_real": [], "euler_model": [], "exact_model": []}
    n_blocks = 0

    for i in range(args.seeds):
        seed = args.base_seed + i
        rng = np.random.default_rng(seed)
        env = MPCResidualEnv(episode_seconds=args.episode_seconds, seed=seed)
        states, actions, next_states, poses = collect_episode_with_pose(
            env, rng, args.episode_seconds, args.hold_ticks, args.macro_k)
        n_steps = len(actions)
        n_full_blocks = n_steps // args.horizon_steps
        print(f"seed={seed}: {n_steps} macro-steps -> {n_full_blocks} full "
              f"{args.horizon_steps}-step blocks")

        for b in range(n_full_blocks):
            lo = b * args.horizon_steps
            hi = lo + args.horizon_steps
            x0, y0, yaw0 = (torch.tensor(float(v)) for v in poses[lo])
            x1_true, y1_true = float(poses[hi][0]), float(poses[hi][1])
            action_const = torch.tensor(actions[lo], dtype=torch.float32)

            # (1)/(2): integration-only, real ground-truth velocities each macro-step.
            xe, ye, yawe = x0.clone(), y0.clone(), yaw0.clone()
            xa, ya, yawa = x0.clone(), y0.clone(), yaw0.clone()
            for t in range(lo, hi):
                v_fwd = torch.tensor(float(next_states[t][_V_FWD]))
                v_lat = torch.tensor(float(next_states[t][_V_LAT]))
                yaw_rate = torch.tensor(float(next_states[t][_YAW_RATE]))
                xe, ye, yawe = integrate_pose(xe, ye, yawe, v_fwd, v_lat, yaw_rate, macro_dt)
                xa, ya, yawa = integrate_pose_exact(xa, ya, yawa, v_fwd, v_lat, yaw_rate, macro_dt)
            errs["euler_real"].append(float(np.hypot(float(xe) - x1_true, float(ye) - y1_true)))
            errs["exact_real"].append(float(np.hypot(float(xa) - x1_true, float(ya) - y1_true)))

            # (3)/(4): full rollout, model-predicted velocities (autoregressive), same as
            # planner.plan()'s actual CEM rollout but with a single fixed (real) action.
            state_t = torch.from_numpy(states[lo:lo + 1]).float()
            xm_e, ym_e, yawm_e = x0.clone(), y0.clone(), yaw0.clone()
            xm_a, ym_a, yawm_a = x0.clone(), y0.clone(), yaw0.clone()
            with torch.no_grad():
                s = state_t
                for t in range(args.horizon_steps):
                    s = model(s, action_const.unsqueeze(0))
                    v_fwd = s[0, _V_FWD]
                    v_lat = s[0, _V_LAT]
                    yaw_rate = s[0, _YAW_RATE]
                    xm_e, ym_e, yawm_e = integrate_pose(xm_e, ym_e, yawm_e, v_fwd, v_lat, yaw_rate, macro_dt)
                    xm_a, ym_a, yawm_a = integrate_pose_exact(xm_a, ym_a, yawm_a, v_fwd, v_lat, yaw_rate, macro_dt)
            errs["euler_model"].append(float(np.hypot(float(xm_e) - x1_true, float(ym_e) - y1_true)))
            errs["exact_model"].append(float(np.hypot(float(xm_a) - x1_true, float(ym_a) - y1_true)))
            n_blocks += 1

    print(f"\n{n_blocks} total {args.horizon_steps}-macro-step ({args.horizon_steps * macro_dt:.1f}s) "
          f"blocks across {args.seeds} held-out seeds.")
    print(f"\nFinal-position error vs. real ground truth, after one planner-horizon-equivalent block:")
    print(f"{'estimate':30s} {'mean (m)':10s} {'median (m)':12s} {'max (m)':10s}")
    labels = {
        "euler_real": "(1) integration-only, Euler",
        "exact_real": "(2) integration-only, exact-arc",
        "euler_model": "(3) full rollout, Euler [current]",
        "exact_model": "(4) full rollout, exact-arc",
    }
    for key, label in labels.items():
        vals = np.array(errs[key])
        print(f"{label:30s} {vals.mean():10.4f} {np.median(vals):12.4f} {vals.max():10.4f}")

    improve = (np.array(errs["euler_model"]) - np.array(errs["exact_model"]))
    n_improved = int((improve > 0).sum())
    n_worsened = int((improve < 0).sum())
    print(f"\n(3) vs (4), per-block: exact-arc better in {n_improved}/{n_blocks} blocks, "
          f"worse in {n_worsened}/{n_blocks} (mean delta {improve.mean():+.4f}m).")


if __name__ == "__main__":
    main()
