"""Stage 2 evaluation: does the learned-world-model CEM planner (navigate_wam.py) actually
reach goals as well as the sibling go1_rl_walk/mpc/navigate_mpc.py's hand-designed steering
laws (strafe and turn-to-face)? Judged across randomized multi-seed trials, not a single run
-- see memory "go1-eval-method": a single favorable/unfavorable run is not a verdict.

Runs each of {wam, mpc-strafe, mpc-turn} on the SAME set of randomly generated (seed, goal)
trials (same env seed per trial across all three methods, for an apples-to-apples
comparison -- same initial state, same random terrain-free flat ground), by invoking each
script as a subprocess and parsing its own printed summary line (simpler and less
error-prone than importing and refactoring three separate argparse `main()`s that weren't
written to be called as a library).

Usage (needs go1_rl_walk/mpc's own venv):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_navigate.py --trials 8
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WAM_DIR = Path(__file__).resolve().parent
PYTHON = WAM_DIR.parent / "go1_rl_walk" / "mpc" / "venv" / "bin" / "python3"
NAVIGATE_MPC = WAM_DIR.parent / "go1_rl_walk" / "mpc" / "navigate_mpc.py"

REACHED_RE = re.compile(r"Reached goal .* in ([\d.]+)s, final distance ([\d.]+) m")
NOT_REACHED_RE = re.compile(r"Did NOT reach the goal .* \(final distance ([\d.]+) m")
FELL_RE = re.compile(r"FELL/QP-FAILED .* \(final distance ([\d.]+) m\)")


def parse_result(stdout: str):
    """Returns (status, time_s_or_None, final_dist)."""
    m = REACHED_RE.search(stdout)
    if m:
        return "reached", float(m.group(1)), float(m.group(2))
    m = FELL_RE.search(stdout)
    if m:
        return "fell", None, float(m.group(1))
    m = NOT_REACHED_RE.search(stdout)
    if m:
        return "timeout", None, float(m.group(1))
    return "unknown", None, None


def run_trial(method: str, seed: int, goal, seconds: float, checkpoint: str, horizon_steps: int = None):
    gx, gy = goal
    if method == "wam":
        cmd = [str(PYTHON), str(WAM_DIR / "navigate_wam.py"),
               "--goal-x", str(gx), "--goal-y", str(gy), "--seed", str(seed),
               "--seconds", str(seconds), "--checkpoint", checkpoint]
        if horizon_steps is not None:
            cmd += ["--horizon-steps", str(horizon_steps)]
    elif method == "wam-chunked":
        cmd = [str(PYTHON), str(WAM_DIR / "navigate_wam_chunked.py"),
               "--goal-x", str(gx), "--goal-y", str(gy), "--seed", str(seed),
               "--seconds", str(seconds), "--checkpoint",
               "checkpoints/world_model_chunked_large.pt"]
    elif method in ("mpc-strafe", "mpc-turn"):
        steering = "strafe" if method == "mpc-strafe" else "turn"
        cmd = [str(PYTHON), str(NAVIGATE_MPC),
               "--goal-x", str(gx), "--goal-y", str(gy), "--seed", str(seed),
               "--seconds", str(seconds), "--steering", steering]
    else:
        raise ValueError(method)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(WAM_DIR if method in ("wam", "wam-chunked") else NAVIGATE_MPC.parent),
                           capture_output=True, text=True, timeout=seconds + 60)
    wall = time.time() - t0
    status, t_reach, final_dist = parse_result(proc.stdout)
    if status == "unknown":
        print(f"  [warn] could not parse output for {method} seed={seed} goal={goal}:\n"
              f"{proc.stdout[-500:]}\n{proc.stderr[-500:]}", file=sys.stderr)
    return status, t_reach, final_dist, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--base-seed", type=int, default=100)
    ap.add_argument("--goal-seed", type=int, default=42)
    ap.add_argument("--min-dist", type=float, default=1.5)
    ap.add_argument("--max-dist", type=float, default=3.0)
    ap.add_argument("--checkpoint", type=str, default="checkpoints/world_model.pt")
    ap.add_argument("--methods", nargs="+", default=["wam", "mpc-strafe", "mpc-turn"])
    ap.add_argument("--horizon-steps", type=int, default=None,
                     help="passthrough to navigate_wam.py's --horizon-steps (wam method only); "
                          "default None leaves navigate_wam.py's own default (10) unchanged")
    args = ap.parse_args()

    rng = np.random.default_rng(args.goal_seed)
    trials = []
    for i in range(args.trials):
        angle = rng.uniform(-np.pi, np.pi)
        dist = rng.uniform(args.min_dist, args.max_dist)
        goal = (dist * np.cos(angle), dist * np.sin(angle))
        seed = args.base_seed + i
        trials.append((seed, goal))

    results = {m: [] for m in args.methods}
    for method in args.methods:
        print(f"=== {method} ===")
        for seed, goal in trials:
            status, t_reach, final_dist, wall = run_trial(
                method, seed, goal, args.seconds, args.checkpoint, args.horizon_steps)
            results[method].append((status, t_reach, final_dist))
            print(f"  seed={seed:4d} goal=({goal[0]:+.2f},{goal[1]:+.2f}): "
                  f"{status:8s} t={t_reach if t_reach is not None else float('nan'):5.1f}s "
                  f"final_dist={final_dist:.3f}m  ({wall:.1f}s wall)")

    print("\n=== Summary (randomized multi-seed trials, same seeds/goals across methods) ===")
    print(f"{'method':12s} {'success':8s} {'fell':6s} {'timeout':8s} "
          f"{'mean_t_reach':13s} {'mean_final_dist':16s}")
    for method in args.methods:
        rows = results[method]
        n = len(rows)
        n_reached = sum(1 for s, _, _ in rows if s == "reached")
        n_fell = sum(1 for s, _, _ in rows if s == "fell")
        n_timeout = sum(1 for s, _, _ in rows if s == "timeout")
        t_reaches = [t for s, t, _ in rows if s == "reached"]
        final_dists = [d for _, _, d in rows if d is not None]
        mean_t = np.mean(t_reaches) if t_reaches else float("nan")
        mean_d = np.mean(final_dists) if final_dists else float("nan")
        print(f"{method:12s} {n_reached}/{n:<6d} {n_fell:<6d} {n_timeout:<8d} "
              f"{mean_t:13.2f} {mean_d:16.3f}")


if __name__ == "__main__":
    main()
