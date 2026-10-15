"""Stage 3b verification: multi-seed before/after comparison of the A* global planner
(wam/global_planner.py) against Stage 3's original CEM-only obstacle avoidance, on the same
4 start/goal pairs used for Stage 3's one-off qualitative table -- this time across several
seeds per pair, since single-seed numbers were repeatedly shown (Stage 2/4's own eval
scripts) to be misleadingly inconsistent on their own.

Usage:
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_obstacles.py \
        --seeds 3 --base-seed 100 --seconds 45
"""
import argparse
import re
import subprocess
from pathlib import Path

REACHED_RE = re.compile(r"Reached goal .* in ([\d.]+)s, final distance ([\d.]+) m")
FELL_RE = re.compile(r"FELL/QP-FAILED .* \(final distance ([\d.]+) m\)")
TIMEOUT_RE = re.compile(r"Did NOT reach the goal .* \(final distance ([\d.]+) m\)")

PAIRS = [
    ("no obstacle", 1.5, 1.5, 1.5, 6.5),
    ("detour A", 1.5, 1.5, 10.5, 6.5),
    ("detour B (corner-stall case)", 10.5, 1.5, 1.5, 6.5),
    ("detour C (corner-stall case)", 1.5, 6.5, 10.5, 1.5),
]


def run_one(start_x, start_y, goal_x, goal_y, seed, seconds, disable_global_planner):
    cmd = ["/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3",
           str(Path(__file__).resolve().parent / "navigate_wam_obstacles.py"),
           "--start-x", str(start_x), "--start-y", str(start_y),
           "--goal-x", str(goal_x), "--goal-y", str(goal_y),
           "--seed", str(seed), "--seconds", str(seconds)]
    if disable_global_planner:
        cmd.append("--disable-global-planner")
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent))
    text = out.stdout + out.stderr
    m = REACHED_RE.search(text)
    if m:
        return "reached", float(m.group(1)), float(m.group(2))
    m = FELL_RE.search(text)
    if m:
        return "fell", None, float(m.group(1))
    m = TIMEOUT_RE.search(text)
    if m:
        return "timeout", None, float(m.group(1))
    return "unknown", None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=45.0)
    args = ap.parse_args()

    for disable_global_planner, label in [(True, "A* DISABLED (Stage 3 original)"),
                                           (False, "A* ENABLED (Stage 3b)")]:
        print(f"=== {label} ===")
        for name, sx, sy, gx, gy in PAIRS:
            results = []
            for i in range(args.seeds):
                seed = args.base_seed + i
                status, t, final_dist = run_one(sx, sy, gx, gy, seed, args.seconds,
                                                 disable_global_planner)
                results.append((seed, status, t, final_dist))
            summary = ", ".join(
                f"seed={s}:{st}" + (f"@{t:.1f}s" if t is not None else f"(d={d:.2f}m)")
                for s, st, t, d in results)
            print(f"  {name:32s} ({sx},{sy})->({gx},{gy}): {summary}")
        print()


if __name__ == "__main__":
    main()
