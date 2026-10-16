"""Stage 8 verification: multi-seed/phase comparison of navigate_wam_moving_obstacle.py with
the obstacle cost enabled vs. disabled -- same before/after discipline Stage 3 used to confirm
its (static) obstacle cost was actually load-bearing, not just harmless, now repeated for a
MOVING obstacle. Each trial varies both the env seed (friction/mass domain randomization) and
the obstacle's phase offset (WHEN in its sweep it crosses the robot's straight-line path
relative to episode start) -- a single phase would either always or never produce a
close encounter, hiding the real behavior (see memory "go1-eval-method").

Usage:
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_moving_obstacle.py \
        --trials 8 --seconds 25 --base-seed 300
"""
import argparse
import re
import subprocess
from pathlib import Path

import numpy as np

REACHED_RE = re.compile(r"Reached goal .* in ([\d.]+)s, final distance ([\d.]+) m, "
                         r"\d+ lidar scans, min distance to moving obstacle ([\d.]+) m")
FELL_RE = re.compile(r"FELL/QP-FAILED .* \(final distance ([\d.]+) m, "
                      r"min distance to moving obstacle ([\d.]+) m\)")
TIMEOUT_RE = re.compile(r"Did NOT reach the goal .* \(final distance ([\d.]+) m, "
                         r"min distance to moving obstacle ([\d.]+) m\)")


def run_one(seed, phase, seconds, disable_obstacle_cost):
    cmd = ["/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3",
           str(Path(__file__).resolve().parent / "navigate_wam_moving_obstacle.py"),
           "--seed", str(seed), "--obstacle-phase", str(phase), "--seconds", str(seconds)]
    if disable_obstacle_cost:
        cmd.append("--disable-obstacle-cost")
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent))
    text = out.stdout + out.stderr
    m = REACHED_RE.search(text)
    if m:
        return "reached", float(m.group(2)), float(m.group(3))
    m = FELL_RE.search(text)
    if m:
        return "fell", float(m.group(1)), float(m.group(2))
    m = TIMEOUT_RE.search(text)
    if m:
        return "timeout", float(m.group(1)), float(m.group(2))
    print(f"  [warn] could not parse output for seed={seed} phase={phase:.2f}:\n{text[-500:]}")
    return "unknown", None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--base-seed", type=int, default=300)
    args = ap.parse_args()

    phases = np.linspace(0.0, 2 * np.pi, args.trials, endpoint=False)

    for disable_obstacle_cost, label in [(True, "OBSTACLE COST DISABLED (Stage 2's goal-only)"),
                                          (False, "OBSTACLE COST ENABLED (Stage 8)")]:
        print(f"=== {label} ===")
        results = []
        for i, phase in enumerate(phases):
            seed = args.base_seed + i
            status, final_dist, min_dist = run_one(seed, phase, args.seconds, disable_obstacle_cost)
            results.append((seed, phase, status, final_dist, min_dist))
            print(f"  seed={seed:4d} phase={phase:.2f}: {status:8s} "
                  f"final_dist={final_dist:.2f}m min_dist_to_obstacle={min_dist:.2f}m")

        n = len(results)
        n_reached = sum(1 for r in results if r[2] == "reached")
        n_fell = sum(1 for r in results if r[2] == "fell")
        n_timeout = sum(1 for r in results if r[2] == "timeout")
        min_dists = [r[4] for r in results if r[4] is not None]
        print(f"  Summary: {n_reached}/{n} reached, {n_fell} fell, {n_timeout} timed out, "
              f"mean min-distance-to-obstacle {np.mean(min_dists):.2f}m "
              f"(worst {np.min(min_dists):.2f}m)\n")


if __name__ == "__main__":
    main()
