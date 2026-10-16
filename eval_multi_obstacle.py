"""Stage 10 verification: multi-seed/phase comparison of navigate_wam_multi_obstacle.py (TWO
independently-moving obstacles) across three conditions -- obstacle cost disabled (is it
still needed at all with two obstacles?), frozen at last-sensed position (Stage 8's
assumption), and velocity-extrapolated per track (Stage 9/10). Varies both the env seed and
`--obstacle1-phase` (with `--obstacle2-phase` kept pi radians offset from it, the script's
own default relationship) across trials, same multi-seed/phase discipline every prior
stage's eval script uses.

Usage:
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_multi_obstacle.py \
        --trials 8 --seconds 30 --base-seed 300
"""
import argparse
import re
import subprocess
from pathlib import Path

import numpy as np

REACHED_RE = re.compile(r"Reached goal .* in ([\d.]+)s, final distance ([\d.]+) m, "
                         r"\d+ lidar scans, min distance to moving obstacles ([\d.]+) m")
FELL_RE = re.compile(r"FELL/QP-FAILED .* \(final distance ([\d.]+) m, "
                      r"min distance to moving obstacles ([\d.]+) m\)")
TIMEOUT_RE = re.compile(r"Did NOT reach the goal .* \(final distance ([\d.]+) m, "
                         r"min distance to moving obstacles ([\d.]+) m\)")


def run_one(seed, phase, seconds, mode):
    cmd = ["/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3",
           str(Path(__file__).resolve().parent / "navigate_wam_multi_obstacle.py"),
           "--seed", str(seed), "--obstacle1-phase", str(phase),
           "--obstacle2-phase", str(phase + np.pi), "--seconds", str(seconds)]
    if mode == "disabled":
        cmd.append("--disable-obstacle-cost")
    elif mode == "frozen":
        cmd.append("--disable-extrapolation")
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent))
    text = out.stdout + out.stderr
    m = REACHED_RE.search(text)
    if m:
        return "reached", float(m.group(1)), float(m.group(2)), float(m.group(3))
    m = FELL_RE.search(text)
    if m:
        return "fell", None, float(m.group(1)), float(m.group(2))
    m = TIMEOUT_RE.search(text)
    if m:
        return "timeout", None, float(m.group(1)), float(m.group(2))
    print(f"  [warn] could not parse output for seed={seed} phase={phase:.2f} mode={mode}:\n{text[-500:]}")
    return "unknown", None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--base-seed", type=int, default=300)
    args = ap.parse_args()

    phases = np.linspace(0.0, 2 * np.pi, args.trials, endpoint=False)

    for mode, label in [("disabled", "OBSTACLE COST DISABLED (goal-only)"),
                        ("frozen", "FROZEN at last-sensed position (Stage 8)"),
                        ("extrapolated", "VELOCITY-EXTRAPOLATED, per-track (Stage 9/10)")]:
        print(f"=== {label} ===")
        results = []
        for i, phase in enumerate(phases):
            seed = args.base_seed + i
            status, t_reach, final_dist, min_dist = run_one(seed, phase, args.seconds, mode)
            results.append((seed, phase, status, t_reach, final_dist, min_dist))
            t_str = "  --  " if t_reach is None else f"{t_reach:5.1f}s"
            print(f"  seed={seed:4d} phase={phase:.2f}: {status:8s} t={t_str} "
                  f"final_dist={final_dist:.2f}m min_dist_to_obstacles={min_dist:.2f}m")

        n = len(results)
        n_reached = sum(1 for r in results if r[2] == "reached")
        n_fell = sum(1 for r in results if r[2] == "fell")
        n_timeout = sum(1 for r in results if r[2] == "timeout")
        min_dists = [r[5] for r in results if r[5] is not None]
        reach_times = [r[3] for r in results if r[3] is not None]
        print(f"  Summary: {n_reached}/{n} reached, {n_fell} fell, {n_timeout} timed out, "
              f"mean min-distance-to-obstacles {np.mean(min_dists):.2f}m (worst {np.min(min_dists):.2f}m), "
              f"mean time-to-reach {np.mean(reach_times) if reach_times else float('nan'):.2f}s\n")


if __name__ == "__main__":
    main()
