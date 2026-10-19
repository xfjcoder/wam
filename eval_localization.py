"""Stage 13 verification: multi-seed before/after comparison of ground-truth pose vs.
`--noisy-localization`'s belief pose (noisy odometry + ICP scan-matching, slam_sim's own
Stage 4 approach ported onto the real Go2 in navigate_wam_obstacles.py), on the SAME 4
start/goal pairs eval_obstacles.py already uses for its A*-on/off comparison -- same
reasoning: single-seed numbers have repeatedly been shown (this project's own history) to be
misleadingly inconsistent on their own.

Usage:
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_localization.py \
        --seeds 3 --base-seed 100 --seconds 45
"""
import argparse
import re
import subprocess
from pathlib import Path

REACHED_RE = re.compile(r"Reached goal .* in ([\d.]+)s, final distance ([\d.]+) m")
FELL_RE = re.compile(r"FELL/QP-FAILED .* \(final distance ([\d.]+) m\)")
TIMEOUT_RE = re.compile(r"Did NOT reach the goal .* \(final distance ([\d.]+) m\)")
DRIFT_RE = re.compile(
    r"Final localization drift: belief \(ICP-corrected\) = ([\d.]+)m, "
    r"raw odometry = ([\d.]+)m\. ICP converged on (\d+)/(\d+) scans")

PAIRS = [
    ("no obstacle", 1.5, 1.5, 1.5, 6.5),
    ("detour A", 1.5, 1.5, 10.5, 6.5),
    ("detour B (corner-stall case)", 10.5, 1.5, 1.5, 6.5),
    ("detour C (corner-stall case)", 1.5, 6.5, 10.5, 1.5),
]


def run_one(start_x, start_y, goal_x, goal_y, seed, seconds, noisy_localization):
    cmd = ["/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3",
           str(Path(__file__).resolve().parent / "navigate_wam_obstacles.py"),
           "--start-x", str(start_x), "--start-y", str(start_y),
           "--goal-x", str(goal_x), "--goal-y", str(goal_y),
           "--seed", str(seed), "--seconds", str(seconds)]
    if noisy_localization:
        cmd.append("--noisy-localization")
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent))
    text = out.stdout + out.stderr

    drift = None
    dm = DRIFT_RE.search(text)
    if dm:
        drift = (float(dm.group(1)), float(dm.group(2)), int(dm.group(3)), int(dm.group(4)))

    m = REACHED_RE.search(text)
    if m:
        return "reached", float(m.group(1)), float(m.group(2)), drift
    m = FELL_RE.search(text)
    if m:
        return "fell", None, float(m.group(1)), drift
    m = TIMEOUT_RE.search(text)
    if m:
        return "timeout", None, float(m.group(1)), drift
    return "unknown", None, None, drift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=45.0)
    args = ap.parse_args()

    for noisy_localization, label in [(False, "GROUND-TRUTH POSE (original)"),
                                       (True, "NOISY LOCALIZATION (Stage 13 belief pose)")]:
        print(f"=== {label} ===")
        for name, sx, sy, gx, gy in PAIRS:
            results = []
            for i in range(args.seeds):
                seed = args.base_seed + i
                status, t, final_dist, drift = run_one(
                    sx, sy, gx, gy, seed, args.seconds, noisy_localization)
                results.append((seed, status, t, final_dist, drift))
            summary = ", ".join(
                f"seed={s}:{st}" + (f"@{t:.1f}s" if t is not None else f"(d={d:.2f}m)")
                for s, st, t, d, drift in results)
            print(f"  {name:32s} ({sx},{sy})->({gx},{gy}): {summary}")
            if noisy_localization:
                for s, st, t, d, drift in results:
                    if drift is not None:
                        belief_d, odom_d, n_conv, n_att = drift
                        print(f"      seed={s}: belief drift={belief_d:.3f}m, "
                              f"odom drift={odom_d:.3f}m, ICP converged {n_conv}/{n_att}")
        print()


if __name__ == "__main__":
    main()
