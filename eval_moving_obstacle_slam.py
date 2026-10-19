"""Verification for navigate_wam_moving_obstacle_slam.py: does Stage 8/9's moving-obstacle
tracker still correctly separate "the real obstacle" from "the static walls" once its live
occupancy grid is built from Stage 13's drifting/corrected SLAM belief pose instead of ground
truth? Same before/after discipline as eval_moving_obstacle.py's own 8-phase protocol (varies
both env seed and the obstacle's phase offset so a single lucky/unlucky phase can't hide the
real behavior -- see memory "go1-eval-method"), extended to 3 conditions the way
eval_localization.py/eval_localization_terrain.py already do: ground truth, noisy localization,
noisy + --icp-degeneracy-filter.

The direct diagnostic for the motivating hypothesis (belief-pose drift could make a genuinely
static wall cell fall outside the tracker's static-reference match radius and get misclassified
as a ghost dynamic obstacle) is `max_tracks_seen`: there is only ever ONE real moving obstacle
in this scenario, so any trial reporting more than 1 simultaneous track is a direct, unambiguous
positive for that hypothesis -- not something that has to be inferred from task outcome alone.

Usage:
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_moving_obstacle_slam.py \
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
TRACKS_RE = re.compile(r"Max simultaneous dynamic tracks seen: (\d+)")
DRIFT_RE = re.compile(
    r"Final localization drift: belief \(ICP-corrected\) = ([\d.]+)m, "
    r"raw odometry = ([\d.]+)m\. ICP converged on (\d+)/(\d+) scans")


def run_one(seed, phase, seconds, noisy_localization, degeneracy_filter=False):
    cmd = ["/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3",
           str(Path(__file__).resolve().parent / "navigate_wam_moving_obstacle_slam.py"),
           "--seed", str(seed), "--obstacle-phase", str(phase), "--seconds", str(seconds)]
    if noisy_localization:
        cmd.append("--noisy-localization")
    if degeneracy_filter:
        cmd.append("--icp-degeneracy-filter")
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent))
    text = out.stdout + out.stderr

    tracks = None
    tm = TRACKS_RE.search(text)
    if tm:
        tracks = int(tm.group(1))
    drift = None
    dm = DRIFT_RE.search(text)
    if dm:
        drift = (float(dm.group(1)), float(dm.group(2)), int(dm.group(3)), int(dm.group(4)))

    m = REACHED_RE.search(text)
    if m:
        return "reached", float(m.group(2)), float(m.group(3)), tracks, drift
    m = FELL_RE.search(text)
    if m:
        return "fell", float(m.group(1)), float(m.group(2)), tracks, drift
    m = TIMEOUT_RE.search(text)
    if m:
        return "timeout", float(m.group(1)), float(m.group(2)), tracks, drift
    print(f"  [warn] could not parse output for seed={seed} phase={phase:.2f}:\n{text[-500:]}")
    return "unknown", None, None, tracks, drift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--base-seed", type=int, default=300)
    args = ap.parse_args()

    phases = np.linspace(0.0, 2 * np.pi, args.trials, endpoint=False)

    conditions = [
        (False, False, "GROUND TRUTH POSE (Stage 8/9's own baseline)"),
        (True, False, "NOISY LOCALIZATION (lidar+ICP belief pose)"),
        (True, True, "NOISY LOCALIZATION + --icp-degeneracy-filter"),
    ]

    for noisy_localization, degeneracy_filter, label in conditions:
        print(f"=== {label} ===")
        results = []
        for i, phase in enumerate(phases):
            seed = args.base_seed + i
            status, final_dist, min_dist, tracks, drift = run_one(
                seed, phase, args.seconds, noisy_localization, degeneracy_filter)
            results.append((seed, phase, status, final_dist, min_dist, tracks, drift))
            ghost_note = ""
            if tracks is not None and tracks > 1:
                ghost_note = f" ** GHOST TRACK(S): max_tracks={tracks} **"
            drift_note = ""
            if drift is not None:
                drift_note = f" belief_drift={drift[0]:.3f}m odom_drift={drift[1]:.3f}m"
            print(f"  seed={seed:4d} phase={phase:.2f}: {status:8s} "
                  f"final_dist={final_dist:.2f}m min_dist_to_obstacle={min_dist:.2f}m"
                  f"{drift_note}{ghost_note}")

        n = len(results)
        n_reached = sum(1 for r in results if r[2] == "reached")
        n_fell = sum(1 for r in results if r[2] == "fell")
        n_timeout = sum(1 for r in results if r[2] == "timeout")
        min_dists = [r[4] for r in results if r[4] is not None]
        tracks_seen = [r[5] for r in results if r[5] is not None]
        n_ghost_trials = sum(1 for t in tracks_seen if t > 1)
        print(f"  Summary: {n_reached}/{n} reached, {n_fell} fell, {n_timeout} timed out, "
              f"mean min-distance-to-obstacle {np.mean(min_dists):.2f}m "
              f"(worst {np.min(min_dists):.2f}m), "
              f"ghost-track trials: {n_ghost_trials}/{len(tracks_seen)} "
              f"(max track count seen: {max(tracks_seen) if tracks_seen else 'n/a'})\n")


if __name__ == "__main__":
    main()
