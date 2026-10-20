"""Stage 16 escalation: does the SLAM-belief-drift-induced-ghosting hypothesis (refuted on the
default 12x8m island world by eval_moving_obstacle_slam.py -- ghost-track rate was the same
under ground truth as under noisy localization) still hold once the moving-obstacle tracker is
combined with Stage 13's own two SLAM-degenerate geometries instead? Same "sparse"/"corridor"
worlds `eval_localization_sparse.py` built (both via `World.loop_corridor()`, just different
`--world-outer`/`--world-island` sizes -- see that script's docstring for the geometric
rationale), driven through `navigate_wam_moving_obstacle_slam.py` instead of
`navigate_wam_obstacles.py` so the moving-obstacle tracker and `max_tracks_seen` ghost-track
diagnostic are active too.

This stresses both systems TOGETHER on worlds where Stage 13 already found real, documented
localization effects (sparse: long blind stretches with zero ICP correspondences; corridor:
along-corridor-axis underconstrained ICP) -- if belief-drift-induced ghosting is a real effect
anywhere, degraded localization on these worlds is the best place to find it, since any new
localization-ghosting interaction will stand out against already-known confounds rather than
being confused with them.

Same reporting convention as both eval_moving_obstacle_slam.py (ghost-track/ max_tracks_seen)
and eval_localization_sparse.py (per-scenario, per-pair, multi-condition, with belief/odometry
drift and ICP convergence rate), varying the obstacle's phase per trial the way
eval_moving_obstacle_slam.py does (not just the seed) so a single lucky/unlucky obstacle timing
can't hide the real behavior.

Each pair sets `--obstacle-axis` to whichever axis is PERPENDICULAR to that pair's direction of
travel, matching Stage 8/9's own default-world scenario (travel along y, obstacle oscillates
along x -- a crossing obstacle, not a blocking one). Discovered this mattered via a timing probe:
the naive default axis="x" on an E-W-travel corridor pair put the obstacle oscillating LENGTHWISE
along the robot's own path (blocking, not crossing) -- a harder, non-representative scenario that
alone explained a 60s non-reach, not a real localization/tracking effect.

Usage:
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_moving_obstacle_slam_sparse.py \
        --phases 4 --base-seed 100 --seconds 50
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

SCENARIOS = [
    ("sparse", (30.0, 30.0), (1.0, 1.0, 2.0, 2.0), [
        ("blind crossing E-W", 5.0, 15.0, 25.0, 15.0, "y"),
        ("blind crossing N-S", 15.0, 5.0, 15.0, 25.0, "x"),
    ]),
    ("corridor", (24.0, 8.0), (4.0, 2.0, 20.0, 6.0), [
        ("long bottom corridor", 2.0, 1.0, 22.0, 1.0, "y"),
        ("long top corridor", 2.0, 7.0, 22.0, 7.0, "y"),
    ]),
]


def run_one(world_outer, world_island, start_x, start_y, goal_x, goal_y, obstacle_axis, seed,
            phase, seconds, noisy_localization, degeneracy_filter=False):
    cmd = ["/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3",
           str(Path(__file__).resolve().parent / "navigate_wam_moving_obstacle_slam.py"),
           "--world-outer", str(world_outer[0]), str(world_outer[1]),
           "--world-island", str(world_island[0]), str(world_island[1]),
           str(world_island[2]), str(world_island[3]),
           "--start-x", str(start_x), "--start-y", str(start_y),
           "--goal-x", str(goal_x), "--goal-y", str(goal_y),
           "--obstacle-axis", obstacle_axis,
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
    ap.add_argument("--phases", type=int, default=4)
    ap.add_argument("--base-seed", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=50.0)
    ap.add_argument("--with-degeneracy-filter", action="store_true",
                     help="add a third condition -- noisy localization WITH "
                          "--icp-degeneracy-filter -- same as eval_localization_sparse.py.")
    ap.add_argument("--filter-only", action="store_true",
                     help="skip the GROUND TRUTH and plain NOISY LOCALIZATION conditions and "
                          "run ONLY the --icp-degeneracy-filter condition -- for following up "
                          "on an already-collected baseline without re-running it. Implies "
                          "--with-degeneracy-filter.")
    args = ap.parse_args()

    phases = np.linspace(0.0, 2 * np.pi, args.phases, endpoint=False)

    if args.filter_only:
        conditions = [(True, True, "NOISY LOCALIZATION + --icp-degeneracy-filter")]
    else:
        conditions = [(False, False, "GROUND TRUTH POSE"),
                      (True, False, "NOISY LOCALIZATION (lidar+ICP belief pose)")]
        if args.with_degeneracy_filter:
            conditions.append((True, True, "NOISY LOCALIZATION + --icp-degeneracy-filter"))

    for scenario_name, world_outer, world_island, pairs in SCENARIOS:
        print(f"##### scenario: {scenario_name}  outer={world_outer} island={world_island} #####")
        for name, sx, sy, gx, gy, obstacle_axis in pairs:
            print(f"--- pair: {name} ({sx},{sy})->({gx},{gy}) obstacle-axis={obstacle_axis} ---")
            for noisy_localization, degeneracy_filter, label in conditions:
                print(f"=== {label} ===")
                results = []
                for i, phase in enumerate(phases):
                    seed = args.base_seed + i
                    status, final_dist, min_dist, tracks, drift = run_one(
                        world_outer, world_island, sx, sy, gx, gy, obstacle_axis, seed, phase,
                        args.seconds, noisy_localization, degeneracy_filter)
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
                mean_min = f"{np.mean(min_dists):.2f}m" if min_dists else "n/a"
                worst_min = f"{np.min(min_dists):.2f}m" if min_dists else "n/a"
                print(f"  Summary: {n_reached}/{n} reached, {n_fell} fell, {n_timeout} timed out, "
                      f"mean min-distance-to-obstacle {mean_min} (worst {worst_min}), "
                      f"ghost-track trials: {n_ghost_trials}/{len(tracks_seen)} "
                      f"(max track count seen: {max(tracks_seen) if tracks_seen else 'n/a'})\n")


if __name__ == "__main__":
    main()
