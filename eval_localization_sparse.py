"""Stage 13 stress test: does `--noisy-localization` (navigate_wam_obstacles.py) hold up on
worlds that are deliberately WORSE for ICP scan-matching than the default 12x8m island world
every other eval in this repo uses? The Stage 13 README section flagged this as untested
scope, not a known failure -- this answers it.

Two distinct SLAM-degenerate geometries, both built via the SAME `World.loop_corridor()`
slam_sim already provides (just different `--world-outer`/`--world-island` sizes, a Stage 13.1
addition to navigate_wam_obstacles.py -- no slam_sim code touched):

  "sparse": outer=(30,30), a token 1x1 island tucked in a corner (geometrically irrelevant --
  present only because loop_corridor requires an island rectangle). The two start/goal pairs
  cross the ~20m-wide empty middle, where EVERY wall (island included) is well beyond the
  lidar's 8m max range -- for a long stretch, every beam misses, scan_to_local_points returns
  zero hit points, and ICP has literally nothing to correct against (falls back to raw,
  uncorrected odometry prediction for that scan, same as it already does near-field when the
  map is too sparse to match -- this just makes the sparse stretch much longer).

  "corridor": outer=(24,8), island=(4,2,20,6) -- the SAME 12x8-style loop_corridor shape
  stretched lengthwise so the top and bottom corridors are each a 20m-long, 2m-wide straight
  hallway (vs. the default world's ~4m straight stretches). The two pairs drive straight down
  one of these corridors. This is the textbook ICP-degenerate case flagged in slam_sim's own
  world.py docstring (the reason a LOOP was built in the first place): translating the robot
  along a straight corridor's own axis leaves the local wall scan looking identical, so the
  component of pose correction along the corridor is underconstrained -- ICP can converge
  with near-zero residual while still being wrong about exactly how far along the corridor it
  is.

Same reporting convention as eval_localization.py: ground-truth-pose vs. noisy-localization,
multi-seed, with belief/odometry drift and ICP convergence rate per trial.

Usage:
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_localization_sparse.py \
        --seeds 3 --base-seed 100 --seconds 50
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

SCENARIOS = [
    ("sparse", (30.0, 30.0), (1.0, 1.0, 2.0, 2.0), [
        ("blind crossing E-W", 5.0, 15.0, 25.0, 15.0),
        ("blind crossing N-S", 15.0, 5.0, 15.0, 25.0),
    ]),
    ("corridor", (24.0, 8.0), (4.0, 2.0, 20.0, 6.0), [
        ("long bottom corridor", 2.0, 1.0, 22.0, 1.0),
        ("long top corridor", 2.0, 7.0, 22.0, 7.0),
    ]),
]


def run_one(world_outer, world_island, start_x, start_y, goal_x, goal_y, seed, seconds,
            noisy_localization, degeneracy_filter=False, confidence_margin=0.0,
            reject_below=0):
    cmd = ["/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3",
           str(Path(__file__).resolve().parent / "navigate_wam_obstacles.py"),
           "--world-outer", str(world_outer[0]), str(world_outer[1]),
           "--world-island", str(world_island[0]), str(world_island[1]),
           str(world_island[2]), str(world_island[3]),
           "--start-x", str(start_x), "--start-y", str(start_y),
           "--goal-x", str(goal_x), "--goal-y", str(goal_y),
           "--seed", str(seed), "--seconds", str(seconds)]
    if noisy_localization:
        cmd.append("--noisy-localization")
    if degeneracy_filter:
        cmd.append("--icp-degeneracy-filter")
    if confidence_margin > 0:
        cmd += ["--icp-confidence-margin", str(confidence_margin)]
    if reject_below > 0:
        cmd += ["--icp-reject-below-correspondences", str(reject_below)]
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
    ap.add_argument("--seconds", type=float, default=50.0)
    ap.add_argument("--with-degeneracy-filter", action="store_true",
                     help="add a third condition -- noisy localization WITH "
                          "--icp-degeneracy-filter -- to test the fix against the exact "
                          "degenerate geometries that motivated it.")
    ap.add_argument("--with-confidence-margin", type=float, default=0.0,
                     help="add a fourth condition -- noisy localization WITH "
                          "--icp-degeneracy-filter AND --icp-confidence-margin set to this "
                          "value -- to test the count-based confidence check against the "
                          "sparse-world low-correspondence-count case it specifically targets.")
    ap.add_argument("--with-reject-below", type=int, default=0,
                     help="add a condition -- noisy localization WITH --icp-degeneracy-filter "
                          "AND --icp-reject-below-correspondences set to this value -- the "
                          "hard-cutoff alternative tried after --with-confidence-margin's "
                          "scaling ramp came back a wash on the sparse blind-crossing case.")
    args = ap.parse_args()

    conditions = [(False, False, 0.0, 0, "GROUND-TRUTH POSE"),
                  (True, False, 0.0, 0, "NOISY LOCALIZATION (Stage 13 belief pose)")]
    if args.with_degeneracy_filter:
        conditions.append((True, True, 0.0, 0, "NOISY LOCALIZATION + --icp-degeneracy-filter"))
    if args.with_confidence_margin > 0:
        conditions.append((True, True, args.with_confidence_margin, 0,
                            f"NOISY LOCALIZATION + filter + confidence-margin={args.with_confidence_margin:.0f}"))
    if args.with_reject_below > 0:
        conditions.append((True, True, 0.0, args.with_reject_below,
                            f"NOISY LOCALIZATION + filter + reject-below={args.with_reject_below}"))

    for scenario_name, world_outer, world_island, pairs in SCENARIOS:
        print(f"##### scenario: {scenario_name}  outer={world_outer} island={world_island} #####")
        for noisy_localization, degeneracy_filter, confidence_margin, reject_below, label in conditions:
            print(f"=== {label} ===")
            for name, sx, sy, gx, gy in pairs:
                results = []
                for i in range(args.seeds):
                    seed = args.base_seed + i
                    status, t, final_dist, drift = run_one(
                        world_outer, world_island, sx, sy, gx, gy, seed, args.seconds,
                        noisy_localization, degeneracy_filter, confidence_margin, reject_below)
                    results.append((seed, status, t, final_dist, drift))
                summary = ", ".join(
                    f"seed={s}:{st}" + (f"@{t:.1f}s" if t is not None else f"(d={d:.2f}m)")
                    for s, st, t, d, drift in results)
                print(f"  {name:24s} ({sx},{sy})->({gx},{gy}): {summary}")
                if noisy_localization:
                    for s, st, t, d, drift in results:
                        if drift is not None:
                            belief_d, odom_d, n_conv, n_att = drift
                            print(f"      seed={s}: belief drift={belief_d:.3f}m, "
                                  f"odom drift={odom_d:.3f}m, ICP converged {n_conv}/{n_att}")
            print()


if __name__ == "__main__":
    main()
