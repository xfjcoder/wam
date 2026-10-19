"""Stage 15: does `--noisy-localization` (Stage 13's lidar + ICP belief pose) still work when
the island world's ground is ALSO uneven (Stage 4/5's terrain)? Neither prior stage tested this
combination -- Stage 13 was flat-ground only; Stage 4/5's terrain checks always used
ground-truth pose. A genuinely new interaction worth checking: `slam.mj_lidar.mj_lidar_scan`
casts rays from a FIXED height relative to (x, y, yaw) assuming a level body -- terrain-induced
pitch/roll (the robot's base tilting as it steps over bumps) could distort the scan geometry in
a way the flat-ground numbers never exercised, independent of anything ICP itself can fix.

Same style as `eval_obstacles_terrain.py`/`eval_localization.py`: multi-seed, terrain-trained
checkpoint, terrain ENABLED, same 4 start/goal pairs, ground-truth vs. noisy-localization vs.
noisy + `--icp-degeneracy-filter`.

Usage:
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_localization_terrain.py \
        --seeds 3 --base-seed 300 --terrain-amplitude-max 0.08 --seconds 45
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


def run_one(start_x, start_y, goal_x, goal_y, seed, seconds, checkpoint, terrain_amplitude_max,
            noisy_localization, degeneracy_filter=False):
    cmd = ["/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3",
           str(Path(__file__).resolve().parent / "navigate_wam_obstacles.py"),
           "--start-x", str(start_x), "--start-y", str(start_y),
           "--goal-x", str(goal_x), "--goal-y", str(goal_y),
           "--checkpoint", checkpoint,
           "--terrain-amplitude-max", str(terrain_amplitude_max),
           "--seed", str(seed), "--seconds", str(seconds)]
    if noisy_localization:
        cmd.append("--noisy-localization")
    if degeneracy_filter:
        cmd.append("--icp-degeneracy-filter")
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
    ap.add_argument("--base-seed", type=int, default=300)
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--checkpoint", type=str, default="checkpoints/world_model_terrain.pt")
    ap.add_argument("--terrain-amplitude-max", type=float, default=0.08)
    args = ap.parse_args()

    conditions = [
        (False, False, "GROUND TRUTH POSE (terrain, Stage 4/5's own baseline)"),
        (True, False, "NOISY LOCALIZATION (lidar+ICP belief pose, terrain enabled)"),
        (True, True, "NOISY LOCALIZATION + --icp-degeneracy-filter"),
    ]

    for noisy_localization, degeneracy_filter, label in conditions:
        print(f"=== {label} ===")
        for name, sx, sy, gx, gy in PAIRS:
            results = []
            for i in range(args.seeds):
                seed = args.base_seed + i
                status, t, final_dist, drift = run_one(
                    sx, sy, gx, gy, seed, args.seconds, args.checkpoint,
                    args.terrain_amplitude_max, noisy_localization, degeneracy_filter)
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
