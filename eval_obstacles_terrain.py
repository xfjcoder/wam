"""Stage 5: does obstacle-aware planning (Stage 3b: CEM obstacle cost + A* global waypoints)
still work when the island world's ground is ALSO uneven (Stage 4's terrain)? Neither prior
stage tested this combination -- Stage 3/3b were flat-ground only, Stage 4's downstream
navigation check used the open, obstacle-free default world.

Multi-seed (same style as eval_obstacles.py/eval_terrain.py) comparison across the same 4
start/goal pairs, terrain ENABLED, flat-trained vs terrain-trained checkpoint -- since
Stage 4 already showed a single seed/episode can flip which checkpoint "wins".

Usage:
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_obstacles_terrain.py \
        --seeds 3 --base-seed 300 --terrain-amplitude-max 0.08 --seconds 45
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


def run_one(start_x, start_y, goal_x, goal_y, seed, seconds, checkpoint, terrain_amplitude_max,
            terrain_footstep_planning):
    cmd = ["/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3",
           str(Path(__file__).resolve().parent / "navigate_wam_obstacles.py"),
           "--start-x", str(start_x), "--start-y", str(start_y),
           "--goal-x", str(goal_x), "--goal-y", str(goal_y),
           "--checkpoint", checkpoint,
           "--terrain-amplitude-max", str(terrain_amplitude_max),
           "--seed", str(seed), "--seconds", str(seconds)]
    if terrain_footstep_planning:
        cmd.append("--terrain-footstep-planning")
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
    ap.add_argument("--base-seed", type=int, default=300)
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--terrain-amplitude-max", type=float, default=0.08)
    ap.add_argument("--flat-checkpoint", type=str, default="checkpoints/world_model.pt")
    ap.add_argument("--terrain-checkpoint", type=str, default="checkpoints/world_model_terrain.pt")
    ap.add_argument("--terrain-footstep-planning", action="store_true")
    args = ap.parse_args()

    for checkpoint, label in [(args.flat_checkpoint, "flat-trained model"),
                               (args.terrain_checkpoint, "terrain-trained model")]:
        print(f"=== {label} ({checkpoint}), TERRAIN amplitude_max={args.terrain_amplitude_max} ===")
        n_reached, n_total = 0, 0
        for name, sx, sy, gx, gy in PAIRS:
            results = []
            for i in range(args.seeds):
                seed = args.base_seed + i
                status, t, final_dist = run_one(sx, sy, gx, gy, seed, args.seconds,
                                                 checkpoint, args.terrain_amplitude_max,
                                                 args.terrain_footstep_planning)
                results.append((seed, status, t, final_dist))
                n_total += 1
                n_reached += int(status == "reached")
            summary = ", ".join(
                f"seed={s}:{st}" + (f"@{t:.1f}s" if t is not None else f"(d={d:.2f}m)")
                for s, st, t, d in results)
            print(f"  {name:32s} ({sx},{sy})->({gx},{gy}): {summary}")
        print(f"  -> {n_reached}/{n_total} trials reached the goal\n")


if __name__ == "__main__":
    main()
