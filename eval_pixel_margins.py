"""Stage 12 mini-plan, reopened with one new hypothesis: steps 19/20 closed with "the
remaining failures look like close-quarters maneuvering/control accuracy near the island's
corners... tighter CEM safety margins specifically around A* waypoints" as the one thing left
untried. This tests exactly that, nothing else.

The concrete mismatch: `navigate_wam_pixel.py`'s A* inflates obstacles by `--robot-radius`
(default 0.35) when deciding which grid cells are passable, but the CEM planner's OWN
`--pixel-safety-margin` (default 0.6) wants MORE clearance than that before it's happy -- so
A* can hand the CEM a waypoint that threads a 0.35m-clearance line past the island corner, and
the CEM then has to swerve hard (a "sharp commanded turn", exactly what step 20's trace
observed right before the fall) to retroactively honor its own larger margin. Matching
`--robot-radius` to `--pixel-safety-margin` removes that conflict at the source: A* routes
waypoints the CEM will actually be happy to track in a straight-ish line.

Single lever, nothing else changed (checkpoints, seeds, n-beams, everything else stays at
step 20's exact defaults) -- this project's standing discipline of testing one hypothesis at a
time, not combining this with the OTHER half of the closing note (trajectory-tracking/control-
authority tuning, e.g. --max-yaw-accel), which stays untested unless this one doesn't pan out.

Usage:
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_pixel_margins.py --seeds 5
"""
import argparse
import re
import subprocess
from pathlib import Path

REACHED_RE = re.compile(r"Reached goal .* in ([\d.]+)s, final distance ([\d.]+) m")
FELL_RE = re.compile(r"FELL/QP-FAILED .* \(final distance ([\d.]+) m\)")
TIMEOUT_RE = re.compile(r"Did NOT reach the goal .* \(final distance ([\d.]+) m\)")

PIXEL_CKPT = "checkpoints/pixel_world_model_wall_explore_depth.pt"
COST_CKPT = "checkpoints/pixel_cost_head_explore_far.pt"


def run_one(seed, seconds, robot_radius):
    cmd = ["/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3",
           str(Path(__file__).resolve().parent / "navigate_wam_pixel.py"),
           "--pixel-checkpoint", PIXEL_CKPT, "--cost-checkpoint", COST_CKPT,
           "--seed", str(seed), "--seconds", str(seconds),
           "--robot-radius", str(robot_radius)]
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
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    for robot_radius, label in [(0.35, "robot-radius=0.35 (step 20 default, A*/CEM margins MISMATCHED)"),
                                 (0.6, "robot-radius=0.60 (matches --pixel-safety-margin)")]:
        print(f"=== {label} ===")
        results = []
        for i in range(args.seeds):
            seed = args.base_seed + i
            status, t, final_dist = run_one(seed, args.seconds, robot_radius)
            results.append((seed, status, t, final_dist))
            tag = f"@{t:.1f}s" if t is not None else f"(d={final_dist:.2f}m)"
            print(f"  seed={seed}: {status}{tag}")
        n_reached = sum(1 for _, st, _, _ in results if st == "reached")
        print(f"  -> {n_reached}/{args.seeds} reached")
        print()


if __name__ == "__main__":
    main()
