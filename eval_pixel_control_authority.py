"""Stage 12 mini-plan, second half of the one untested hypothesis left at closure: steps
19/20's closing note offered two candidate fixes for "close-quarters maneuvering/control
accuracy near the island's corners" -- (1) tighter CEM safety margins around A* waypoints
(tested in eval_pixel_margins.py, a clean negative: 0/5 reached either way) and (2) trajectory-
tracking/control authority near obstacles. This tests (2): capping how fast the commanded
forward/yaw-rate setpoint can change (`--max-accel`/`--max-yaw-accel`) so a newly-replanned
waypoint can't demand the "sharp commanded turn" step 20's trace blamed for the fall, at the
cost of slower reaction to a legitimately close call.

Single lever vs. step 20's exact defaults, same checkpoints/seeds, nothing else changed.

Usage:
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_pixel_control_authority.py --seeds 5
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


def run_one(seed, seconds, max_accel, max_yaw_accel):
    cmd = ["/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3",
           str(Path(__file__).resolve().parent / "navigate_wam_pixel.py"),
           "--pixel-checkpoint", PIXEL_CKPT, "--cost-checkpoint", COST_CKPT,
           "--seed", str(seed), "--seconds", str(seconds),
           "--max-accel", str(max_accel), "--max-yaw-accel", str(max_yaw_accel)]
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

    for max_accel, max_yaw_accel, label in [
            (1.5, 3.0, "max-accel=1.5 max-yaw-accel=3.0 (step 20 default)"),
            (0.8, 1.2, "max-accel=0.8 max-yaw-accel=1.2 (gentler, capped turn authority)")]:
        print(f"=== {label} ===")
        results = []
        for i in range(args.seeds):
            seed = args.base_seed + i
            status, t, final_dist = run_one(seed, args.seconds, max_accel, max_yaw_accel)
            results.append((seed, status, t, final_dist))
            tag = f"@{t:.1f}s" if t is not None else f"(d={final_dist:.2f}m)"
            print(f"  seed={seed}: {status}{tag}")
        n_reached = sum(1 for _, st, _, _ in results if st == "reached")
        print(f"  -> {n_reached}/{args.seeds} reached")
        print()


if __name__ == "__main__":
    main()
