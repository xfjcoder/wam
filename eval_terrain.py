"""Stage 4 evaluation: does training on terrain-inclusive rollout data actually make the
world model's open-loop rollout MORE accurate on terrain, without regressing its accuracy
on flat ground? A single held-out episode's RMSE is noisy (confirmed directly: the same
two checkpoints swapped which one "won" depending on which seed's episode was tested) --
see memory "go1-eval-method": judge across repeated randomized trials, not a single run.

Runs eval_rollout.py as a subprocess across multiple seeds for every (checkpoint, terrain)
combination and reports mean +/- std RMSE, same reason eval_navigate.py shells out to
navigate_mpc.py rather than importing its main() -- simpler and less error-prone than
refactoring a script that wasn't written to be called as a library.

Usage (needs go1_rl_walk/mpc's own venv):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_terrain.py --trials 6
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

WAM_DIR = Path(__file__).resolve().parent
PYTHON = WAM_DIR.parent / "go1_rl_walk" / "mpc" / "venv" / "bin" / "python3"

RMSE_RE = re.compile(r"Overall RMSE across all \d+ state dims: ([\d.]+)")


def run_eval(checkpoint: str, seed: int, terrain_amplitude_max: float) -> float:
    cmd = [str(PYTHON), str(WAM_DIR / "eval_rollout.py"),
           "--checkpoint", checkpoint, "--seed", str(seed),
           "--terrain-amplitude-max", str(terrain_amplitude_max),
           "--record", "/tmp/eval_terrain_scratch.png"]
    proc = subprocess.run(cmd, cwd=str(WAM_DIR), capture_output=True, text=True, timeout=60)
    m = RMSE_RE.search(proc.stdout)
    if not m:
        print(f"  [warn] could not parse RMSE for checkpoint={checkpoint} seed={seed}:\n"
              f"{proc.stdout[-500:]}\n{proc.stderr[-500:]}", file=sys.stderr)
        return float("nan")
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--base-seed", type=int, default=500)
    ap.add_argument("--terrain-amplitude-max", type=float, default=0.08)
    ap.add_argument("--flat-checkpoint", type=str, default="checkpoints/world_model.pt")
    ap.add_argument("--terrain-checkpoint", type=str, default="checkpoints/world_model_terrain.pt")
    args = ap.parse_args()

    seeds = [args.base_seed + i for i in range(args.trials)]
    conditions = [
        ("flat-trained model, FLAT ground", args.flat_checkpoint, 0.0),
        ("flat-trained model, TERRAIN", args.flat_checkpoint, args.terrain_amplitude_max),
        ("terrain-trained model, FLAT ground", args.terrain_checkpoint, 0.0),
        ("terrain-trained model, TERRAIN", args.terrain_checkpoint, args.terrain_amplitude_max),
    ]

    print(f"=== {args.trials} held-out seeds per condition: {seeds} ===\n")
    results = {}
    for label, checkpoint, terrain_amp in conditions:
        rmses = [run_eval(checkpoint, seed, terrain_amp) for seed in seeds]
        results[label] = rmses
        vals = ", ".join(f"{r:.3f}" for r in rmses)
        print(f"{label:38s}: [{vals}]")

    print(f"\n{'condition':38s} {'mean RMSE':10s} {'std':8s}")
    for label, _, _ in conditions:
        rmses = np.array(results[label])
        print(f"{label:38s} {np.nanmean(rmses):10.4f} {np.nanstd(rmses):8.4f}")


if __name__ == "__main__":
    main()
