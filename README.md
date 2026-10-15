# World Action Model (WAM)

A learned forward-dynamics "world model" for the Unitree Go2 quadruped: given the robot's
current state and a commanded action, predict what state comes next -- without running the
real physics simulator. Trained entirely on rollout data collected from the sibling
`go1_rl_walk` project's own MuJoCo Go2 + classical convex-MPC controller.

A separate project (and repo) from `go1_rl_walk` and `slam_sim` (sibling directories),
following the same convention: each new kind of problem gets its own repo, not a branch of
an existing one.

## Scoping decisions

- **Data source: reuse the real Go2 quadruped, not a toy environment.** `go1_rl_walk/mpc`'s
  `MPCResidualEnv` (real MuJoCo rigid-body dynamics + classical convex-MPC balance/footstep
  control) is already a working, verified simulator -- generating rollouts from it means the
  world model learns real quadruped dynamics (stance-phase impacts, balance recovery, slip)
  from the start, not an idealized toy. Zero files in `go1_rl_walk` are modified; this
  project only ever imports from it, the same convention `slam_sim`'s Stage 6+ established.
- **Action = the velocity command, not the 12-dim residual torque.** `MPCResidualEnv` has
  two control levers: `env.step(residual_action)` (a 12-dim per-tick torque correction on
  top of the MPC) and `env.set_command(speed, lateral, yaw_rate)` (the 3-dim target
  velocity the MPC steers toward). Every demo across `go1_rl_walk` and `slam_sim` that
  reached a final, adopted state used `residual_action = 0` (pure classical MPC, no RL
  correction) -- so the 3-dim velocity command is the control lever actually in use, and
  the natural "action" for a world model meant to eventually support planning/navigation
  (Stage 2's goal) the same way `go1_rl_walk/mpc/navigate_mpc.py` already does by hand.
- **State = a reduced physical state, not the full 47-dim RL observation.** `MPCResidualEnv`'s
  own observation already bundles `[v_body(3), w_body(3), roll, pitch, qpos(12), qvel(12),
  mpc_tau(12), cmd(3)]` (47 dims) -- this project predicts everything except `mpc_tau` (an
  internal control signal, not "world state") and `cmd` (that's the action, not something to
  predict), plus `base_height` (tracked separately in `info`, not in `obs`). 33 dims total:
  `[v_body(3), w_body(3), roll, pitch, base_height, qpos(12), qvel(12)]`.
- **Predict state DELTAS at a 50ms macro-step, not raw next-state at the 5ms physics tick.**
  `MPCResidualEnv` runs its control loop at 200Hz (5ms/tick) -- predicting a single tick's
  change is almost an identity function (not a meaningfully hard or demonstrable learning
  problem) and too fine-grained for a future planner to reason about. Every 10 physics
  ticks (50ms), with the velocity command held constant across them, is recorded as one
  world-model transition `(state_t, action, state_{t+1})` -- a timescale closer to what
  `go1_rl_walk`'s own navigation loop already re-plans at.
- **Low-dimensional state prediction first, not pixels or a learned latent space.** The
  simplest version of this problem (direct state-to-state regression, a small MLP) gets
  built and verified before anything more ambitious -- same staged discipline as
  `slam_sim`'s own kinematic-robot-first choice.
- **Stage 1 = prediction only; Stage 2 = use it for planning.** Verify the learned model's
  open-loop rollouts actually track the real simulator's before building anything on top of
  it. Only once that's trustworthy does it make sense to use the model for something (e.g.
  a learned-model MPC/planner, replacing or augmenting `navigate_mpc.py`'s hand-designed
  steering law) -- a user-requested two-part plan ("do 1 then 2"), not a speculative
  extension tacked on early.

## Staged plan

1. **Data collection + a simple MLP dynamics model, verified by open-loop rollout
   comparison against the real simulator** -- DONE.
2. **Use the learned model for planning/control** -- DONE.
3. **Obstacle-aware planning, reusing `slam_sim`'s lidar + occupancy map** -- DONE.
4. **Extend to uneven terrain** -- DONE (partial/honest result, see below).

### Stage 1: data collection, training, and open-loop verification

`collect_data.py` ran 50 training + 10 validation episodes (10s each, real Go2 + classical
MPC, random held-then-ramped velocity commands -- see "Scoping decisions"), yielding 9,806
training / 1,855 validation transitions (a handful of episodes fell short of the full 200
transitions/episode: rarer falls even with the ramped commands, discarded rather than kept
as contaminated data).

**First attempt (one-step loss only) failed open-loop, in an instructive way.** Training
`wam/model.py`'s MLP purely on one-step `(state, action) -> next_state` prediction looked
great by the metric it was trained on (val one-step MAE 0.16), but `eval_rollout.py`'s real
test -- feed the model's own predictions back in autoregressively for a full 10s, 200-step
held-out episode, no peeking at ground truth -- told a different story: it tracked the real
simulator closely for about 8 seconds, then **blew up exponentially** (roll reaching 6 rad,
base height "climbing" past 2m by t=10s). Overall rollout RMSE: 8.71. This is the classic
compounding-error failure mode of autoregressive models trained only on one-step loss, not
a bug in the data or the env -- confirmed by the plot showing a clean, accurate trajectory
for most of the episode before the error visibly starts compounding near t=8s.

**Fix: train with an H-step rollout loss, not just one-step.** `train_world_model.py` now
trains on a COMBINATION of the one-step loss (cheap, stable gradient) and a 20-step
(1.0s) autoregressive rollout loss -- unrolling the model itself during training and
penalizing the full trajectory's error, not just a single step's, the same approach
model-based RL methods (PETS, MBPO, Dreamer) use for exactly this reason. Result: overall
rollout RMSE dropped from 8.71 to **0.44** (20x), with every plotted quantity (base height,
roll, pitch, forward velocity, yaw rate) tracking the real simulator closely across the
*entire* 10-second, 200-step held-out episode -- no blowup. The model catches one real
transient (a ~0.03m base-height bounce around t=2.3s) a bit late but recovers and keeps
tracking afterward.

<p align="center">
  <img src="media/rollout_comparison.png" width="620" alt="Five time-series panels (base height, roll, pitch, forward velocity, yaw rate) comparing the real MuJoCo simulator (green, solid) against the learned world model's open-loop autoregressive rollout (purple, dashed) over a full 10-second held-out episode -- the two track closely throughout, including through several direction/speed changes">
</p>

Needs `go1_rl_walk/mpc`'s own venv (`mujoco`, `torch`, `casadi`/`pinocchio` all live there):

```bash
/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 collect_data.py \
    --episodes 50 --val-episodes 10 --out data/rollouts.npz
/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 train_world_model.py \
    --data data/rollouts.npz --checkpoint checkpoints/world_model.pt
/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_rollout.py \
    --checkpoint checkpoints/world_model.pt --record media/rollout_comparison.png
```

### Stage 2: a learned-model CEM planner, replacing the hand-designed steering law

The world model's own 33-dim state has no world-frame `(x, y, yaw)` in it (by design -- see
"Scoping decisions"), so a planner that wants to reason about a world-frame goal has to
dead-reckon pose itself from the state components the model *does* predict: body-frame
forward/lateral velocity and yaw-rate, integrated forward each step (`wam/pose.py`, same
body->world rotation convention `mpc_residual_env.py` itself uses). The planner
(`wam/planner.py`'s `CEMPlanner`) is a standard shooting method: sample many candidate
*constant* velocity commands, roll each one forward through the learned model for a short
horizon (10 macro-steps = 0.5s) while dead-reckoning pose alongside it, score each rollout
by final distance-to-goal + heading alignment + a stability penalty (keeping roll/pitch/
height near the training distribution's own typical standing values, since the model was
never trained on falls and can't be trusted to extrapolate into them), keep the best-scoring
"elite" samples, refit the sampling distribution around them, and repeat for 3 iterations
(cross-entropy method). Candidates are a single constant command across the whole horizon,
not a freely time-varying sequence, deliberately: Stage 1's training data only ever changed
its target command every 0.5s, so a constant-command rollout matches what the model
actually saw during training, where a faster-switching sequence would ask it to extrapolate
beyond its own training distribution.

`navigate_wam.py` drives the real Go2 (same zero-RL-residual, classical-MPC convention as
every other script here) toward a world-frame goal by replanning every 0.25s (5 macro-steps)
with the CEM planner and holding its chosen command in between (rate-limited the same way
`navigate_mpc.py`'s is, for the same QP-stability reason). Every tick still reads the REAL
ground-truth pose from the simulator for "where am I now" -- only the planner's internal
look-ahead rollout uses the learned model and its own dead-reckoned pose estimate to score
candidates.

**Evaluated against `go1_rl_walk/mpc/navigate_mpc.py`'s two hand-designed steering laws**
(`strafe`: the original crab-toward-goal law; `turn`: turn-to-face + drive forward), across
10 randomized (seed, goal) trials -- same seeds/goals for all three methods, goals 1.5-3.0m
away in a random direction, so the same initial state and target set is attempted by all
three. See "go1-eval-method": judged across repeated randomized trials, not a single run.

| method      | success | mean time-to-reach | mean final distance |
|-------------|---------|---------------------|----------------------|
| **wam (CEM planner)** | **10/10** | **3.28 s** | **0.043 m** |
| mpc strafe  | 10/10   | 5.27 s              | 0.136 m              |
| mpc turn    | 10/10   | 4.56 s              | 0.154 m              |

All three reached every one of the 10 goals with zero falls -- the learned-model planner is
just as reliable as the hand-designed laws here, and noticeably faster and more precise:
~30-38% faster to reach, and 3-4x tighter on final distance (closer than the 0.15m arrival
tolerance itself, vs. the analytic laws settling right around it). The gap is plausible, not
just noise: the CEM planner jointly optimizes all 3 command dimensions (speed, lateral,
yaw-rate) against the actual predicted outcome every replan, where `compute_nav_command`/
`compute_nav_command_turn` are decoupled proportional laws recomputed from instantaneous
error alone. Reproduce with:

```bash
/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_navigate.py \
    --trials 10 --seconds 15 --base-seed 200 --goal-seed 7
```

<p align="center">
  <img src="media/navigate_wam_demo.gif" width="480" alt="The real Go2, driven purely by the learned-model CEM planner's chosen velocity commands, turning to face and driving to a goal requiring roughly a 180-degree turn">
</p>

```bash
/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 navigate_wam.py \
    --goal-x 3.0 --goal-y 1.0 --checkpoint checkpoints/world_model.pt --record out.gif
```

### Stage 3: obstacle-aware planning, reusing `slam_sim`'s lidar + occupancy map

Everything sensing/mapping-related here is reused COMPLETELY UNMODIFIED from the sibling
`slam_sim` project's own earlier stages, not re-derived: `slam.world.World.loop_corridor()`
(a 12x8m rectangle with a 4x4m island in the middle -- a straight line between opposite
corners is blocked, forcing genuine routing around it), `demo_stage6_go2_world.py`'s
`build_env_xml`/wall-injection for putting that island into the real Go2's MuJoCo scene,
`slam.mj_lidar.mj_lidar_scan`'s real ray-cast lidar, and `slam.occupancy_grid.OccupancyGrid`
for the log-odds map those scans build up over time. This stage only adds the planning
side: `wam/planner.py`'s `CEMPlanner` now accepts an `obstacle_points` argument (the
occupancy grid's currently-known occupied cell centers) and scores every candidate
rollout's predicted (x, y) against the nearest one -- a soft quadratic penalty inside a
0.5m safety margin, plus a large flat penalty if a rollout predicts actually crossing a
0.35m collision margin. Deliberately uses the REAL ground-truth pose (not `slam_sim`'s own
ICP-corrected belief pose) to build the map -- localization-under-drift is a separate,
already-solved problem over in `slam_sim`'s Stage 4/6; this stage's new ground is planning
around what's been sensed, not re-demonstrating localization. One real consequence: the map
only contains what's been scanned so far, so the planner can only avoid what it has already
seen, same as any real lidar-then-plan stack.

**Confirmed the obstacle cost is actually necessary, not just harmless:** with it disabled
(`--disable-obstacle-cost`, Stage 2's planner unchanged), the robot drives straight at the
island and falls/QP-fails almost immediately (final distance 7.6-7.8m from a ~10.3m start).
With it enabled, the same goal is reached safely, routing around the island:

<p align="center">
  <img src="media/navigate_wam_obstacles_baseline_crash.gif" width="340" alt="Without the obstacle cost, the planner drives the real Go2 straight at the island obstacle and it falls">
  <img src="media/navigate_wam_obstacles_robot.gif" width="340" alt="With the obstacle cost, the planner routes the real Go2 around the island to reach the same goal">
</p>
<p align="center">
  <img src="media/navigate_wam_obstacles_map.gif" width="480" alt="Overhead view: the occupancy grid building up from lidar scans as the robot (green trail) routes around the island toward the goal">
</p>

**Evaluated across 4 start/goal pairs in the same world** (one with no obstacle in the way,
as a sanity check the planner doesn't detour needlessly; three forcing a real detour around
different sides/corners of the island):

| start &rarr; goal | obstacle in the way? | result |
|---|---|---|
| (1.5,1.5) &rarr; (1.5,6.5) | no (clear corridor) | reached in 5.9s -- no needless detour |
| (1.5,1.5) &rarr; (10.5,6.5) | yes | reached in 30.5-31.6s (reproduced twice) |
| (10.5,1.5) &rarr; (1.5,6.5) | yes | safe, but did not finish in 45s (final distance 2.96m) |
| (1.5,6.5) &rarr; (10.5,1.5) | yes | safe, but did not finish in 45s (final distance 7.02m) |

Zero falls across all 4 trials, including the two that didn't finish -- the obstacle cost's
core safety claim held in every case. But it's not a uniformly efficient router: the two
unfinished trials both stall near a sharp convex corner of the island, creeping at a few
cm/s rather than committing to a decisive turn around it (one of these, given more time,
eventually broke free and finished at 31.3s -- not stuck forever, just slow and
inconsistent). This is a real, explainable limitation of this specific design, not a bug:
the obstacle cost only ever looks at where the planner's own SHORT (0.5s), DEAD-RECKONED
rollout predicts the robot will be -- near a sharp corner, small pose-integration error
inside that short horizon can make a genuinely-clear turn look like it still clips the
safety margin, so the CEM settles for a locally-safe near-zero-progress crawl instead. A
geometric path planner (e.g. A* over the occupancy grid, feeding the CEM planner a
waypoint beyond the corner rather than the raw goal) is the standard fix for exactly this
failure mode -- a natural follow-on, not attempted here to keep this stage's own scope to
"can a learned-model planner be made obstacle-aware at all", which it demonstrably can.

```bash
/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 navigate_wam_obstacles.py \
    --goal-x 10.5 --goal-y 6.5 --checkpoint checkpoints/world_model.pt \
    --record-robot media/robot.gif --record-map media/map.gif
```

### Stage 4: extend to uneven terrain -- a genuine but partial result

`MPCResidualEnv`'s own `terrain_amplitude_range` constructor argument already resamples a
fresh random bump amplitude at every `reset()` (same mechanism it uses for
speed/lateral/yaw-rate ranges) -- so `collect_data.py` and `eval_rollout.py` both gained a
`--terrain-amplitude-max` flag that just threads a `(0.0, max)` range through, no new
terrain-generation code needed.

**Confirmed the generalization gap first, not assumed it:** Stage 1's flat-only checkpoint's
open-loop rollout RMSE on terrain (0.08m bumps) is meaningfully worse than on flat ground --
across 6 held-out seeds, mean RMSE 0.62 (flat) vs. **1.33** (terrain), driven mostly by leg
joint-velocity error (qvel RMSE up to ~1.5-2x worse per-joint on terrain). Expected: terrain
disturbances are an exogenous input the model never saw, not something the (state, action)
pair alone can anticipate.

**Collected a terrain-inclusive dataset and retrained** (`collect_data.py --terrain-amplitude-
max 0.08`, 50 train + 10 val 10s episodes, 9708/1928 transitions -- comparable size to Stage
1's flat dataset). The first training run surfaced a real instability: `one_step_loss` and
`rollout_train_loss` both suddenly spiked ~8x around epoch 100 (one outlier batch through the
H-step rollout loss's repeated model composition producing an unusually large gradient --
not seen on flat data, where the gradient magnitudes stay smaller throughout) and only
partially recovered by epoch 199. **Fixed with gradient-norm clipping** (`--grad-clip 1.0`,
now on by default in `train_world_model.py`) -- the retrained run's loss curve was smooth
throughout, with a better best checkpoint (val rollout MAE 0.297 vs. 0.340 without clipping).

**Evaluated across 6 held-out seeds per condition** (`eval_terrain.py`), not a single
episode -- a single episode's RMSE was actually MISLEADING here: the first terrain-trained
checkpoint happened to score *better* than the flat checkpoint on one specific held-out
episode and *worse* on another, which would have supported opposite conclusions depending
on which seed got tested. See [[go1-review-before-trusting]].

| condition | mean RMSE | std |
|---|---|---|
| flat-trained model, flat ground | 0.62 | 0.23 |
| flat-trained model, terrain | 1.33 | 0.73 |
| terrain-trained model, flat ground | 0.86 | 0.35 |
| terrain-trained model, terrain | 1.03 | 0.43 |

**Honest conclusion: a real but partial improvement, not a clean fix like Stages 1-3.**
Training on terrain data measurably helps on terrain (mean RMSE down ~22%, std down ~40% --
fewer of the occasional much-worse episodes the flat model produced on terrain) but doesn't
fully close the gap back to flat-ground-level accuracy, and costs a real regression on flat
ground itself (0.62 -> 0.86). This is the ordinary cost of broadening a training
distribution to cover a harder case, not a bug -- the same trade-off `go1_rl_walk`'s own
flat-vs-random-terrain RL staging ran into (see [[go1-staged-plan]]).

**At the downstream navigation level, this difference didn't clearly show up.** Ran
`navigate_wam.py` with `--terrain-amplitude 0.08` across the same 5 (seed, goal) pairs for
both checkpoints: both reached 4/5 goals with similar times and final distances, and both
fell on the SAME seed (200) -- suggesting that particular terrain realization was just hard
for the real classical-MPC controller regardless of which learned model informed planning,
not a world-model-accuracy effect. The modest rollout-accuracy edge the terrain-trained
model has didn't translate into a visibly better navigation success rate at this sample
size -- a genuinely inconclusive result at the planning level, reported as such rather than
rounded up to a win.

```bash
/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 collect_data.py \
    --terrain-amplitude-max 0.08 --episodes 50 --val-episodes 10 --out data/rollouts_terrain.npz
/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 train_world_model.py \
    --data data/rollouts_terrain.npz --checkpoint checkpoints/world_model_terrain.pt
/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_terrain.py --trials 6
```

<p align="center">
  <img src="media/rollout_comparison_terrain.png" width="620" alt="Five time-series panels comparing the real MuJoCo simulator on terrain (green, solid) against the terrain-trained world model's open-loop autoregressive rollout (purple, dashed)">
</p>

## Repository layout

```
wam/
├── wam/
│   ├── state.py     # the 33-dim state / 3-dim action this project predicts, sliced from
│   │                  MPCResidualEnv's own 47-dim obs + info
│   ├── model.py       # the MLP dynamics model itself (delta prediction, built-in normalization)
│   ├── pose.py          # dead-reckon world (x, y, yaw) from the model's predicted v_body/w_body
│   └── planner.py         # Stage 2/3: CEM shooting planner, optional obstacle cost
├── collect_data.py      # Stage 1: generate rollouts from the real Go2 + classical MPC
├── train_world_model.py   # Stage 1: one-step + H-step rollout loss training
├── eval_rollout.py          # Stage 1: open-loop rollout vs. real simulator, numeric + plot
├── navigate_wam.py            # Stage 2: drive the real Go2 to a goal via the CEM planner
├── eval_navigate.py              # Stage 2: multi-seed comparison vs. navigate_mpc.py's laws
├── navigate_wam_obstacles.py       # Stage 3: obstacle-aware planning in slam_sim's island world
├── eval_terrain.py                   # Stage 4: multi-seed flat-vs-terrain model comparison
├── data/rollouts.npz          # collected train/val transitions (flat ground)
├── data/rollouts_terrain.npz    # Stage 4: collected train/val transitions (terrain)
├── checkpoints/world_model.pt   # Stage 1-3's checkpoint (flat-ground data only)
├── checkpoints/world_model_terrain.pt  # Stage 4's checkpoint (terrain-inclusive data)
├── media/rollout_comparison.png  # Stage 1's verification plot
├── media/rollout_comparison_terrain.png  # Stage 4's verification plot (on terrain)
├── media/navigate_wam_demo.gif    # Stage 2's planner-driven navigation demo
├── media/navigate_wam_obstacles_*.gif  # Stage 3's before/after + occupancy-map demos
├── requirements.txt
└── README.md
```
