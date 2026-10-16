# WAM Tutorial: Learning and Using a World Model for a Real Quadruped

This is a hands-on walkthrough of the WAM ("World Action Model") project: what it is,
why each piece is built the way it is, and how to run it yourself. For the full
build log (exact results, dead ends, honest limitations) see [README.md](README.md) --
this document is the teaching path through the same codebase, not a replacement for it.

## What problem is this solving?

Most robot controllers in the sibling `go1_rl_walk` project either hand-design a control
law (classical MPC) or learn one end-to-end with RL. WAM asks a third question: **can you
learn a cheap, approximate simulator of the robot instead** -- predict what state comes
next given the current state and a command, without running real physics -- and then use
that *learned simulator* for planning, the way a person mentally rehearses "if I push
here, the robot will probably tip that way" before acting?

This is the "world model" idea from model-based RL (PETS, MBPO, Dreamer): train a
dynamics model, then plan against it instead of against reality. The appeal is sample
efficiency and controllability -- once you have a model, you can ask it arbitrary
"what if" questions (what if I go here instead?) without touching the real robot/simulator
at all.

## Before you run anything

Every script in this repo needs both `mujoco` (to run `go1_rl_walk/mpc`'s `MPCResidualEnv`
simulator) and `torch` (to train/run the world model). Neither is installed in a venv of
its own here -- everything reuses `go1_rl_walk/mpc`'s existing venv:

```bash
/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 <script>.py ...
```

You'll also want `slam_sim` checked out as a sibling directory (`../slam_sim`) for the
obstacle-avoidance stage -- its lidar/occupancy-grid code is imported directly, never
copied.

## Part 1: the world model itself

### What it predicts

Open `wam/state.py`. The model doesn't predict MuJoCo's full internal state -- it predicts
a 33-number summary: body-frame linear/angular velocity (6), roll/pitch (2), base height
(1), and all 12 joint positions + 12 joint velocities (24). That's deliberately *not* the
full 47-dim observation `MPCResidualEnv` exposes to its own RL residual -- two fields are
dropped: `mpc_tau` (the classical controller's internal torque, not "world state") and
`cmd` (that's the *action*, not something to predict).

The action is 3 numbers: `(target_speed, target_lateral, target_yaw_rate)` -- the same
velocity command `navigate_mpc.py` already drives the real robot with. Not the 12-dim
joint-torque residual `MPCResidualEnv.step()` also accepts; every demo in this whole
project family runs with that residual at zero, so the velocity command is the control
lever actually in use.

### Why predict a delta, not the raw next state

Open `wam/model.py`. `WorldModel.forward()` returns `state + predicted_delta`, not the
delta itself. Most of the 33 numbers barely change in 50ms (a joint angle moves a few
degrees, base height barely shifts) -- regressing the raw next value would waste the
network's capacity relearning "give me back approximately what I was just given," rather
than learning the much more informative *change*. Both the state and the delta are
Z-score normalized internally (as model buffers, so they travel with the checkpoint) --
this is just good regression hygiene, not something conceptually deep.

### Collecting training data

```bash
/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 collect_data.py \
    --episodes 50 --val-episodes 10 --out data/rollouts.npz
```

This runs the real `MPCResidualEnv` (classical convex-MPC controller, zero RL residual)
with randomly sampled velocity commands, held constant for `--hold-ticks` physics ticks
(0.5s) then resampled -- and *ramped* toward each new target rather than snapped
instantly (an instant jump destabilizes the classical MPC; this is the same rate-limiting
trick `navigate_mpc.py` uses). Every `--macro-k` ticks (50ms), one `(state, action,
next_state)` transition is recorded. You get back an `.npz` with train/val splits.

### Training, and the mistake worth understanding

```bash
/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 train_world_model.py \
    --data data/rollouts.npz --checkpoint checkpoints/world_model.pt
```

If you only trained on one-step prediction error, you'd get a model that looks *great* by
its own metric (next-state error near zero) but is actually useless for planning. Here's
why: a planner doesn't call the model once -- it calls it over and over, feeding each
prediction back in as the next input (autoregressive rollout), exactly like the model
"imagining" several seconds into the future. A model trained only on isolated one-step
pairs has no incentive to be *self-consistent* across many such calls; tiny errors
compound, and by a few seconds in the "robot" it's imagining has usually face-planted
through the floor or is spinning at 50 rad/s.

`train_world_model.py` fixes this directly: alongside the cheap one-step loss, every
epoch also unrolls the model `--rollout-horizon` steps (default 20, i.e. 1 second)
*autoregressively*, comparing the whole imagined trajectory against the real one, and
backpropagates through the whole unroll. This is standard in model-based RL (PETS/MBPO/
Dreamer all do this), and it's the difference between a model that's merely accurate and
one that's actually *usable*.

One more thing worth knowing if you ever retrain on a new data distribution: backpropagating
through 20 repeated applications of the same network is a form of BPTT, and BPTT
occasionally produces an outlier-large gradient from one unlucky batch, bad enough to
visibly corrupt the shared weights for many epochs afterward (this happened for real when
terrain data was introduced later -- see README's Stage 4). `--grad-clip 1.0` (on by
default) guards against this.

### Checking it actually works

```bash
/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_rollout.py \
    --checkpoint checkpoints/world_model.pt --record media/rollout_comparison.png
```

This is the real test: run a full held-out episode through the *actual* MuJoCo simulator,
then separately unroll the model autoregressively (its own predictions fed back in, never
peeking at ground truth) for the same length, and compare. A good training-loss number is
not sufficient evidence the model works -- this open-loop rollout comparison is the thing
that's actually load-bearing, and it's what caught the one-step-only model's compounding-
error failure the first time around.

<p align="center">
  <img src="media/rollout_comparison.png" width="620" alt="Five time-series panels: real simulator vs. the world model's own autoregressive rollout, closely tracking for the full episode">
</p>

## Part 2: planning with the model (CEM)

Now you have a function `next_state = model(state, action)`. How do you use it to *choose*
actions, not just predict their effects?

### The shooting-method idea

Open `wam/planner.py`. `CEMPlanner` is a cross-entropy-method shooting planner -- the same
family as MPC/MPPI/PETS, just shooting through a *learned* model instead of a hand-written
one:

1. Sample many candidate actions from a Gaussian.
2. Roll each one forward through the model for a short horizon, scoring the outcome.
3. Keep the best-scoring ("elite") samples.
4. Refit the Gaussian to the elites' mean/std.
5. Repeat a few iterations, then commit to the final mean as the chosen action.

One deliberate, non-obvious choice: each candidate is a *single constant command* held
across the whole horizon, not a time-varying sequence. This matches how the training data
was collected (a command held for 0.5s before resampling) -- asking the model to
extrapolate to an action-switching rate it never saw during training would be asking it
to generalize somewhere it has no evidence about.

### Why you need a pose on top of a state with no pose in it

The 33-dim state has no world-frame `(x, y, yaw)` in it (that was a deliberate Stage-1
scoping choice -- see `wam/state.py`). But a planner chasing a world-frame goal needs to
know where its imagined rollout *is*. `wam/pose.py`'s `integrate_pose` dead-reckons
`(x, y, yaw)` forward from the model's own predicted body-frame velocity and yaw-rate,
purely *inside* the planner's imagined rollout -- the real executed pose the robot
actually ends up at always comes straight from the simulator's ground truth, never from
this integration. Confusing those two would be a subtle, serious bug: dead-reckoning
accumulates error, which is fine for a half-second imagined rollout but would be
disastrous if used as the robot's actual belief about where it is.

### Driving the real robot

```bash
/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 navigate_wam.py \
    --goal-x 3.0 --goal-y 1.0 --checkpoint checkpoints/world_model.pt --record out.gif
```

`navigate_wam.py` replans every `--replan-every` macro-steps, feeding the planner the
*real* current state/pose each time, and rate-limits the chosen command before sending it
to the simulator (the same QP-stability reason `navigate_mpc.py` already rate-limits its
own commands). Compare it against the project's existing hand-designed steering laws:

```bash
/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_navigate.py --trials 10
```

<p align="center">
  <img src="media/navigate_wam_demo.gif" width="480" alt="The CEM planner driving the real Go2 to a goal requiring roughly a 180-degree turn">
</p>

## Part 3: obstacle avoidance (reusing `slam_sim`, not reinventing it)

`navigate_wam_obstacles.py` adds a lidar-built occupancy map and an obstacle cost to the
exact same planner -- but notice it imports `slam_sim`'s `World`, `OccupancyGrid`, and
`mj_lidar_scan` directly, unmodified. This project's whole convention (shared with
`go1_rl_walk` imports) is: never duplicate sensing/mapping code that's already built and
tested in a sibling repo.

The cost addition itself is small -- in `CEMPlanner.plan()`, each rollout step's predicted
`(x, y)` is scored against the nearest currently-known occupied cell: a soft quadratic
penalty inside a safety margin, a large flat penalty if a rollout would cross a collision
margin. No change to the shooting method itself, just one more additive term.

```bash
/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 navigate_wam_obstacles.py \
    --goal-x 10.5 --goal-y 6.5 --checkpoint checkpoints/world_model.pt \
    --record-robot media/robot.gif --record-map media/map.gif
```

<p align="center">
  <img src="media/navigate_wam_obstacles_baseline_crash.gif" width="340" alt="Without the obstacle cost: drives straight into the island obstacle">
  <img src="media/navigate_wam_obstacles_robot.gif" width="340" alt="With the obstacle cost: routes around the island">
</p>

### When local reasoning isn't enough: adding a global planner

The obstacle cost above only ever looks a few tenths of a second ahead (the CEM horizon).
Near a sharp corner, that short a horizon can make a perfectly clear turn *look* blocked
(small dead-reckoning error inside the horizon is enough to make the margin look violated)
-- the planner can get stuck creeping at a few cm/s instead of committing to the turn.
`wam/global_planner.py` fixes this the standard way: A* over the same occupancy grid
(inflated by the robot's own collision margin), simplified by greedy line-of-sight
"string-pulling" down to a few long legs, recomputed fresh every replan. The CEM is then
handed the nearest *waypoint* as its goal instead of the real, possibly-behind-a-corner
goal -- so it only ever has to solve the easy "walk toward this nearby point" problem, and
the global path handles the actual routing.

<p align="center">
  <img src="media/navigate_wam_astar_robot.gif" width="340" alt="With A* waypoints, the robot smoothly rounds a corner instead of stalling">
  <img src="media/navigate_wam_astar_map.gif" width="340" alt="The A* path bending around the obstacle, drawn over the occupancy grid">
</p>

This is a nice illustration of a general pattern worth internalizing: a short-horizon
*local* optimizer (the CEM) and a whole-map *global* planner (A*) solve genuinely
different problems, and combining "global planner picks a nearby sub-goal, local
optimizer chases it" is usually a better division of labor than trying to make either one
solve the whole thing alone.

## Part 4: generalizing to uneven terrain

Everything above was trained on flat ground. `collect_data.py`/`eval_rollout.py` both
accept `--terrain-amplitude-max` (threading straight into `MPCResidualEnv`'s own
`terrain_amplitude_range`, no new terrain-generation code needed) -- so you can collect a
second dataset and train a second checkpoint on bumpy ground:

```bash
/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 collect_data.py \
    --terrain-amplitude-max 0.08 --episodes 50 --val-episodes 10 \
    --out data/rollouts_terrain.npz
/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 train_world_model.py \
    --data data/rollouts_terrain.npz --checkpoint checkpoints/world_model_terrain.pt
```

The lesson worth taking from this stage (full numbers in README) isn't "terrain training
fixes everything" -- it's a textbook generalization trade-off: the terrain-trained model
gets meaningfully better on terrain, but gives up some accuracy on flat ground, and the
improvement doesn't obviously translate into better downstream navigation success. That's
a completely ordinary cost of broadening a training distribution, not a bug -- and it's
exactly the kind of thing a single-episode check would have reported inconsistently
(this project caught a real case of a single held-out episode "voting" for the wrong
model, which is why every claim here is backed by a multi-seed comparison, never one run).

```bash
/home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 eval_terrain.py --trials 6
```

## Running a multi-seed comparison yourself

Every claim in this project is backed by a script like `eval_navigate.py`/
`eval_terrain.py`/`eval_obstacles.py` -- never a single demo run. The pattern, if you want
to check a change of your own, is always the same shape:

1. Pick N seeds (6-10 is typical here).
2. Run both the "before" and "after" configuration on the *same* seeds (paired, not two
   independent samples).
3. Report success/fall rate and a relevant continuous metric (time-to-goal, RMSE, ...)
   across all N, not just the mean of one run.
4. If the two configurations' numbers are close, check *which specific seeds* flipped
   outcome, not just the aggregate rate -- a single lucky/unlucky seed can make a wash
   look like a win or vice versa.

A single demo GIF is for *showing* a behavior exists; it is never, on its own, evidence
that a change is actually an improvement.

## Where things live

```
wam/
├── wam/
│   ├── state.py          # 33-dim state / 3-dim action definitions
│   ├── model.py           # the MLP dynamics model
│   ├── pose.py             # dead-reckon (x, y, yaw) from the model's own predictions
│   ├── planner.py           # CEM shooting planner (+ optional obstacle cost)
│   └── global_planner.py     # A* over an occupancy grid, feeds the CEM a waypoint
├── collect_data.py    # generate training rollouts from the real simulator
├── train_world_model.py  # one-step + H-step rollout loss training
├── eval_rollout.py          # open-loop rollout vs. real simulator (the real correctness check)
├── navigate_wam.py            # drive the real robot to a goal via the CEM planner
├── navigate_wam_obstacles.py    # same, with lidar-built obstacle avoidance
├── eval_navigate.py / eval_obstacles.py / eval_terrain.py / eval_obstacles_terrain.py
│                                 # multi-seed comparisons backing every claim above
├── checkpoints/                   # trained model weights
└── data/                            # collected rollout datasets
```

For the detailed build history -- exact numbers, what didn't work and why, every honest
limitation -- read [README.md](README.md) next.
