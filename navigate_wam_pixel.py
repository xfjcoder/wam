"""Pixel-rendering mini-plan, step 17: wires step 15's latent cost head into an actual CEM
planner, finally closing the "never wired in" gap every earlier step in this mini-plan
flagged. Unlike navigate_wam_obstacles.py (Stage 3), this script builds NO occupancy map and
runs NO lidar at all -- obstacle-avoidance here comes entirely from the onboard camera's depth
frame, encoded (wam/pixel_encoder.py), rolled forward per CEM candidate through the pixel
dynamics model (wam/pixel_dynamics.py), and scored by the frozen LatentCostHead (wam/
pixel_cost.py) -- step 15's whole point was making exactly this loop (decode never called)
fast enough for CEM, and wam/planner.py's CEMPlanner step 17 addition is what actually calls
it now.

Deliberately split responsibilities, not a wholesale replacement of the state-based planner:
the STATE-based WorldModel (checkpoints/world_model.pt) still owns pose dead-reckoning,
goal-distance, heading, and stability cost, exactly as it does in every earlier navigate_wam_*
script -- a bare predicted min-depth number has no (x, y) to integrate a goal distance from,
so asking the pixel model to do that job too would be forcing it to do something it fundamentally
can't observe. The pixel model's ONLY job here is "how close is the nearest thing in view",
read straight from the same onboard camera (wam/camera.py) steps 1-3 built and verified,
instead of navigate_wam_obstacles.py's lidar-built occupancy grid.

This is a REACTIVE, local-only planner (no A* global waypoint routing the way
navigate_wam_obstacles.py's Stage 3b addition has) -- an object only affects planning once
it's actually in the camera's forward-facing field of view, and nothing is remembered once it
leaves frame (no mapping). Expect this to behave like Stage 3's ORIGINAL un-augmented local
CEM (corner-stall-prone on a goal that requires routing around something), not Stage 3b's
A*-assisted version -- the honest comparison point for this script is Stage 3's local-only
baseline, not its global-planner-equipped descendant.

Usage (needs go1_rl_walk/mpc's own venv for mujoco/torch, and slam_sim importable as a
sibling repo):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 navigate_wam_pixel.py \
        --goal-x 10.5 --goal-y 6.5 \
        --pixel-checkpoint checkpoints/pixel_world_model_wall_explore_depth.pt \
        --cost-checkpoint checkpoints/pixel_cost_head_explore.pt \
        --record-robot media/navigate_wam_pixel_robot.gif
"""
import argparse
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import tempfile
from pathlib import Path

import mujoco as mj
import numpy as np
import torch

GO1_RL_WALK = Path(__file__).resolve().parent.parent / "go1_rl_walk"
SLAM_SIM = Path(__file__).resolve().parent.parent / "slam_sim"
sys.path.insert(0, str(GO1_RL_WALK / "mpc"))
sys.path.insert(0, str(GO1_RL_WALK))
sys.path.insert(0, str(SLAM_SIM))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mpc_residual_env import MPCResidualEnv, DEFAULT_XML, _fix_meshdir_absolute  # noqa: E402
from demo_stage6_go2_world import inject_walls                                   # noqa: E402
from slam.world import World                                                     # noqa: E402
from slam.viz import render_frame, save_gif                                      # noqa: E402

from wam.model import WorldModel                                   # noqa: E402
from wam.planner import CEMPlanner                                 # noqa: E402
from wam.state import STATE_DIM, ACTION_DIM, extract_state         # noqa: E402
from wam.camera import OnboardCamera, inject_onboard_camera        # noqa: E402
from wam.pixel_encoder import ConvAutoencoder, depth_to_tensor_01  # noqa: E402
from wam.pixel_dynamics import LatentDynamics                      # noqa: E402
from wam.pixel_cost import LatentCostHead                          # noqa: E402


def build_xml(tmp_dir: Path, world: World) -> Path:
    xml_text = Path(DEFAULT_XML).read_text()
    xml_text = _fix_meshdir_absolute(xml_text, Path(DEFAULT_XML).resolve().parent)
    xml_text = inject_walls(xml_text, world.walls)
    xml_text = inject_onboard_camera(xml_text)
    out_path = tmp_dir / "go2_navigate_pixel.xml"
    out_path.write_text(xml_text)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-x", type=float, default=1.5)
    parser.add_argument("--start-y", type=float, default=1.5)
    parser.add_argument("--goal-x", type=float, default=10.5)
    parser.add_argument("--goal-y", type=float, default=6.5)
    parser.add_argument("--tolerance", type=float, default=0.3)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/world_model.pt")
    parser.add_argument("--pixel-checkpoint", type=str,
                         default="checkpoints/pixel_world_model_wall_explore_depth.pt")
    parser.add_argument("--cost-checkpoint", type=str,
                         default="checkpoints/pixel_cost_head_explore.pt")
    parser.add_argument("--depth-clip", type=float, default=3.0)
    parser.add_argument("--horizon-steps", type=int, default=10)
    parser.add_argument("--pixel-horizon-steps", type=int, default=3,
                         help="cap the camera-cost term to this many of horizon-steps's steps "
                              "(default 3) -- train_pixel_cost.py's own rollout check found "
                              "the cost head's predicted-latent correlation collapsing fast on "
                              "turning-inclusive data (0.80/0.53/0.35 at steps 1/2/3), so "
                              "scoring the full 10-step horizon with it would inject noisy, "
                              "eventually anti-correlated cost into the later steps.")
    parser.add_argument("--replan-every", type=int, default=5, help="macro-steps between replans")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-elites", type=int, default=20)
    parser.add_argument("--n-iters", type=int, default=3)
    parser.add_argument("--w-pixel-obstacle", type=float, default=40.0)
    parser.add_argument("--w-pixel-collision", type=float, default=200.0)
    parser.add_argument("--pixel-safety-margin", type=float, default=0.6)
    parser.add_argument("--pixel-collision-margin", type=float, default=0.35)
    parser.add_argument("--disable-pixel-cost", action="store_true",
                         help="plan with goal-only cost (ignores the camera entirely) -- for "
                              "an intentional before/after comparison against the camera-aware "
                              "planner on the SAME island world.")
    parser.add_argument("--max-accel", type=float, default=1.5)
    parser.add_argument("--max-yaw-accel", type=float, default=3.0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--record-robot", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # CEMPlanner's own candidate sampling (torch.randn) was never seeded anywhere in this
    # project -- confirmed directly: re-running an identical --seed produced a DIFFERENT
    # final distance. --seed only controlled env/numpy randomness. Without this, a 3-seed
    # comparison partly measures CEM sampling noise, not the configuration being tested.
    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WorldModel(STATE_DIM, ACTION_DIM, hidden=ckpt["hidden"], n_hidden=ckpt["n_hidden"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pixel_ckpt = torch.load(args.pixel_checkpoint, map_location="cpu", weights_only=False)
    assert pixel_ckpt.get("modality", "rgb") == "depth", (
        "navigate_wam_pixel.py's cost head targets metric min-depth -- needs a depth-modality "
        f"pixel checkpoint (got modality={pixel_ckpt.get('modality')!r})")
    encoder = ConvAutoencoder(latent_dim=pixel_ckpt["latent_dim"], resolution=pixel_ckpt["resolution"],
                               channels=pixel_ckpt["channels"])
    encoder.load_state_dict(pixel_ckpt["encoder_state"])
    encoder.eval()
    pixel_dynamics = LatentDynamics(latent_dim=pixel_ckpt["latent_dim"],
                                     action_dim=pixel_ckpt["action_dim"])
    pixel_dynamics.load_state_dict(pixel_ckpt["dynamics_state"])
    pixel_dynamics.eval()

    cost_ckpt = torch.load(args.cost_checkpoint, map_location="cpu", weights_only=False)
    cost_head = LatentCostHead(latent_dim=cost_ckpt["latent_dim"])
    cost_head.load_state_dict(cost_ckpt["cost_head_state"])
    cost_head.eval()

    world = World.loop_corridor()
    tmp_dir = Path(tempfile.mkdtemp(prefix="wam_navigate_pixel_"))
    xml_path = build_xml(tmp_dir, world)

    env = MPCResidualEnv(
        xml_path=str(xml_path), episode_seconds=args.seconds,
        speed_range=(0.0, 0.0), lateral_range=(0.0, 0.0), yaw_rate_range=(0.0, 0.0),
        seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    env.mujoco_go2.data.qpos[0:2] = [args.start_x, args.start_y]
    mj.mj_forward(env.mujoco_go2.model, env.mujoco_go2.data)

    cam = OnboardCamera(env.mujoco_go2.model, resolution=pixel_ckpt["resolution"])

    planner = CEMPlanner(
        model, speed_range=(-0.4, 0.8), lateral_range=(-0.3, 0.3), yaw_rate_range=(-1.0, 1.0),
        horizon_steps=args.horizon_steps, macro_dt=10 / 200.0,
        n_samples=args.n_samples, n_elites=args.n_elites, n_iters=args.n_iters,
        w_obstacle=0.0, w_collision=0.0,
        pixel_dynamics=None if args.disable_pixel_cost else pixel_dynamics,
        pixel_cost_head=None if args.disable_pixel_cost else cost_head,
        w_pixel_obstacle=args.w_pixel_obstacle, w_pixel_collision=args.w_pixel_collision,
        pixel_safety_margin=args.pixel_safety_margin,
        pixel_collision_margin=args.pixel_collision_margin,
        pixel_horizon_steps=args.pixel_horizon_steps)

    goal = np.array([args.goal_x, args.goal_y])
    max_cmd_delta = args.max_accel / 200.0
    max_yaw_delta = args.max_yaw_accel / 200.0
    max_steps = int(args.seconds * 200)
    x, y, yaw = args.start_x, args.start_y, 0.0
    cur_fwd, cur_lat, cur_yaw = 0.0, 0.0, 0.0
    target_action = np.zeros(3, dtype=np.float32)
    true_trail = []
    reached, fell = False, False
    t_reached = None

    def _add_marker(scene, pos_xy, rgba, size=0.15):
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(
            g, mj.mjtGeom.mjGEOM_SPHERE, np.array([size, 0, 0]),
            np.array([pos_xy[0], pos_xy[1], 0.05]), np.eye(3).flatten(),
            np.array(rgba, dtype=np.float32))
        scene.ngeom += 1

    robot_renderer, robot_cam = None, None
    if args.record_robot:
        robot_cam = mj.MjvCamera()
        robot_cam.type = mj.mjtCamera.mjCAMERA_FREE
        robot_cam.lookat[:] = [world.bounds[1] / 2.0, world.bounds[3] / 2.0, 0.0]
        robot_cam.distance = 17.0
        robot_cam.elevation = -60
        robot_cam.azimuth = 90
        robot_renderer = mj.Renderer(env.mujoco_go2.model, height=480, width=640)
        START_RGBA = (0.1, 0.9, 0.1, 1.0)
        GOAL_RGBA = (0.9, 0.1, 0.1, 1.0)
    robot_frames = []

    for step in range(1, max_steps + 1):
        dist = float(np.hypot(goal[0] - x, goal[1] - y))
        if dist <= args.tolerance:
            if not reached:
                reached, t_reached = True, step / 200.0

        if reached:
            target_action[:] = 0.0
        elif (step - 1) % (args.replan_every * 10) == 0:
            info_for_state = {"base_height": float(env.mujoco_go2.data.qpos[2])}
            state = extract_state(obs, info_for_state)

            pixel_latent0 = None
            if not args.disable_pixel_cost:
                depth_frame = cam.depth(env.mujoco_go2.data)
                depth_t = depth_to_tensor_01(depth_frame[None, ...], clip=args.depth_clip)
                with torch.no_grad():
                    pixel_latent0 = encoder.encode(depth_t)[0]

            result = planner.plan(state, (x, y, yaw), tuple(goal), warm_start=target_action,
                                   pixel_latent0=pixel_latent0)
            target_action = result.action.numpy()
            if args.verbose:
                print(f"  t={step/200.0:5.1f}s pos=({x:+.2f},{y:+.2f}) yaw={yaw:+.2f} "
                      f"dist={dist:.2f} action={target_action} cost={result.cost:.2f} "
                      f"pixel_cost={result.pixel_cost:.2f}")

        cur_fwd += np.clip(target_action[0] - cur_fwd, -max_cmd_delta, max_cmd_delta)
        cur_lat += np.clip(target_action[1] - cur_lat, -max_cmd_delta, max_cmd_delta)
        cur_yaw += np.clip(target_action[2] - cur_yaw, -max_yaw_delta, max_yaw_delta)
        env.set_command(cur_fwd, cur_lat, cur_yaw)

        obs, _, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
        x, y = float(info["base_pos"][0]), float(info["base_pos"][1])
        yaw = float(info["yaw"])
        true_trail.append([x, y])

        if args.record_robot and step % 16 == 0:
            robot_renderer.update_scene(env.mujoco_go2.data, camera=robot_cam)
            _add_marker(robot_renderer.scene, (args.start_x, args.start_y), START_RGBA)
            _add_marker(robot_renderer.scene, goal, GOAL_RGBA)
            robot_frames.append(robot_renderer.render().copy())

        if terminated or truncated:
            fell = terminated
            break
        if reached and step > (t_reached * 200 + 200):
            break

    cam.close()
    final_dist = float(np.hypot(goal[0] - x, goal[1] - y))
    if fell:
        print(f"FELL/QP-FAILED before reaching the goal (final distance {final_dist:.2f} m).")
    elif reached:
        print(f"Reached goal ({goal[0]:.2f}, {goal[1]:.2f}) in {t_reached:.1f}s, "
              f"final distance {final_dist:.2f} m.")
    else:
        print(f"Did NOT reach the goal within {args.seconds:.0f}s "
              f"(final distance {final_dist:.2f} m).")

    if robot_renderer is not None:
        robot_renderer.close()
    if args.record_robot and robot_frames:
        Path(args.record_robot).parent.mkdir(parents=True, exist_ok=True)
        save_gif(robot_frames, args.record_robot, fps=200.0 / 16)
        print(f"Saved {len(robot_frames)} frames to {args.record_robot}")


if __name__ == "__main__":
    main()
