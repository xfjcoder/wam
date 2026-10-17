"""Pixel-rendering mini-plan, step 3: a visual sanity check for wam/camera.py's onboard
depth_cam BEFORE any of it gets wired into collect_data.py -- the whole point is to catch a
bad mount angle (e.g. looking at the robot's own legs, or straight at the sky) immediately,
the same failure go1_rl_walk's own depth_cam comment documents having actually happened once
during ITS tuning.

Builds the real go2_mesh_for_mpc.xml with slam_sim's loop_corridor walls injected (so there's
actual structure -- a wall/island -- in view, not just an empty floor) plus the onboard
camera from wam/camera.py, runs a few seconds of scripted (not planned -- no world model/CEM
needed for this check) forward-then-turn-toward-the-wall motion, and dumps a handful of
paired RGB/depth PNGs to media/camera_sanity/ to be looked at directly.

No encoder, no data collection, no model/planner changes -- purely "does the camera see
something sensible" infrastructure, scoped deliberately narrow per the project's mini-plan.

A fall during the final turn phase is expected, not a bug: that phase rotates in place while
the trunk is still pressed against the wall from the approach phase, which this sim has no
qualms about reporting as a real contact-induced QP failure -- it happens AFTER the frames
this script cares about have already been saved.

Usage (needs go1_rl_walk/mpc's own venv, and slam_sim importable as a sibling repo):
    /home/cstu/robo/go1_rl_walk/mpc/venv/bin/python3 check_onboard_camera.py
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import tempfile
from pathlib import Path

import numpy as np

GO1_RL_WALK = Path(__file__).resolve().parent.parent / "go1_rl_walk"
SLAM_SIM = Path(__file__).resolve().parent.parent / "slam_sim"
sys.path.insert(0, str(GO1_RL_WALK / "mpc"))
sys.path.insert(0, str(GO1_RL_WALK))
sys.path.insert(0, str(SLAM_SIM))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mpc_residual_env import MPCResidualEnv, DEFAULT_XML, _fix_meshdir_absolute  # noqa: E402

from demo_stage6_go2_world import inject_walls                 # noqa: E402
from slam.world import World                                    # noqa: E402

from wam.camera import OnboardCamera, inject_onboard_camera      # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "media" / "camera_sanity"


def build_camera_xml(walls, tmp_dir: Path) -> Path:
    xml_text = Path(DEFAULT_XML).read_text()
    xml_text = _fix_meshdir_absolute(xml_text, Path(DEFAULT_XML).resolve().parent)
    xml_text = inject_walls(xml_text, walls)
    xml_text = inject_onboard_camera(xml_text)
    out_path = tmp_dir / "go2_camera_sanity.xml"
    out_path.write_text(xml_text)
    return out_path


def main():
    from PIL import Image

    world = World.loop_corridor()
    tmp_dir = Path(tempfile.mkdtemp(prefix="wam_camera_sanity_"))
    xml_path = build_camera_xml(world.walls, tmp_dir)

    env = MPCResidualEnv(
        xml_path=str(xml_path), episode_seconds=12.0,
        speed_range=(0.0, 0.0), lateral_range=(0.0, 0.0), yaw_rate_range=(0.0, 0.0),
        seed=0,
    )
    obs, _ = env.reset(seed=0)
    cam = OnboardCamera(env.mujoco_go2.model, resolution=128)

    # Start just inside the loop_corridor world, headed straight at the island wall so the
    # camera has something unambiguous (a wall, not just open floor) to see within a few
    # seconds -- World.loop_corridor()'s island sits in the middle of its 12x8m rectangle.
    start_x, start_y = 1.5, 4.0
    env.mujoco_go2.data.qpos[0] = start_x
    env.mujoco_go2.data.qpos[1] = start_y

    # Four scripted phases, no planner: stand (baseline frame), walk straight toward the
    # wall long enough to physically reach/stall against it (real contact, not a teleport --
    # a teleported pose was tried and discarded here, see wam/camera.py's module docstring
    # for why), hold there briefly for the closest-range frames, then turn in place -- enough
    # variety to confirm the mount reacts sensibly to both translation and rotation, without
    # needing the learned model or CEM at all.
    phases = [("stand", 0.0, 0.0, 0.0, 2.0),
              ("approach", 0.35, 0.0, 0.0, 7.0),
              ("hold", 0.35, 0.0, 0.0, 2.0),
              ("turn", 0.0, 0.0, 0.6, 3.0)]

    frame_idx = 0
    cur_fwd = cur_lat = cur_yaw = 0.0
    max_cmd_delta = 1.5 / 200.0
    max_yaw_delta = 3.0 / 200.0
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, tgt_fwd, tgt_lat, tgt_yaw, seconds in phases:
        n_ticks = int(seconds * 200)
        save_every = max(1, n_ticks // 3)
        for tick in range(n_ticks):
            cur_fwd += np.clip(tgt_fwd - cur_fwd, -max_cmd_delta, max_cmd_delta)
            cur_lat += np.clip(tgt_lat - cur_lat, -max_cmd_delta, max_cmd_delta)
            cur_yaw += np.clip(tgt_yaw - cur_yaw, -max_yaw_delta, max_yaw_delta)
            env.set_command(cur_fwd, cur_lat, cur_yaw)
            obs, _, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
            if terminated or truncated:
                print(f"FELL/QP-failed during phase '{name}' at tick {tick} -- stopping early.")
                break
            if tick % save_every == 0:
                x, y = float(info["base_pos"][0]), float(info["base_pos"][1])
                rgb = cam.rgb(env.mujoco_go2.data)
                depth = cam.depth(env.mujoco_go2.data)
                # Clipped to 1.2m (not this camera's full range) purely for human-visible
                # contrast in the saved PNG -- this mount's useful obstacle-sensing range is
                # well under 1m anyway (see wam/camera.py's module docstring), and a 3m clip
                # compressed everything into a nearly-flat dark image.
                depth_vis = np.clip(depth, 0.0, 1.2) / 1.2
                depth_u8 = (depth_vis * 255).astype(np.uint8)
                Image.fromarray(rgb).save(OUT_DIR / f"{frame_idx:02d}_{name}_rgb.png")
                Image.fromarray(depth_u8).save(OUT_DIR / f"{frame_idx:02d}_{name}_depth.png")
                print(f"  [{frame_idx:02d}] phase={name:8s} t={tick/200.0:4.1f}s "
                      f"pos=({x:5.2f},{y:5.2f}) depth range=[{depth.min():.2f},{depth.max():.2f}]m")
                frame_idx += 1
        else:
            continue
        break

    cam.close()
    print(f"Saved {frame_idx} paired RGB/depth frames to {OUT_DIR}/")


if __name__ == "__main__":
    main()
