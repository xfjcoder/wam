"""Pixel-rendering mini-plan, steps 1-2: an onboard, body-fixed camera on the exact MuJoCo
model `MPCResidualEnv` drives (go1_rl_walk/mpc/go2_mesh_for_mpc.xml), reusing go1_rl_walk's
own already-tuned `depth_cam` placement (assets/go2_mesh_camera.xml) verbatim. That mount
(pos="0.32 0 -0.02", aimed ~26deg below horizontal) was already empirically fixed once over
there -- an earlier attempt (x=0.2, steeper) looked almost entirely at the robot's own front
legs/trunk underside, confirmed by rendering an RGB test image -- so there is no reason to
re-derive it here.

Confirmed (not just reused on faith) that this exact mount ALSO sees chest-height obstacles,
not just go1_rl_walk's own near-field ground/terrain: driving straight at slam_sim's
loop_corridor island wall (0.6m tall) and sampling this camera at several distances shows a
clean, monotonically shrinking far-depth as the wall fills more of the frame (2.53m->1.63m
->0.91m->0.46m at trunk_x=1.0/2.0/2.8/3.3) -- exactly the obstacle-proximity signal a
pixel-based planner would need, with no extra tuning required. (It fails only in the last few
cm of actual wall contact, where the camera's own forward offset puts it past the wall's thin
0.05m slab -- irrelevant for any real planning use, which reacts long before contact.)

Getting to that confirmation took a real wrong turn worth recording: the SAME wall was
initially invisible at every distance and every pitch angle tried (5/10/15/20/26 degrees),
which looked like a geometry problem with the mount. The actual cause was nothing to do with
the mount at all -- `mj.Renderer.update_scene()`'s default `MjvOption` only shows geom groups
0-2 (confirmed directly: `mj.MjvOption().geomgroup == [1,1,1,0,0,0]`), and slam_sim's own
`inject_walls`/this project's `inject_moving_obstacle` both put their geoms in group 4 (a
"collision/sensing only" convention lidar/mj_ray never cared about, since THEIR group
filtering is passed explicitly per call). A RASTERIZED camera silently respects that default,
so every obstacle was invisible to RGB/depth rendering despite being squarely in frame. Every
render call here now passes an explicit all-groups-visible `MjvOption` to avoid silently
reproducing that for any future obstacle/geom added in some other group.

`inject_onboard_camera` is the same plain string-splice XML-injection technique
`wam/moving_obstacle.py`'s `inject_moving_obstacle` and slam_sim's own `inject_walls` already
use, kept HERE (not added to go1_rl_walk/mpc's own XML) for the same "add zero code to the
sibling repo" convention `navigate_wam_obstacles.py` already follows.

`OnboardCamera` wraps `mujoco.Renderer` for BOTH an RGB frame and a depth frame from the same
depth_cam viewpoint: RGB is for a human to actually look at and sanity-check the mount (a
mount bug is obvious in an RGB image -- "that's the robot's own leg" -- but not obvious
staring at a raw depth array), depth is what go1_env.py's own RL camera observation already
uses and is the more likely first encoder input if a later stage trains on this.

Known quirk, worth recording rather than re-discovering later: a raw `mj_ray` cast from this
camera's own `cam_xpos` immediately self-intersects the trunk's own chassis mesh at ~0.01m in
EVERY direction tried, regardless of trunk pose -- the camera's mount point sits right
at/inside that mesh's convex-hull approximation. `mujoco.Renderer`'s rasterized output is
unaffected (its near/far-plane + backface handling evidently discards the enclosing geometry)
-- harmless for anything going through `Renderer` (this module's whole purpose), but a future
stage must NOT reuse this exact camera position/offset for a raw `mj_ray`-based probe (e.g. a
lidar-style sensor rigidly mounted at the same point) without first re-checking for this.
"""
import mujoco as mj
import numpy as np

_DEPTH_CAM_SNIPPET = '<camera name="depth_cam" pos="0.32 0 -0.02" xyaxes="0 -1 0 0.436 0 0.9"/>'
_ANCHOR = '<site name="imu" pos="0 0 0" size="0.01" />'


def inject_onboard_camera(xml_text: str) -> str:
    assert xml_text.count(_ANCHOR) == 1, "expected exactly one imu site to anchor the camera after"
    return xml_text.replace(_ANCHOR, f"{_ANCHOR}\n      {_DEPTH_CAM_SNIPPET}")


class OnboardCamera:
    """`mj.Renderer.update_scene()`'s default `MjvOption` only shows geom groups 0-2 --
    confirmed directly (`mj.MjvOption().geomgroup == [1,1,1,0,0,0]`), not documented anywhere
    obvious. `slam_sim`'s own `inject_walls`/this project's `inject_moving_obstacle` both put
    their geoms in group 4 (a convention meant for "collision/sensing only, decluttered out of
    casual on-screen viewers"), which lidar/mj_ray never cared about (group filtering there is
    passed explicitly per call) but a RASTERIZED camera silently respects by default -- without
    this, walls/obstacles are invisible to RGB/depth rendering even though they're squarely in
    frame, which is exactly the dead end this module's own tuning hit before finding it. Every
    render call here passes an explicit all-groups-visible option to avoid silently reproducing
    that."""
    _ALL_GROUPS = mj.MjvOption()
    _ALL_GROUPS.geomgroup[:] = 1

    def __init__(self, model, resolution: int = 64):
        self.resolution = resolution
        self._renderer = mj.Renderer(model, height=resolution, width=resolution)

    def rgb(self, data) -> np.ndarray:
        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(data, camera="depth_cam", scene_option=self._ALL_GROUPS)
        return self._renderer.render().copy()

    def depth(self, data) -> np.ndarray:
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(data, camera="depth_cam", scene_option=self._ALL_GROUPS)
        return self._renderer.render().copy()

    def close(self):
        self._renderer.close()
