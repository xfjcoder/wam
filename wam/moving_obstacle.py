"""Stage 8: a single moving (mocap-driven) obstacle, for testing the planner against
something that isn't where the last lidar scan saw it.

A plain `<geom>` under `<worldbody>` -- what every static wall/island geom in
`demo_stage6_go2_world.py`'s `inject_walls` already is -- is permanently welded to the world
at MuJoCo compile time; nothing can move it afterward. A MuJoCo mocap body (no joint, no
inertia -- its pose is written directly into `data.mocap_pos`/`data.mocap_quat` each step,
not simulated) is the only way to relocate a geom after the model is compiled. Confirmed
directly (not assumed) that `slam.mj_lidar.mj_lidar_scan`'s `mj_ray` call reads live geom
transforms every call, so the lidar sees a moved mocap obstacle with zero changes needed
there, and that `slam.occupancy_grid.OccupancyGrid.update` already does log-odds free-space
clearing along each beam -- a cell that goes stale (obstacle moved away) decays back toward
"free" over a few scans rather than leaving a permanent ghost trail.

Injected via the same plain string-splice technique `inject_walls`/`_inject_terrain_xml`
already use, kept HERE rather than added to `slam_sim` -- same "add zero code to the sibling
repo" convention `navigate_wam_obstacles.py` already follows for everything else it reuses.
"""
import numpy as np


class MovingObstacle:
    def __init__(self, name: str, center, axis: str, amplitude: float, period: float,
                 phase: float = 0.0, size: float = 0.3, height: float = 0.6):
        """Oscillates sinusoidally along ONE world axis ('x' or 'y'), centered at `center`
        (cx, cy), +/- amplitude, with the given period (s) and phase offset (rad). `phase`
        lets a multi-seed evaluation vary WHEN in its sweep the obstacle crosses the robot's
        path, instead of every seed seeing an identical schedule relative to episode start."""
        assert axis in ("x", "y")
        self.name = name
        self.center = (float(center[0]), float(center[1]))
        self.axis = axis
        self.amplitude = float(amplitude)
        self.period = float(period)
        self.phase = float(phase)
        self.size = float(size)
        self.height = float(height)

    def position_at(self, t: float):
        offset = self.amplitude * np.sin(2 * np.pi * t / self.period + self.phase)
        x, y = self.center
        if self.axis == "x":
            x = x + offset
        else:
            y = y + offset
        return float(x), float(y)

    def xml_snippet(self) -> str:
        x0, y0 = self.position_at(0.0)
        return (f'<body name="{self.name}" mocap="true" pos="{x0:.4f} {y0:.4f} 0">'
                f'<geom name="{self.name}_geom" type="box" group="4" contype="1" '
                f'conaffinity="1" rgba="0.85 0.15 0.15 1" '
                f'size="{self.size:.4f} {self.size:.4f} {self.height / 2:.4f}" '
                f'pos="0 0 {self.height / 2:.4f}"/></body>')


def inject_moving_obstacle(xml_text: str, obstacle: MovingObstacle) -> str:
    assert xml_text.count("</worldbody>") == 1, "expected exactly one </worldbody>"
    return xml_text.replace("</worldbody>", f"    {obstacle.xml_snippet()}\n  </worldbody>")
