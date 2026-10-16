"""Stage 9: tell the moving obstacle's lidar-sensed points apart from the static
wall/island points already in the occupancy grid, and track its velocity over successive
scans -- lets the planner predict WHERE the obstacle will be a few tenths of a second from
now instead of assuming it stays frozen at its last-sensed position for the whole rollout
horizon (Stage 8's own documented limitation).

Still obeys the sense-then-plan philosophy Stage 3/8 established: this tracker only ever
sees `OccupancyGrid.occupied_points()` (what lidar has actually sensed), never the
obstacle's ground-truth position or its analytic motion model (`MovingObstacle.position_at`
is never imported here). The one prior it DOES use is a STATIC-only reference occupancy
grid, built once at startup by scanning the environment's known wall/island geometry with
no obstacle present -- a reasonable "I was given a floor plan" assumption for a robot
operating in a fixed building, and not the same thing as knowing the obstacle's own
position or motion.
"""
from collections import deque

import numpy as np


class DynamicObstacleTracker:
    def __init__(self, static_points, match_radius: float = 0.2, max_speed: float = 0.8,
                 window: int = 10, min_samples: int = 8):
        """static_points: (N,2) occupied-cell centers from the STATIC-only reference grid.
        match_radius: a live occupied cell within this distance of a static point is
        treated as that same static feature (wall/island), not a dynamic obstacle -- should
        be a bit larger than the occupancy grid's own resolution to absorb cell-center
        rounding. max_speed: sanity clamp (m/s) on the estimated obstacle speed, guarding
        against noise producing a wild extrapolated velocity -- set close to the fastest
        speed an obstacle in this scenario could plausibly move at, not a loose bound.
        window/min_samples: velocity is estimated from the FIRST vs. LAST position in a
        rolling window of the last `window` scans, not consecutive-scan finite differencing
        -- differencing two nearby noisy samples amplifies that noise (the scan-to-scan
        centroid jitters with which part of the obstacle's surface lidar happened to see);
        a longer baseline averages it out. `min_samples` gates off any velocity estimate
        (falls back to a frozen/zero-velocity assumption) until the window has enough
        sightings to make that longer-baseline estimate meaningful."""
        self.static_points = np.asarray(static_points, dtype=np.float64).reshape(-1, 2)
        self.match_radius = match_radius
        self.max_speed = max_speed
        self.min_samples = min_samples
        self._history = deque(maxlen=window)
        self.velocity = np.zeros(2)

    def _dynamic_mask(self, occupied_points: np.ndarray) -> np.ndarray:
        if len(occupied_points) == 0:
            return np.zeros(0, dtype=bool)
        if len(self.static_points) == 0:
            return np.ones(len(occupied_points), dtype=bool)
        d = np.linalg.norm(
            occupied_points[:, None, :] - self.static_points[None, :, :], axis=2).min(axis=1)
        return d > self.match_radius

    def dynamic_points(self, occupied_points) -> np.ndarray:
        """Points NOT explained by the static reference map -- candidate sightings of the
        moving obstacle."""
        occupied_points = np.asarray(occupied_points, dtype=np.float64).reshape(-1, 2)
        return occupied_points[self._dynamic_mask(occupied_points)]

    def static_subset(self, occupied_points) -> np.ndarray:
        """The complement of `dynamic_points` -- what a planner's safety-net obstacle cost
        should keep treating as always-there and frozen (walls/island)."""
        occupied_points = np.asarray(occupied_points, dtype=np.float64).reshape(-1, 2)
        return occupied_points[~self._dynamic_mask(occupied_points)]

    def update(self, occupied_points, t_now: float):
        """Feed one lidar scan's occupied points (from the LIVE grid) at time t_now.
        Returns (pos, vel) estimate of the moving obstacle, or None if nothing dynamic has
        been sensed in this scan."""
        dyn = self.dynamic_points(occupied_points)
        if len(dyn) == 0:
            return None
        pos = dyn.mean(axis=0)
        self._history.append((t_now, pos))
        if len(self._history) >= self.min_samples:
            t0, pos0 = self._history[0]
            dt = t_now - t0
            if dt > 1e-3:
                sample_vel = (pos - pos0) / dt
                speed = np.linalg.norm(sample_vel)
                if speed > self.max_speed:
                    sample_vel = sample_vel * (self.max_speed / speed)
                self.velocity = sample_vel
        return pos, self.velocity.copy()


def build_static_reference_points(static_model, static_data, world, resolution: float = 0.1,
                                   vantage_spacing: float = 1.5, margin: float = 0.3,
                                   n_beams: int = 360, occ_threshold: float = 0.7):
    """Scan the environment's STATIC-only MuJoCo model (built from XML with no obstacle
    injected) from a grid of vantage points inside the free corridor, and return the
    resulting occupied-cell centers -- the one-time reference `DynamicObstacleTracker` needs
    to recognize "this is a wall, not the obstacle" from a single scan."""
    import mujoco as mj
    from slam.occupancy_grid import OccupancyGrid
    from slam.mj_lidar import mj_lidar_scan

    xmin, xmax, ymin, ymax = world.bounds
    xs = np.arange(xmin + margin, xmax - margin, vantage_spacing)
    ys = np.arange(ymin + margin, ymax - margin, vantage_spacing)
    static_grid = OccupancyGrid(world.bounds, resolution=resolution)
    for vx in xs:
        for vy in ys:
            if not world.is_free(float(vx), float(vy), margin=margin):
                continue
            ranges, hit, angles = mj_lidar_scan(
                static_model, static_data, (float(vx), float(vy)), 0.0, n_beams=n_beams)
            static_grid.update((float(vx), float(vy), 0.0), ranges, hit, angles)
    return static_grid.occupied_points(threshold=occ_threshold)
