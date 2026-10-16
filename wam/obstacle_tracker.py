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

Stage 10 generalizes this from "one dynamic blob" to however many are actually out there:
dynamic points are first CLUSTERED (simple radius-based union-find -- obstacles are
spatially separated, no fancier clustering needed), then each scan's clusters are matched
to the previous scan's tracked clusters by nearest-centroid gating (greedy, not Hungarian --
a handful of slow-moving obstacles a replan-cadence apart never need the optimal assignment
a crowded scene would). Each track keeps its own position/velocity history independently, so
two obstacles moving in different directions don't get blended into one meaningless average
the way a single global centroid would.
"""
from collections import deque

import numpy as np


def _cluster_points(points: np.ndarray, cluster_radius: float) -> list:
    """Greedy union-find clustering: group points mutually within `cluster_radius` of some
    other point in the same group. Returns a list of per-cluster centroids."""
    n = len(points)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) <= cluster_radius:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [points[idx].mean(axis=0) for idx in groups.values()]


class DynamicObstacleTracker:
    def __init__(self, static_points, match_radius: float = 0.2, cluster_radius: float = 0.9,
                 track_gate: float = 1.0, max_speed: float = 0.8, window: int = 10,
                 min_samples: int = 8, max_missed: int = 6):
        """static_points: (N,2) occupied-cell centers from the STATIC-only reference grid.
        match_radius: a live occupied cell within this distance of a static point is
        treated as that same static feature (wall/island), not a dynamic obstacle -- should
        be a bit larger than the occupancy grid's own resolution to absorb cell-center
        rounding. cluster_radius: dynamic points within this of each other are grouped as
        sightings of the SAME obstacle -- should be a bit larger than one obstacle's own
        footprint, and well under the separation between two distinct obstacles. track_gate:
        a new scan's cluster centroid is matched to an existing track only if within this
        distance of that track's last known position (greedy nearest-centroid, not an
        optimal assignment -- fine for a handful of obstacles replanned every fraction of a
        second, where consecutive sightings of the same obstacle never jump far). max_speed:
        sanity clamp (m/s) on any one track's estimated speed. window/min_samples: velocity
        is estimated from the FIRST vs. LAST position in a rolling window of the last
        `window` sightings of that SAME track, not consecutive-scan finite differencing --
        differencing two nearby noisy samples amplifies that noise; a longer baseline
        averages it out. max_missed: a track not matched for this many consecutive scans
        (obstacle temporarily occluded or out of range) is dropped rather than kept forever
        on a stale position."""
        self.static_points = np.asarray(static_points, dtype=np.float64).reshape(-1, 2)
        self.match_radius = match_radius
        self.cluster_radius = cluster_radius
        self.track_gate = track_gate
        self.max_speed = max_speed
        self.window = window
        self.min_samples = min_samples
        self.max_missed = max_missed
        self._tracks = []  # each: {"history": deque[(t, pos)], "velocity": (2,), "missed": int}

    def _dynamic_mask(self, occupied_points: np.ndarray) -> np.ndarray:
        if len(occupied_points) == 0:
            return np.zeros(0, dtype=bool)
        if len(self.static_points) == 0:
            return np.ones(len(occupied_points), dtype=bool)
        d = np.linalg.norm(
            occupied_points[:, None, :] - self.static_points[None, :, :], axis=2).min(axis=1)
        return d > self.match_radius

    def dynamic_points(self, occupied_points) -> np.ndarray:
        """Points NOT explained by the static reference map -- candidate sightings of
        moving obstacles (of however many there are)."""
        occupied_points = np.asarray(occupied_points, dtype=np.float64).reshape(-1, 2)
        return occupied_points[self._dynamic_mask(occupied_points)]

    def static_subset(self, occupied_points) -> np.ndarray:
        """The complement of `dynamic_points` -- what a planner's safety-net obstacle cost
        should keep treating as always-there and frozen (walls/island)."""
        occupied_points = np.asarray(occupied_points, dtype=np.float64).reshape(-1, 2)
        return occupied_points[~self._dynamic_mask(occupied_points)]

    def update(self, occupied_points, t_now: float) -> list:
        """Feed one lidar scan's occupied points (from the LIVE grid) at time t_now.
        Returns a list of (pos, vel) estimates, one per currently-tracked obstacle (possibly
        empty, possibly more than one)."""
        dyn = self.dynamic_points(occupied_points)
        centroids = _cluster_points(dyn, self.cluster_radius)

        unmatched_tracks = set(range(len(self._tracks)))
        unmatched_centroids = set(range(len(centroids)))
        pairs = []
        for ti in range(len(self._tracks)):
            last_pos = self._tracks[ti]["history"][-1][1]
            best_cj, best_d = None, self.track_gate
            for cj in unmatched_centroids:
                d = float(np.linalg.norm(centroids[cj] - last_pos))
                if d < best_d:
                    best_d, best_cj = d, cj
            if best_cj is not None:
                pairs.append((ti, best_cj))
                unmatched_tracks.discard(ti)
                unmatched_centroids.discard(best_cj)

        for ti, cj in pairs:
            track = self._tracks[ti]
            track["history"].append((t_now, centroids[cj]))
            track["missed"] = 0
            if len(track["history"]) >= self.min_samples:
                t0, pos0 = track["history"][0]
                dt = t_now - t0
                if dt > 1e-3:
                    sample_vel = (centroids[cj] - pos0) / dt
                    speed = np.linalg.norm(sample_vel)
                    if speed > self.max_speed:
                        sample_vel = sample_vel * (self.max_speed / speed)
                    track["velocity"] = sample_vel

        for ti in unmatched_tracks:
            self._tracks[ti]["missed"] += 1
        self._tracks = [t for t in self._tracks if t["missed"] <= self.max_missed]

        for cj in unmatched_centroids:
            self._tracks.append({
                "history": deque([(t_now, centroids[cj])], maxlen=self.window),
                "velocity": np.zeros(2), "missed": 0})

        return [(t["history"][-1][1], t["velocity"].copy()) for t in self._tracks]


def build_static_reference_points(static_model, static_data, world, resolution: float = 0.1,
                                   vantage_spacing: float = 0.6, margin: float = 0.25,
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
