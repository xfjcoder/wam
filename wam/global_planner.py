"""Stage 3b: a geometric global planner (A*) over the lidar-built occupancy grid, producing
a sparse sequence of waypoints for the CEM planner to chase.

This fixes Stage 3's documented corner-stall limitation (see README): CEM's cost is scored
over a short (10-step, ~0.5s) horizon using dead-reckoned pose, so near a sharp convex corner
it can't "see" that a clear path exists just past the bend -- every sampled action looks
either blocked or barely-progressing, and it stalls. A* instead reasons about the WHOLE known
map at once and returns a path that actually reaches the goal; the CEM then only ever has to
solve the much easier local problem of reaching the next nearby waypoint, never the whole
routing-around-an-obstacle problem itself.

Recomputed from scratch every replan (receding-horizon, no waypoint-index state to track):
cheap on a map this size (~10k cells), and naturally adapts as lidar reveals more of the map.

Same fundamental limitation as the rest of Stage 3: this only sees what's already been
scanned, so it can fail to find a path that actually exists if the map doesn't yet show it.
"""
import heapq

import numpy as np
from scipy.ndimage import binary_dilation

from slam.occupancy_grid import _bresenham


def _inflate(occ: np.ndarray, radius_cells: int) -> np.ndarray:
    if radius_cells <= 0:
        return occ
    structure = np.ones((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=bool)
    return binary_dilation(occ, structure=structure)


_NEIGHBORS = [(-1, -1, 2 ** 0.5), (-1, 0, 1.0), (-1, 1, 2 ** 0.5), (0, -1, 1.0),
              (0, 1, 1.0), (1, -1, 2 ** 0.5), (1, 0, 1.0), (1, 1, 2 ** 0.5)]


def _astar(occ: np.ndarray, start_cell, goal_cell):
    ny, nx = occ.shape
    sx, sy = start_cell
    gx, gy = goal_cell
    if not (0 <= sx < nx and 0 <= sy < ny and 0 <= gx < nx and 0 <= gy < ny):
        return None
    if occ[sy, sx] or occ[gy, gx]:
        return None

    def h(cx, cy):
        return float(np.hypot(gx - cx, gy - cy))

    open_heap = [(h(sx, sy), 0.0, (sx, sy))]
    came_from = {}
    g_score = {(sx, sy): 0.0}
    visited = set()
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == (gx, gy):
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path
        cx, cy = cur
        for dx, dy, cost in _NEIGHBORS:
            ncx, ncy = cx + dx, cy + dy
            if not (0 <= ncx < nx and 0 <= ncy < ny) or occ[ncy, ncx]:
                continue
            ng = g + cost
            if ng < g_score.get((ncx, ncy), float("inf")):
                g_score[(ncx, ncy)] = ng
                came_from[(ncx, ncy)] = cur
                heapq.heappush(open_heap, (ng + h(ncx, ncy), ng, (ncx, ncy)))
    return None


def _line_of_sight(occ: np.ndarray, a_cell, b_cell) -> bool:
    ny, nx = occ.shape
    for cx, cy in _bresenham(a_cell[0], a_cell[1], b_cell[0], b_cell[1]):
        if not (0 <= cx < nx and 0 <= cy < ny) or occ[cy, cx]:
            return False
    return True


def _simplify(occ: np.ndarray, cells: list) -> list:
    """String-pulling: from each kept point, greedily jump to the FARTHEST remaining point
    still in a straight-line, obstacle-free view -- the standard way to turn a blocky grid
    path into the few long, direct legs a controller actually wants to chase."""
    if len(cells) <= 2:
        return cells
    simplified = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not _line_of_sight(occ, cells[i], cells[j]):
            j -= 1
        simplified.append(cells[j])
        i = j
    return simplified


def plan_path(grid, start_xy, goal_xy, robot_radius: float = 0.35, occ_threshold: float = 0.7):
    """Returns a list of (x, y) world-frame waypoints from start_xy to goal_xy (excluding
    start_xy itself, ending exactly at goal_xy), routed through known-free space in `grid`
    and simplified to the minimal number of straight legs -- or None if no path exists in
    the currently known map."""
    occ = grid.probability_map() > occ_threshold
    radius_cells = int(np.ceil(robot_radius / grid.resolution))
    occ = _inflate(occ, radius_cells)

    start_cell = grid.world_to_cell(*start_xy)
    goal_cell = grid.world_to_cell(*goal_xy)
    if 0 <= start_cell[1] < occ.shape[0] and 0 <= start_cell[0] < occ.shape[1]:
        occ[start_cell[1], start_cell[0]] = False
    if 0 <= goal_cell[1] < occ.shape[0] and 0 <= goal_cell[0] < occ.shape[1]:
        occ[goal_cell[1], goal_cell[0]] = False

    cell_path = _astar(occ, start_cell, goal_cell)
    if cell_path is None:
        return None
    cell_path = _simplify(occ, cell_path)

    waypoints = []
    for cx, cy in cell_path[1:]:
        x = grid.xmin + (cx + 0.5) * grid.resolution
        y = grid.ymin + (cy + 0.5) * grid.resolution
        waypoints.append((x, y))
    if waypoints:
        waypoints[-1] = tuple(goal_xy)
    else:
        waypoints = [tuple(goal_xy)]
    return waypoints
