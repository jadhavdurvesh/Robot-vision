from __future__ import annotations

from heapq import heappop, heappush

import numpy as np


class MapPathPlanner:
    """A* planner over the persistent local occupancy grid.

    Grid values near 1 are obstacles. Unknown space is penalized but remains
    traversable so the demo can continue exploring instead of getting stuck.
    """

    def __init__(self, obstacle_threshold: float = 0.55, unknown_cost: float = 1.8,
                 diagonal: bool = True):
        self.obstacle_threshold = obstacle_threshold
        self.unknown_cost = unknown_cost
        self.diagonal = diagonal

    def plan(self, grid: np.ndarray, start: tuple[int, int],
             goal: tuple[int, int], max_expansions: int = 30000) -> list[tuple[int, int]]:
        h, w = grid.shape
        sx, sy = start
        gx, gy = goal
        if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
            return []
        if grid[sy, sx] >= self.obstacle_threshold:
            return []

        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)
        ]
        if self.diagonal:
            neighbors += [
                (-1, -1, 1.414), (1, -1, 1.414),
                (-1, 1, 1.414), (1, 1, 1.414)
            ]

        def heuristic(x: int, y: int) -> float:
            return float(np.hypot(gx - x, gy - y))

        open_set = []
        heappush(open_set, (heuristic(sx, sy), 0.0, sx, sy))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        cost_so_far = {(sx, sy): 0.0}

        expansions = 0
        while open_set and expansions < max_expansions:
            _, current_cost, x, y = heappop(open_set)
            if (x, y) == (gx, gy):
                path = [(x, y)]
                while (x, y) in came_from:
                    x, y = came_from[(x, y)]
                    path.append((x, y))
                path.reverse()
                return path

            expansions += 1
            for dx, dy, step in neighbors:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                value = float(grid[ny, nx])
                if value >= self.obstacle_threshold:
                    continue

                # Unknown/weakly explored cells are allowed but discouraged.
                cell_cost = 1.0 + self.unknown_cost * max(0.0, 0.25 - value)
                new_cost = current_cost + step * cell_cost
                old = cost_so_far.get((nx, ny))
                if old is None or new_cost < old:
                    cost_so_far[(nx, ny)] = new_cost
                    came_from[(nx, ny)] = (x, y)
                    heappush(
                        open_set,
                        (new_cost + heuristic(nx, ny), new_cost, nx, ny),
                    )

        return []
