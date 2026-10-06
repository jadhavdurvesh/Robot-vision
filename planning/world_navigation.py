from __future__ import annotations

from heapq import heappop, heappush

import cv2
import numpy as np


class PersistentWorldNavigator:
    """Persistent 2D world occupancy map used for local/global-ish routing.

    The map is intentionally bounded. Evidence from the camera is transformed
    into the odometry world frame, accumulated, inflated for safety, and then
    searched with A*. Unknown space is allowed with a small exploration cost.
    """

    def __init__(self, size: int = 320, meters_per_cell: float = 0.05,
                 decay: float = 0.995, max_range: float = 7.0):
        self.size = size
        self.meters_per_cell = meters_per_cell
        self.decay = decay
        self.max_range = max_range
        self.occupied = np.zeros((size, size), dtype=np.float32)
        self.explored = np.zeros((size, size), dtype=np.float32)
        self.origin_world = np.zeros(2, dtype=np.float64)
        self.initialized = False

    def reset(self):
        self.occupied.fill(0)
        self.explored.fill(0)
        self.origin_world[:] = 0
        self.initialized = False

    def _cell(self, x: float, z: float) -> tuple[int, int]:
        cx = self.size // 2 + int((x - self.origin_world[0]) / self.meters_per_cell)
        cy = self.size // 2 - int((z - self.origin_world[1]) / self.meters_per_cell)
        return cx, cy

    def _world(self, cell: tuple[int, int]) -> tuple[float, float]:
        x, y = cell
        wx = self.origin_world[0] + (x - self.size // 2) * self.meters_per_cell
        wz = self.origin_world[1] + (self.size // 2 - y) * self.meters_per_cell
        return float(wx), float(wz)

    def update(self, depth: np.ndarray | None, detections, K: np.ndarray,
               position: np.ndarray, rotation: np.ndarray,
               metric: bool = False):
        current = np.asarray(position, dtype=np.float64)
        if not self.initialized:
            self.origin_world[:] = current[[0, 2]]
            self.initialized = True

        self.occupied *= self.decay
        self.explored *= self.decay

        if depth is None:
            return

        h, w = depth.shape[:2]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        R = np.asarray(rotation, dtype=np.float64)

        # Sparse ground/obstacle evidence. We deliberately avoid treating the
        # ceiling and upper walls as ground obstacles.
        ys = np.linspace(int(h * 0.56), int(h * 0.94), 22).astype(int)
        xs = np.linspace(int(w * 0.08), int(w * 0.92), 30).astype(int)
        for v in ys:
            for u in xs:
                z = float(depth[v, u])
                if not np.isfinite(z):
                    continue
                if metric:
                    if not 0.25 < z < self.max_range:
                        continue
                else:
                    if not 0.08 < z < 1.0:
                        continue
                    z = 0.45 + (1.0 - float(np.clip(z, 0, 1))) * 2.2

                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                if y < -0.7:
                    continue

                pw = current + R @ np.array([x, y, z], dtype=np.float64)
                distance = float(np.hypot(pw[0] - current[0], pw[2] - current[2]))
                if distance > self.max_range:
                    continue
                gx, gy = self._cell(pw[0], pw[2])
                if 0 <= gx < self.size and 0 <= gy < self.size:
                    self.explored[gy, gx] = min(1.0, self.explored[gy, gx] + 0.05)

        # Explicit detected obstacles are much stronger evidence than generic
        # depth samples.
        for det in detections:
            x0, x1 = max(0, int(det.x1)), min(w, int(det.x2 + 1))
            y0, y1 = max(0, int(det.y1)), min(h, int(det.y2 + 1))
            if x0 >= x1 or y0 >= y1:
                continue
            patch = depth[y0:y1, x0:x1]
            valid = patch[np.isfinite(patch)]
            if valid.size == 0:
                continue
            z = float(np.median(valid))
            if metric:
                if not 0.3 < z < self.max_range:
                    continue
            else:
                z = 0.45 + (1.0 - float(np.clip(z, 0, 1))) * 2.2

            u = (float(det.x1) + float(det.x2)) * 0.5
            x = (u - cx) * z / fx
            y = (float(det.y2) - cy) * z / fy
            pw = current + R @ np.array([x, y, z], dtype=np.float64)
            gx, gy = self._cell(pw[0], pw[2])
            if 0 <= gx < self.size and 0 <= gy < self.size:
                radius = max(3, int((det.x2 - det.x1) / max(1, w) * 48))
                cv2.circle(self.occupied, (gx, gy), radius, 1.0, -1)

    def plan(self, position: np.ndarray, forward_distance: float = 2.2,
             safety_cells: int = 4) -> list[tuple[int, int]]:
        if not self.initialized:
            return []

        start = self._cell(float(position[0]), float(position[2]))
        gx_world = float(position[0])
        gz_world = float(position[2]) + forward_distance
        goal = self._cell(gx_world, gz_world)

        if not self._inside(start) or not self._inside(goal):
            return []

        blocked = (self.occupied >= 0.55).astype(np.uint8)
        if np.any(blocked):
            kernel = np.ones((safety_cells * 2 + 1, safety_cells * 2 + 1), np.uint8)
            blocked = cv2.dilate(blocked, kernel)

        if blocked[start[1], start[0]]:
            blocked[start[1], start[0]] = 0
        if blocked[goal[1], goal[0]]:
            goal = self._nearest_free(goal, blocked)
            if goal is None:
                return []

        route = self._astar(blocked, start, goal)
        return route

    def _inside(self, p):
        return 0 <= p[0] < self.size and 0 <= p[1] < self.size

    def _nearest_free(self, p, blocked):
        for radius in range(1, 30):
            for dx in range(-radius, radius + 1):
                for dy in (-radius, radius):
                    q = (p[0] + dx, p[1] + dy)
                    if self._inside(q) and not blocked[q[1], q[0]]:
                        return q
            for dy in range(-radius + 1, radius):
                for dx in (-radius, radius):
                    q = (p[0] + dx, p[1] + dy)
                    if self._inside(q) and not blocked[q[1], q[0]]:
                        return q
        return None

    def _astar(self, blocked, start, goal):
        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (1, -1, 1.414),
            (-1, 1, 1.414), (1, 1, 1.414),
        ]

        def h(p):
            return float(np.hypot(goal[0] - p[0], goal[1] - p[1]))

        queue = [(h(start), 0.0, start)]
        came = {}
        cost = {start: 0.0}

        while queue:
            _, g, cur = heappop(queue)
            if cur == goal:
                out = [cur]
                while cur in came:
                    cur = came[cur]
                    out.append(cur)
                return out[::-1]

            for dx, dy, step in neighbors:
                q = (cur[0] + dx, cur[1] + dy)
                if not self._inside(q) or blocked[q[1], q[0]]:
                    continue
                # Prefer explored space, but permit exploration.
                unknown_penalty = max(0.0, 0.35 - float(self.explored[q[1], q[0]]))
                new = g + step * (1.0 + 1.5 * unknown_penalty)
                if new < cost.get(q, float("inf")):
                    cost[q] = new
                    came[q] = cur
                    heappush(queue, (new + h(q), new, q))

        return []

    def route_world(self, route: list[tuple[int, int]]) -> np.ndarray:
        if not route:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(
            [[x, 0.0, z] for x, z in (self._world(p) for p in route)],
            dtype=np.float32,
        )
