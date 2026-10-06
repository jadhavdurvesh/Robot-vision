from __future__ import annotations

import cv2
import numpy as np

from planning.map_planner import MapPathPlanner


class GroundNavigator:
    """Convert camera depth/detections into a small ground-plane navigation grid."""

    def __init__(self, size: int = 160, cell_size: float = 0.05,
                 obstacle_radius: int = 3):
        self.size = size
        self.cell_size = cell_size
        self.grid = np.zeros((size, size), dtype=np.float32)
        self.planner = MapPathPlanner()
        self.obstacle_radius = obstacle_radius

    def reset(self):
        self.grid.fill(0)

    def _cell(self, x: float, z: float) -> tuple[int, int]:
        cx = self.size // 2 + int(x / self.cell_size)
        cy = self.size - 8 - int(z / self.cell_size)
        return cx, cy

    def update(self, depth: np.ndarray | None, detections, K: np.ndarray,
               position: np.ndarray, rotation: np.ndarray,
               metric: bool = False) -> tuple[np.ndarray, list[tuple[int, int]]]:
        # Work in the camera-local ground plane. This makes path selection
        # stable even when long-term odometry drifts.
        self.grid *= 0.96
        h, w = depth.shape[:2] if depth is not None else (1, 1)

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        if depth is not None:
            # Only the lower part of the image contributes to walkable ground.
            ys = np.linspace(int(h * 0.55), int(h * 0.94), 24).astype(int)
            xs = np.linspace(int(w * 0.08), int(w * 0.92), 32).astype(int)
            for v in ys:
                for u in xs:
                    z = float(depth[v, u])
                    if not np.isfinite(z):
                        continue
                    if metric:
                        if not 0.25 < z < 5.0:
                            continue
                    else:
                        if not 0.08 < z < 1.0:
                            continue
                        z = 0.45 + (1.0 - float(np.clip(z, 0, 1))) * 2.2

                    # Reject points high above the camera-ground approximation.
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    if y < -0.65:
                        continue
                    gx, gy = self._cell(x, z)
                    if 0 <= gx < self.size and 0 <= gy < self.size:
                        self.grid[gy, gx] = min(0.25, self.grid[gy, gx] + 0.035)

        # Detected objects become hard obstacles at their estimated bottom point.
        if depth is not None:
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
                    if not 0.3 < z < 5.0:
                        continue
                else:
                    z = 0.45 + (1.0 - float(np.clip(z, 0, 1))) * 2.2

                u = (float(det.x1) + float(det.x2)) * 0.5
                x = (u - cx) * z / fx
                gx, gy = self._cell(x, z)
                if 0 <= gx < self.size and 0 <= gy < self.size:
                    cv2.circle(
                        self.grid, (gx, gy),
                        self.obstacle_radius, 1.0, -1,
                    )

        # Keep a safety margin around hard obstacles.
        hard = (self.grid >= 0.65).astype(np.uint8)
        if np.any(hard):
            kernel = np.ones((7, 7), np.uint8)
            inflated = cv2.dilate(hard, kernel)
            self.grid[inflated > 0] = 1.0

        start = (self.size // 2, self.size - 8)
        # Prefer a forward goal, with a slight center bias.
        goal = (self.size // 2, max(8, self.size // 3))
        route = self.planner.plan(self.grid, start, goal)

        # If the direct goal is blocked, search several forward candidates.
        if not route:
            candidates = [
                (self.size // 2 - 28, self.size // 3),
                (self.size // 2 + 28, self.size // 3),
                (self.size // 2 - 45, self.size // 2),
                (self.size // 2 + 45, self.size // 2),
            ]
            best = []
            for candidate in candidates:
                if 0 <= candidate[0] < self.size and self.grid[candidate[1], candidate[0]] < 0.65:
                    trial = self.planner.plan(self.grid, start, candidate, max_expansions=12000)
                    if len(trial) > len(best):
                        best = trial
            route = best

        return self.grid.copy(), route
