from __future__ import annotations

from dataclasses import dataclass

from planning.local_planner import PlannedPath

import cv2
import numpy as np


@dataclass
class MapPoint:
    x: float
    y: float
    confidence: float
    class_name: str


@dataclass
class LocalMap:
    image: np.ndarray
    points: list[MapPoint]
    coverage: float
    trajectory_length: float = 0.0
    obstacle_count: int = 0
    explored_cells: int = 0
    metric_scale: bool = False


class LocalOccupancyMap:
    """Bounded top-down evidence map in arbitrary monocular scale."""

    def __init__(self, size: int = 320, meters_per_cell: float = 0.035, decay: float = 0.985):
        self.size = size
        self.meters_per_cell = meters_per_cell
        self.decay = decay
        self.grid = np.zeros((size, size), dtype=np.float32)
        self.traversal = np.zeros((size, size), dtype=np.float32)
        self.trajectory: list[tuple[float, float]] = []
        self.last_position = np.zeros(3, dtype=np.float64)
        self.trajectory_length = 0.0

    def reset(self) -> None:
        self.grid.fill(0)
        self.traversal.fill(0)
        self.trajectory.clear()
        self.last_position[:] = 0
        self.trajectory_length = 0.0

    def _world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        cx = self.size // 2 + int(x / self.meters_per_cell)
        cy = self.size // 2 - int(y / self.meters_per_cell)
        return cx, cy

    def update(
        self,
        depth: np.ndarray | None,
        detections,
        position: np.ndarray,
        frame_shape: tuple[int, int, int],
        roi_top_ratio: float,
        metric_depth: bool = False,
        path: PlannedPath | None = None,
        metric_scale: bool = False,
    ) -> LocalMap:
        self.grid *= self.decay
        self.traversal *= self.decay

        # Bounded relative trajectory. Monocular odometry has unknown scale.
        current = np.asarray(position, dtype=np.float64)
        if not self.trajectory or np.linalg.norm(current - self.last_position) > 0.015:
            self.trajectory.append((float(current[0]), float(current[2])))
            if len(self.trajectory) > 500:
                self.trajectory = self.trajectory[-500:]
            if len(self.trajectory) > 1:
                px, py = self.trajectory[-2]
                self.trajectory_length += float(np.hypot(current[0] - px, current[2] - py))
            self.last_position = current.copy()

        h, w = frame_shape[:2]
        points: list[MapPoint] = []

        # Approximate ground traversal evidence. This is deliberately local
        # and qualitative because monocular scale is not yet calibrated.
        if depth is not None:
            for yy in np.linspace(int(h * roi_top_ratio), int(h * 0.94), 12).astype(int):
                row = depth[min(h - 1, yy)]
                if row.size == 0:
                    continue
                for xx in np.linspace(int(w * 0.18), int(w * 0.82), 17).astype(int):
                    d = float(row[min(w - 1, xx)])
                    # Metric indoor depth is distance-like; relative depth is
                    # normalized. Traversal evidence remains deliberately weak.
                    valid_depth = (
                        0.25 < d < 6.0 if metric_depth
                        else 0.15 < d < 0.85
                    )
                    if valid_depth:
                        nx = (xx - w * 0.5) / max(1.0, w) * 2.2
                        ny = (h - yy) / max(1.0, h) * 2.5
                        wx = float(position[0] + nx)
                        wy = float(position[2] + ny)
                        cx, cy = self._world_to_cell(wx, wy)
                        if 0 <= cx < self.size and 0 <= cy < self.size:
                            self.traversal[cy, cx] += 0.035

        # Project semantic obstacles into the local map using relative depth.
        if depth is not None:
            for det in detections:
                cx_img = int((det.x1 + det.x2) * 0.5)
                cy_img = int(det.y2)
                if not (0 <= cx_img < w and 0 <= cy_img < h):
                    continue
                local_depth = float(
                    np.median(
                        depth[
                            max(0, det.y1):min(h, det.y2 + 1),
                            max(0, det.x1):min(w, det.x2 + 1),
                        ]
                    )
                )
                # Map image position + relative depth into a compact local
                # coordinate. Absolute metres are intentionally not claimed.
                lateral = (cx_img - w * 0.5) / max(1.0, w) * 2.0
                forward = (1.0 - cy_img / max(1.0, h)) * 2.2
                forward *= 0.65 + 0.7 * (1.0 - np.clip(local_depth, 0, 1))
                wx = float(position[0] + lateral)
                wy = float(position[2] + forward)
                mx, my = self._world_to_cell(wx, wy)
                radius = max(2, int((det.x2 - det.x1) / max(1, w) * 35))
                if 0 <= mx < self.size and 0 <= my < self.size:
                    cv2.circle(self.grid, (mx, my), radius, 1.0, -1)
                    points.append(
                        MapPoint(wx, wy, float(det.confidence), det.class_name)
                    )

        occupied = np.clip(self.grid, 0, 1)
        explored = np.clip(self.traversal, 0, 1)

        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        canvas[:, :, 1] = (explored * 150).astype(np.uint8)
        canvas[:, :, 2] = (occupied * 255).astype(np.uint8)

        center = self.size // 2

        # Persistent odometry trajectory.
        if len(self.trajectory) > 1:
            traj_px = []
            for tx, ty in self.trajectory:
                px, py = self._world_to_cell(tx, ty)
                if 0 <= px < self.size and 0 <= py < self.size:
                    traj_px.append((px, py))
            if len(traj_px) > 1:
                cv2.polylines(
                    canvas,
                    [np.asarray(traj_px, dtype=np.int32).reshape(-1, 1, 2)],
                    False, (255, 210, 40), 2, cv2.LINE_AA,
                )

        # Project current planned route into the relative map.
        if path is not None and path.points:
            route_px = []
            for px_img, py_img in path.points:
                lateral = (px_img - w * 0.5) / max(1.0, w) * 2.0
                forward = (h - py_img) / max(1.0, h) * 2.2
                wx = float(current[0] + lateral)
                wy = float(current[2] + forward)
                px, py = self._world_to_cell(wx, wy)
                if 0 <= px < self.size and 0 <= py < self.size:
                    route_px.append((px, py))
            if len(route_px) > 1:
                cv2.polylines(
                    canvas,
                    [np.asarray(route_px, dtype=np.int32).reshape(-1, 1, 2)],
                    False, (0, 255, 255), 3, cv2.LINE_AA,
                )

        cv2.drawMarker(
            canvas, (center, center), (255, 255, 255),
            cv2.MARKER_TRIANGLE_UP, 14, 2, cv2.LINE_AA,
        )
        cv2.circle(canvas, (center, center), 5, (255, 255, 255), -1, cv2.LINE_AA)

        obstacle_count = len(points)
        explored_cells = int(np.count_nonzero((occupied + explored) > 0.10))
        coverage = float(explored_cells) / float(self.size * self.size)
        return LocalMap(
            canvas, points, coverage,
            trajectory_length=self.trajectory_length,
            obstacle_count=obstacle_count,
            explored_cells=explored_cells,
            metric_scale=metric_scale,
        )
