from __future__ import annotations

import numpy as np


class PersistentPointCloud:
    """Small bounded world-space point cloud built from depth frames."""

    def __init__(self, max_points: int = 50000, sample_step: int = 12, max_depth: float = 6.0):
        self.max_points = max_points
        self.sample_step = max(2, sample_step)
        self.max_depth = max_depth
        self.points = np.empty((0, 3), dtype=np.float32)

    def reset(self) -> None:
        self.points = np.empty((0, 3), dtype=np.float32)

    def update(self, depth: np.ndarray | None, camera_matrix: np.ndarray,
               position: np.ndarray, rotation: np.ndarray,
               metric: bool = False) -> np.ndarray:
        if depth is None:
            return self.points

        h, w = depth.shape[:2]
        K = np.asarray(camera_matrix, dtype=np.float64)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        ys, xs = np.mgrid[0:h:self.sample_step, 0:w:self.sample_step]
        z = depth[ys, xs].astype(np.float64)
        valid = np.isfinite(z)
        if metric:
            valid &= (z > 0.25) & (z < self.max_depth)
            z = np.clip(z, 0.25, self.max_depth)
        else:
            valid &= (z > 0.08) & (z < 1.0)
            z = 0.45 + (1.0 - np.clip(z, 0.0, 1.0)) * 2.2

        if not np.any(valid):
            return self.points

        u = xs[valid].astype(np.float64)
        v = ys[valid].astype(np.float64)
        zv = z[valid]
        camera = np.column_stack(((u - cx) * zv / fx,
                                  (v - cy) * zv / fy,
                                  zv))

        world = np.asarray(position, dtype=np.float64).reshape(1, 3) + (
            np.asarray(rotation, dtype=np.float64) @ camera.T
        ).T

        # Keep only a useful local radius around the current camera.
        delta = world - np.asarray(position, dtype=np.float64).reshape(1, 3)
        keep = np.linalg.norm(delta, axis=1) <= self.max_depth
        world = world[keep].astype(np.float32)

        if len(world):
            self.points = np.vstack((self.points, world))
            # Voxel-like quantisation removes many duplicates without Open3D.
            quant = np.round(self.points / 0.035).astype(np.int32)
            _, unique = np.unique(quant, axis=0, return_index=True)
            self.points = self.points[np.sort(unique)]

            if len(self.points) > self.max_points:
                self.points = self.points[-self.max_points:]

        return self.points
