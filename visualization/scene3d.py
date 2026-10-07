from __future__ import annotations

import cv2
import numpy as np


def _project(points: np.ndarray, yaw: float = 0.35, pitch: float = -0.55,
             scale: float = 95.0, cx: int = 420, cy: int = 320):
    if points.size == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)

    p = points.astype(np.float32)
    cyaw, syaw = np.cos(yaw), np.sin(yaw)
    x = p[:, 0] * cyaw - p[:, 2] * syaw
    z = p[:, 0] * syaw + p[:, 2] * cyaw

    cp, sp = np.cos(pitch), np.sin(pitch)
    y = p[:, 1] * cp - z * sp
    depth = p[:, 1] * sp + z * cp

    # Orthographic projection keeps the small demo stable.
    px = (cx + x * scale).astype(np.int32)
    py = (cy - y * scale).astype(np.int32)
    return np.column_stack((px, py)), depth


def render_3d_scene(points: np.ndarray, trajectory: np.ndarray | None = None,
                    path: np.ndarray | None = None,
                    depth_ready: bool = False,
                    calibrated: bool = False,
                    odometry_tracking: bool = False,
                    imu_healthy: bool = False,
                    size: tuple[int, int] = (900, 650)) -> np.ndarray:
    """Lightweight OpenCV 3D-style point-cloud view.

    Coordinates are world X/Y/Z. No browser or extra 3D engine is required.
    """
    width, height = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Ground reference grid.
    for v in np.arange(-3.0, 3.01, 0.5):
        a = np.array([[-3.0, 0.0, v], [3.0, 0.0, v]], dtype=np.float32)
        b = np.array([[v, 0.0, -3.0], [v, 0.0, 3.0]], dtype=np.float32)
        for line in (a, b):
            q, _ = _project(line, cx=width // 2, cy=height // 2 + 40)
            if len(q) == 2:
                cv2.line(canvas, tuple(q[0]), tuple(q[1]), (35, 45, 55), 1)

    if points is not None and len(points):
        q, depth = _project(points[:, :3], cx=width // 2, cy=height // 2 + 40)
        order = np.argsort(depth)[::-1]
        for i in order[::max(1, len(order) // 1800)]:
            x, y = q[i]
            if 2 <= x < width - 2 and 2 <= y < height - 2:
                # Height controls point size, giving a simple spatial cue.
                radius = 2 if points[i, 1] < 1.4 else 3
                cv2.circle(canvas, (int(x), int(y)), radius, (90, 190, 255), -1)

    if trajectory is not None and len(trajectory) > 1:
        q, _ = _project(trajectory[:, :3], cx=width // 2, cy=height // 2 + 40)
        cv2.polylines(canvas, [q.reshape(-1, 1, 2)], False, (255, 210, 40), 3, cv2.LINE_AA)

    if path is not None and len(path) > 1:
        q, _ = _project(path[:, :3], cx=width // 2, cy=height // 2 + 40)
        cv2.polylines(canvas, [q.reshape(-1, 1, 2)], False, (0, 255, 90), 4, cv2.LINE_AA)

    # Camera at the origin of the local/world view.
    center = (width // 2, height // 2 + 40)
    cv2.drawMarker(canvas, center, (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 20, 2)
    cv2.circle(canvas, center, 4, (255, 255, 255), -1)

    cv2.rectangle(canvas, (0, 0), (width, 38), (8, 12, 18), -1)
    cv2.putText(canvas, "ROBOT VISION  |  3D ENVIRONMENT",
                (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (235, 240, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"POINTS {0 if points is None else len(points)}",
                (width - 180, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (120, 210, 255), 1, cv2.LINE_AA)

    # Runtime diagnostics make an empty cloud actionable.
    rows = [
        ("DEPTH", depth_ready),
        ("CALIBRATION", calibrated),
        ("ODOMETRY", odometry_tracking),
        ("IMU", imu_healthy),
    ]
    y = 62
    for label, ok in rows:
        cv2.putText(canvas, f"{label}: {'OK' if ok else 'WAIT'}",
                    (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (90, 225, 140) if ok else (150, 155, 165), 1, cv2.LINE_AA)
        y += 20

    if not depth_ready:
        cv2.putText(canvas, "Waiting for depth inference...",
                    (width // 2 - 145, height // 2 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (180, 190, 205), 1, cv2.LINE_AA)
    elif points is None or len(points) == 0:
        cv2.putText(canvas, "Depth ready - no valid 3D points yet",
                    (width // 2 - 175, height // 2 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (180, 190, 205), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Blue: depth cloud   Yellow: trajectory   Green: planned path",
                (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                (180, 190, 200), 1, cv2.LINE_AA)
    return canvas
