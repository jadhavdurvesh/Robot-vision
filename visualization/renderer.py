from __future__ import annotations

import cv2
import numpy as np

from planning.local_planner import PlannedPath
from vision.detector import Detection
from vision.obstacle_map import ObstacleMap


def draw_scene(
    frame: np.ndarray,
    detections: list[Detection],
    obstacle_map: ObstacleMap,
    path: PlannedPath,
    fps: float,
    inference_ms: float = 0.0,
) -> np.ndarray:
    output = frame.copy()
    obstacle_layer = np.zeros_like(output)
    obstacle_layer[obstacle_map.mask > 0] = (50, 50, 180)
    output = cv2.addWeighted(output, 0.78, obstacle_layer, 0.22, 0)

    for det in detections:
        cv2.rectangle(output, (det.x1, det.y1), (det.x2, det.y2), (0, 220, 255), 2)
        label = f"{det.class_name} {det.confidence:.0%}"
        if det.track_id is not None:
            label += f"  ID:{det.track_id}"
        cv2.putText(
            output,
            label,
            (det.x1, max(22, det.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
        )

    if len(path.points) >= 2:
        pts = np.array(path.points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(output, [pts], False, (0, 255, 0), 6, cv2.LINE_AA)
        for point in path.points:
            cv2.circle(output, point, 5, (0, 255, 0), -1, cv2.LINE_AA)

    height, width = output.shape[:2]
    bottom = (width // 2, height - 12)
    target = path.points[-1] if path.points else bottom
    cv2.arrowedLine(output, bottom, target, (255, 200, 0), 3, cv2.LINE_AA, tipLength=0.15)

    status = f"FPS: {fps:.1f} | INFER: {inference_ms:.0f}ms | OBJECTS: {len(detections)} | PATH: {path.direction}"
    cv2.rectangle(output, (0, 0), (min(width, 700), 40), (20, 20, 20), -1)
    cv2.putText(output, status, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    cv2.putText(output, "Q: quit   R: reset   |   GREEN = FREE-SPACE PATH   |   RED = OBSTACLE", (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return output
