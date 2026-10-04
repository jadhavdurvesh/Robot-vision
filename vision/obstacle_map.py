from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vision.detector import Detection


@dataclass
class ObstacleMap:
    mask: np.ndarray
    free_mask: np.ndarray


def build_obstacle_map(
    frame_shape: tuple[int, int, int],
    detections: list[Detection],
    obstacle_classes: set[str],
    roi_top_ratio: float,
    padding_px: int,
) -> ObstacleMap:
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    roi_top = max(0, min(height - 1, int(height * roi_top_ratio)))
    mask[roi_top:, :] = 255

    for det in detections:
        if det.class_name not in obstacle_classes:
            continue
        x1 = max(0, det.x1 - padding_px)
        y1 = max(roi_top, det.y1 - padding_px)
        x2 = min(width - 1, det.x2 + padding_px)
        y2 = min(height - 1, det.y2 + padding_px)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

    free_mask = cv2.bitwise_not(mask)
    free_mask[:roi_top, :] = 0
    return ObstacleMap(mask=mask, free_mask=free_mask)
