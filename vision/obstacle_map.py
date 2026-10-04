from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vision.detector import Detection
from vision.depth import DepthResult


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
    depth_result: DepthResult | None = None,
) -> ObstacleMap:
    height, width = frame_shape[:2]

    # 255 means occupied/obstacle; 0 means currently free.
    obstacle_mask = np.zeros((height, width), dtype=np.uint8)
    roi_top = max(0, min(height - 1, int(height * roi_top_ratio)))

    # Ignore the upper image region when planning local ground navigation.
    planning_region = np.zeros_like(obstacle_mask)
    planning_region[roi_top:, :] = 255

    for det in detections:
        if det.class_name not in obstacle_classes:
            continue

        # Treat the bottom of the detection as its ground-contact estimate.
        # Objects high in the image (e.g. bottles on shelves) should not
        # become floor obstacles merely because they were detected.
        if det.y2 < roi_top:
            continue

        extra_padding = 0
        if depth_result is not None:
            # Use relative depth only to make genuinely near detected objects
            # slightly safer. Never turn the raw depth mask into an obstacle.
            crop = depth_result.depth[
                max(0, det.y1):min(height, det.y2 + 1),
                max(0, det.x1):min(width, det.x2 + 1),
            ]
            if crop.size:
                near_score = float(np.percentile(crop, 20))
                if depth_result.metric:
                    if near_score < 1.0:
                        extra_padding = int(padding_px * 1.5)
                    elif near_score < 1.8:
                        extra_padding = int(padding_px * 0.75)
                else:
                    near_score = float(np.percentile(crop, 80))
                    if near_score > 0.78:
                        extra_padding = int(padding_px * 1.5)
                    elif near_score > 0.62:
                        extra_padding = int(padding_px * 0.75)

        total_padding = padding_px + extra_padding
        x1 = max(0, det.x1 - total_padding)
        y1 = max(roi_top, det.y1 - total_padding)
        x2 = min(width - 1, det.x2 + total_padding)
        y2 = min(height - 1, det.y2 + total_padding)

        if x2 > x1 and y2 > y1:
            cv2.rectangle(obstacle_mask, (x1, y1), (x2, y2), 255, -1)

    obstacle_mask = cv2.bitwise_and(obstacle_mask, planning_region)
    free_mask = cv2.bitwise_and(cv2.bitwise_not(obstacle_mask), planning_region)

    return ObstacleMap(mask=obstacle_mask, free_mask=free_mask)
