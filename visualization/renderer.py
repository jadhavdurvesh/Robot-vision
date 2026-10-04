from __future__ import annotations

import cv2
import numpy as np

from planning.local_planner import PlannedPath
from vision.detector import Detection
from vision.depth import DepthResult
from vision.obstacle_map import ObstacleMap
from vision.odometry import OdometryState
from vision.local_map import LocalMap


def _text(img, text, xy, scale=0.48, color=(235, 240, 245), thickness=1):
    cv2.putText(
        img, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA
    )


def draw_scene(
    frame: np.ndarray,
    detections: list[Detection],
    obstacle_map: ObstacleMap,
    path: PlannedPath,
    fps: float,
    inference_ms: float = 0.0,
    depth_result: DepthResult | None = None,
    odometry: OdometryState | None = None,
    local_map: LocalMap | None = None,
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]

    # --- Optional near-depth overlay ---
    if depth_result is not None:
        near = cv2.resize(
            depth_result.near_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth_layer = np.zeros_like(output)
        depth_layer[near > 0] = (255, 80, 30)
        output = cv2.addWeighted(output, 0.92, depth_layer, 0.08, 0)

    # --- Subtle obstacle overlay ---
    obstacle_layer = np.zeros_like(output)
    obstacle_layer[obstacle_map.mask > 0] = (40, 40, 190)
    output = cv2.addWeighted(output, 0.86, obstacle_layer, 0.14, 0)

    # --- Navigation corridor ---
    corridor_top = int(height * 0.52)
    corridor_bottom = int(height * 0.96)
    cv2.rectangle(
        output,
        (int(width * 0.12), corridor_top),
        (int(width * 0.88), corridor_bottom),
        (255, 180, 0),
        1,
        cv2.LINE_AA,
    )

    # --- Detection boxes ---
    for det in detections:
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 215, 255), 2)

        label = f"{det.class_name} {det.confidence:.0%}"
        if det.track_id is not None:
            label += f" #{det.track_id}"

        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        ly = max(th + 3, y1)
        cv2.rectangle(output, (x1, ly - th - 5), (x1 + tw + 5, ly + 2), (15, 20, 25), -1)
        _text(output, label, (x1 + 3, ly - 2), 0.45, (0, 230, 255), 1)

    # --- Planned path ---
    if len(path.points) >= 2:
        pts = np.array(path.points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(output, [pts], False, (0, 255, 70), 4, cv2.LINE_AA)
        for point in path.points:
            cv2.circle(output, point, 3, (0, 255, 70), -1, cv2.LINE_AA)

    center_bottom = (width // 2, height - 12)
    target = path.points[-1] if path.points else center_bottom

    # Direction arrow
    arrow_color = (0, 255, 70) if path.clear else (0, 60, 255)
    cv2.arrowedLine(
        output, center_bottom, target,
        arrow_color, 3, cv2.LINE_AA, tipLength=0.16
    )

    # --- Compact top status bar ---
    bar_h = 34
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (width, bar_h), (8, 12, 18), -1)
    output = cv2.addWeighted(overlay, 0.86, output, 0.14, 0)

    state = "CLEAR" if path.clear else "BLOCKED"
    state_color = (0, 255, 70) if path.clear else (0, 80, 255)

    _text(output, f"ROBOT VISION", (10, 22), 0.52, (255, 255, 255), 1)
    _text(output, f"{state}", (145, 22), 0.52, state_color, 2)
    _text(output, f"PATH {path.direction}", (235, 22), 0.50, (255, 220, 90), 1)
    _text(output, f"OBJ {len(detections)}", (350, 22), 0.48, (230, 235, 240), 1)
    _text(output, f"{fps:.1f} FPS", (420, 22), 0.48, (180, 220, 255), 1)
    _text(output, f"{inference_ms:.0f}ms", (500, 22), 0.48, (180, 220, 255), 1)
    _text(output, f"CLR {path.clearance:.0%}", (575, 22), 0.44, (150, 235, 190), 1)
    if depth_result is not None:
        _text(output, "DEPTH", (655, 22), 0.44, (255, 170, 90), 1)
    if odometry is not None:
        odo_state = "TRACK" if odometry.tracking else "SEARCH"
        _text(output, f"VO {odo_state}", (710, 22), 0.40, (130, 210, 255), 1)
        _text(output, f"{odometry.inliers}/{odometry.matches}", (775, 22), 0.40, (170, 205, 225), 1)

    # --- Compact object list ---
    if detections:
        panel_x = 8
        panel_y = bar_h + 8
        line_h = 19
        panel_h = 39 + line_h * min(len(detections), 8)

        overlay = output.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (min(width - 8, 260), panel_y + panel_h),
            (8, 12, 18),
            -1,
        )
        output = cv2.addWeighted(overlay, 0.72, output, 0.28, 0)

        _text(output, "OBJECTS", (panel_x + 7, panel_y + 15), 0.43, (255, 255, 255), 1)

        for i, det in enumerate(detections[:8]):
            tid = f"#{det.track_id}" if det.track_id is not None else "-"
            text = f"{i + 1}. {det.class_name}  {det.confidence:.0%}  {tid}"
            _text(
                output,
                text,
                (panel_x + 7, panel_y + 33 + i * line_h),
                0.40,
                (225, 230, 235),
                1,
            )

    # --- Local spatial map inset ---
    if local_map is not None:
        map_img = cv2.resize(local_map.image, (190, 190), interpolation=cv2.INTER_NEAREST)
        mx = width - 200
        my = 48
        overlay = output.copy()
        cv2.rectangle(overlay, (mx - 4, my - 4), (width - 6, my + 194), (8, 12, 18), -1)
        output = cv2.addWeighted(overlay, 0.72, output, 0.28, 0)
        output[my:my + 190, mx:mx + 190] = map_img
        _text(output, "LOCAL MAP", (mx + 6, my + 15), 0.40, (255, 255, 255), 1)
        _text(output, f"COV {local_map.coverage:.1%}", (mx + 6, my + 35), 0.36, (190, 220, 235), 1)
        _text(
            output,
            f"OBS {local_map.obstacle_count}  TRJ {local_map.trajectory_length:.1f}",
            (mx + 6, my + 183),
            0.31,
            (185, 205, 220),
            1,
        )

    # --- Path information card ---
    card_w, card_h = 225, 74
    x1 = width - card_w - 10
    y1 = height - card_h - 10

    overlay = output.copy()
    cv2.rectangle(overlay, (x1, y1), (width - 10, height - 10), (8, 12, 18), -1)
    output = cv2.addWeighted(overlay, 0.76, output, 0.24, 0)

    _text(output, "NAVIGATION", (x1 + 9, y1 + 17), 0.42, (170, 190, 205), 1)
    _text(output, path.direction, (x1 + 9, y1 + 39), 0.62, state_color, 2)
    target_x, target_y = target
    _text(output, f"TARGET  X:{target_x} Y:{target_y}", (x1 + 9, y1 + 59), 0.37, (220, 225, 230), 1)

    # --- Tiny bottom hint ---
    _text(
        output,
        "Q quit   R reset",
        (10, height - 10),
        0.38,
        (210, 215, 220),
        1,
    )

    return output
