from __future__ import annotations

import argparse
import os
import time

import cv2
import yaml

from hybrid.backend import Backend, select_backend
from planning.local_planner import LocalPlanner
from vision.detector import ObjectDetector
from vision.obstacle_map import build_obstacle_map
from visualization.renderer import draw_scene


def parse_source(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Robot Vision live detection and path-planning prototype")
    parser.add_argument("--source", default="0", help="Camera index or mobile-camera stream URL")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--backend", choices=["auto", "local", "cloud"], default=os.getenv("ROBOT_VISION_SELECTED_BACKEND", "auto"))
    parser.add_argument("--cloud-url", default=os.getenv("ROBOT_VISION_CLOUD_URL", ""))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    detector_cfg = cfg["detection"]
    obstacle_cfg = cfg["obstacle_map"]
    planner_cfg = cfg["planner"]

    backend = select_backend(Backend(args.backend), args.cloud_url)
    print(f"Robot Vision backend: {backend.selected} ({backend.reason})")

    if backend.selected == "cloud":
        raise NotImplementedError(
            "Cloud backend selected, but no compatible remote inference service is deployed yet."
        )

    detector = ObjectDetector(
        model_path=detector_cfg["model"],
        confidence=detector_cfg["confidence"],
        iou=detector_cfg["iou"],
        device=detector_cfg["device"],
        imgsz=int(detector_cfg.get("imgsz", 512)),
        half=bool(detector_cfg.get("half", True)),
    )
    planner = LocalPlanner(
        samples=planner_cfg["lateral_samples"],
        top_ratio=planner_cfg["corridor_top_ratio"],
        bottom_ratio=planner_cfg["corridor_bottom_ratio"],
        lookahead_ratio=planner_cfg["lookahead_ratio"],
        margin=planner_cfg["safety_margin_px"],
    )

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
    cap.set(cv2.CAP_PROP_FPS, cfg["camera"]["fps"])

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source: {args.source}")

    previous = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(cfg["camera"]["reconnect_delay_seconds"])
                continue

            detections = detector.detect(frame)
            obstacle_map = build_obstacle_map(
                frame.shape,
                detections,
                set(obstacle_cfg["obstacle_classes"]),
                float(obstacle_cfg["roi_top_ratio"]),
                int(obstacle_cfg["obstacle_padding_px"]),
            )
            path = planner.plan(obstacle_map.free_mask)

            now = time.perf_counter()
            instant_fps = 1.0 / max(1e-6, now - previous)
            fps = instant_fps if fps == 0 else fps * 0.9 + instant_fps * 0.1
            previous = now

            cv2.imshow("Robot Vision", draw_scene(frame, detections, obstacle_map, path, fps))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                planner.reset()
                print("Planner state reset.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
