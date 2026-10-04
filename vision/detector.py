from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    track_id: Optional[int]
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class ObjectDetector:
    """Fast YOLO detection/tracking wrapper optimized for live video."""

    def __init__(
        self,
        model_path: str,
        confidence: float,
        iou: float,
        device: str = "auto",
        imgsz: int = 512,
        half: bool = True,
    ) -> None:
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou
        self.device = None if device == "auto" else device
        self.imgsz = imgsz
        self.half = half

        # CUDA + FP16 substantially reduces inference cost on supported NVIDIA GPUs.
        if device == "auto":
            try:
                import torch
                self.use_half = bool(torch.cuda.is_available() and half)
            except Exception:
                self.use_half = False
        else:
            self.use_half = bool(half and str(device).startswith("cuda"))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            half=self.use_half,
            device=self.device,
            tracker="vision/robot_bytetrack.yaml",
            max_det=100,
            verbose=False,
        )
        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        names = result.names
        if boxes is None:
            return detections

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            cls = int(boxes.cls[i].item())
            track_id = None if boxes.id is None else int(boxes.id[i].item())
            detections.append(
                Detection(
                    track_id=track_id,
                    class_id=cls,
                    class_name=str(names[cls]),
                    confidence=float(boxes.conf[i].item()),
                    x1=int(xyxy[0]),
                    y1=int(xyxy[1]),
                    x2=int(xyxy[2]),
                    y2=int(xyxy[3]),
                )
            )
        return detections
