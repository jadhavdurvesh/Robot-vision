from __future__ import annotations

import time
from dataclasses import dataclass

import cv2


@dataclass
class CameraSource:
    source: str | int
    width: int = 1280
    height: int = 720
    fps: int = 30

    def open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        return cap


class ReconnectingCamera:
    """Camera reader that can recover from temporary wireless stream drops."""

    def __init__(self, source: CameraSource, reconnect_delay: float = 1.0) -> None:
        self.source = source
        self.reconnect_delay = reconnect_delay
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.close()
        self.cap = self.source.open()
        if not self.cap.isOpened():
            self.close()
            raise RuntimeError(f"Could not open camera source: {self.source.source}")

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            self.open()

        ok, frame = self.cap.read()
        if ok:
            return True, frame

        self.close()
        time.sleep(self.reconnect_delay)
        return False, None

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
