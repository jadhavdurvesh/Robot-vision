from __future__ import annotations

from dataclasses import dataclass
import threading
import time
import numpy as np


@dataclass
class IMUSample:
    timestamp: float
    quaternion: np.ndarray
    rotation_rate: np.ndarray


class IMUFusion:
    """Thread-safe phone orientation state used to stabilize visual rotation."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current: IMUSample | None = None
        self.previous_q: np.ndarray | None = None
        self.delta_rotation = np.eye(3, dtype=np.float64)
        self.samples = 0

    @staticmethod
    def quat_normalize(q: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(q))
        return q / n if n > 1e-9 else np.array([1., 0., 0., 0.])

    @staticmethod
    def quat_to_matrix(q: np.ndarray) -> np.ndarray:
        w, x, y, z = IMUFusion.quat_normalize(q)
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ], dtype=np.float64)

    def update(self, timestamp: float, quaternion, rotation_rate=None) -> None:
        q = self.quat_normalize(np.asarray(quaternion, dtype=np.float64))
        rate = np.zeros(3, dtype=np.float64) if rotation_rate is None else np.asarray(rotation_rate, dtype=np.float64)
        with self.lock:
            if self.previous_q is not None:
                self.delta_rotation = self.quat_to_matrix(q) @ self.quat_to_matrix(self.previous_q).T
            self.previous_q = q.copy()
            self.current = IMUSample(float(timestamp), q, rate)
            self.samples += 1

    def consume_delta(self) -> np.ndarray:
        with self.lock:
            delta = self.delta_rotation.copy()
            self.delta_rotation = np.eye(3, dtype=np.float64)
            return delta

    def healthy(self, max_age: float = 0.5) -> bool:
        with self.lock:
            return self.current is not None and (time.time() - self.current.timestamp) <= max_age
