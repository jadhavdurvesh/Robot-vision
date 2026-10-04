from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DepthResult:
    depth: np.ndarray
    near_mask: np.ndarray


class MonocularDepth:
    """Optional Depth Anything V2 backend.

    The model is loaded lazily so the base detector can run without depth.
    Depth values are relative, not calibrated metres.
    """

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf", device: str = "auto") -> None:
        self.model_id = model_id
        self.device = device
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()

    def estimate(self, frame: np.ndarray) -> DepthResult:
        import torch
        from PIL import Image

        self._load()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=Image.fromarray(rgb), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)
            predicted = self._processor.post_process_depth_estimation(
                outputs,
                target_sizes=[(frame.shape[0], frame.shape[1])],
            )[0]["predicted_depth"]

        depth = predicted.detach().float().cpu().numpy()
        depth = cv2.normalize(depth, None, 0.0, 1.0, cv2.NORM_MINMAX)

        # Depth Anything's relative output is normalized here only for visualization.
        # Larger values represent farther/closer depending on model convention; do not
        # interpret this mask as metric distance.
        # Depth Anything V2 predicts relative depth: lower normalized values
        # are nearer, higher values are farther. This is NOT metric distance.
        near_threshold = float(np.percentile(depth, 18))
        near_mask = (depth <= near_threshold).astype(np.uint8) * 255
        return DepthResult(depth=depth, near_mask=near_mask)
