from __future__ import annotations

import base64
import json
import urllib.request


class CloudVisionClient:
    """Small HTTP client for a short-lived remote inference session."""

    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=self.timeout) as response:
                return 200 <= response.status < 300
        except Exception:
            return False

    def infer(self, frame) -> dict:
        import cv2

        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("Could not encode camera frame")

        payload = json.dumps({
            "image_jpeg_base64": base64.b64encode(encoded.tobytes()).decode("ascii")
        }).encode("utf-8")

        request = urllib.request.Request(
            self.base_url + "/infer",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
