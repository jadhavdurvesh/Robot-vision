from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
import urllib.request


class Backend(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    AUTO = "auto"


@dataclass
class BackendStatus:
    selected: str
    reason: str
    cloud_url: str | None = None


def local_gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def cloud_available(url: str, timeout: float = 2.0) -> bool:
    if not url:
        return False
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def select_backend(requested: Backend, cloud_url: str | None = None) -> BackendStatus:
    if requested == Backend.LOCAL:
        return BackendStatus("local", "local backend explicitly selected")

    if requested == Backend.CLOUD:
        if cloud_available(cloud_url or ""):
            return BackendStatus("cloud", "cloud backend reachable", cloud_url)
        return BackendStatus("local", "cloud requested but unavailable; using local fallback", cloud_url)

    if local_gpu_available():
        return BackendStatus("local", "local GPU available")

    if cloud_available(cloud_url or ""):
        return BackendStatus("cloud", "local GPU unavailable; cloud backend reachable", cloud_url)

    return BackendStatus("local", "no cloud backend available; using local CPU fallback", cloud_url)
