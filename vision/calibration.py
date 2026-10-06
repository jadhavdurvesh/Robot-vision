from __future__ import annotations

from pathlib import Path
import json
import cv2
import numpy as np


def default_intrinsics(width: int, height: int) -> np.ndarray:
    focal = 0.90 * max(width, height)
    return np.array([[focal, 0, width * 0.5],
                     [0, focal, height * 0.5],
                     [0, 0, 1]], dtype=np.float64)


def load_intrinsics(path: str | Path, width: int, height: int) -> tuple[np.ndarray, bool]:
    p = Path(path)
    if not p.exists():
        return default_intrinsics(width, height), False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        k = np.asarray(data["camera_matrix"], dtype=np.float64)
        if k.shape != (3, 3):
            raise ValueError("camera_matrix must be 3x3")
        src_w = float(data.get("image_width", width))
        src_h = float(data.get("image_height", height))
        sx, sy = width / max(1.0, src_w), height / max(1.0, src_h)
        k[0, 0] *= sx
        k[0, 2] *= sx
        k[1, 1] *= sy
        k[1, 2] *= sy
        return k, True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return default_intrinsics(width, height), False


def calibrate_checkerboard(image_paths: list[str], board_size=(9, 6), square_size=0.024) -> dict:
    cols, rows = board_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)
    objpoints, imgpoints = [], []
    image_size = None
    for filename in image_paths:
        img = cv2.imread(str(filename))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=0)
        if ok:
            objpoints.append(objp.copy())
            imgpoints.append(corners)
    if len(objpoints) < 8 or image_size is None:
        raise ValueError(f"Need at least 8 valid checkerboard images; found {len(objpoints)}")
    rms, camera_matrix, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    return {"camera_matrix": camera_matrix.tolist(), "distortion": dist.reshape(-1).tolist(),
            "image_width": image_size[0], "image_height": image_size[1],
            "rms_error": float(rms), "board_size": [cols, rows], "square_size_m": float(square_size)}


def save_calibration(path: str | Path, result: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2), encoding="utf-8")
