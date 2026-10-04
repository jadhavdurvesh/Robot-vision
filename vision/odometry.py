from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class OdometryState:
    initialized: bool
    tracking: bool
    matches: int
    inliers: int
    confidence: float
    translation: np.ndarray
    rotation_deg: float
    position: np.ndarray
    scale_known: bool = False


class VisualOdometry:
    """Lightweight monocular visual odometry.

    Uses ORB + an essential matrix. Translation scale is intentionally left
    arbitrary because a single monocular camera cannot recover absolute scale
    without calibration/extra sensors.
    """

    def __init__(
        self,
        width: int = 480,
        max_features: int = 700,
        min_matches: int = 24,
        min_inliers: int = 12,
        focal_ratio: float = 0.90,
    ) -> None:
        self.width = width
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.focal_ratio = focal_ratio

        self.orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=8,
            fastThreshold=12,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.prev_gray: np.ndarray | None = None
        self.prev_kp = None
        self.prev_desc: np.ndarray | None = None
        self.position = np.zeros(3, dtype=np.float64)
        self.rotation = np.eye(3, dtype=np.float64)
        self.last = OdometryState(
            initialized=False,
            tracking=False,
            matches=0,
            inliers=0,
            confidence=0.0,
            translation=np.zeros(3),
            rotation_deg=0.0,
            position=self.position.copy(),
        )

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_kp = None
        self.prev_desc = None
        self.position[:] = 0
        self.rotation[:] = np.eye(3)
        self.last = OdometryState(
            initialized=False,
            tracking=False,
            matches=0,
            inliers=0,
            confidence=0.0,
            translation=np.zeros(3),
            rotation_deg=0.0,
            position=self.position.copy(),
        )

    def update(self, frame: np.ndarray) -> OdometryState:
        h, w = frame.shape[:2]
        scale = min(1.0, self.width / max(1, w))
        if scale < 1:
            gray = cv2.resize(
                frame,
                (self.width, max(64, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            gray = frame

        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        kp, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(kp) < 20:
            self.prev_gray, self.prev_kp, self.prev_desc = gray, kp, desc
            self.last = OdometryState(
                initialized=self.prev_gray is not None,
                tracking=False,
                matches=0,
                inliers=0,
                confidence=0.0,
                translation=np.zeros(3),
                rotation_deg=0.0,
                position=self.position.copy(),
            )
            return self.last

        if self.prev_desc is None:
            self.prev_gray, self.prev_kp, self.prev_desc = gray, kp, desc
            self.last = OdometryState(
                initialized=True,
                tracking=False,
                matches=0,
                inliers=0,
                confidence=0.0,
                translation=np.zeros(3),
                rotation_deg=0.0,
                position=self.position.copy(),
            )
            return self.last

        pairs = self.matcher.knnMatch(self.prev_desc, desc, k=2)
        good = [
            m for m, n in pairs
            if m.distance < 0.72 * n.distance
        ]

        if len(good) < self.min_matches:
            self.prev_gray, self.prev_kp, self.prev_desc = gray, kp, desc
            self.last = OdometryState(
                initialized=True,
                tracking=False,
                matches=len(good),
                inliers=0,
                confidence=0.0,
                translation=np.zeros(3),
                rotation_deg=0.0,
                position=self.position.copy(),
            )
            return self.last

        pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in good])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in good])

        focal = max(gray.shape) * self.focal_ratio
        cx, cy = gray.shape[1] * 0.5, gray.shape[0] * 0.5
        K = np.array(
            [[focal, 0, cx], [0, focal, cy], [0, 0, 1]],
            dtype=np.float64,
        )

        E, mask = cv2.findEssentialMat(
            pts_prev,
            pts_curr,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.2,
        )

        if E is None or mask is None:
            self.prev_gray, self.prev_kp, self.prev_desc = gray, kp, desc
            return self.last

        inlier_count, R, t, pose_mask = cv2.recoverPose(
            E, pts_prev, pts_curr, K, mask=mask
        )

        confidence = float(inlier_count) / max(1, len(good))
        tracking = inlier_count >= self.min_inliers and confidence >= 0.30

        translation = t.reshape(3).astype(np.float64)
        rotation_deg = float(
            np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)))
        )

        if tracking:
            # recoverPose gives X_current = R X_previous + t.
            # Camera center in the previous camera frame is -R.T @ t.
            step = (-R.T @ translation)
            step_norm = np.linalg.norm(step)
            if step_norm > 1e-8:
                step /= step_norm

            # Keep translation bounded because monocular scale is unknown.
            step *= min(1.0, 0.15 + 0.85 * confidence)
            self.position += self.rotation @ step
            self.rotation = self.rotation @ R.T

        self.prev_gray, self.prev_kp, self.prev_desc = gray, kp, desc
        self.last = OdometryState(
            initialized=True,
            tracking=tracking,
            matches=len(good),
            inliers=int(inlier_count),
            confidence=confidence,
            translation=translation,
            rotation_deg=rotation_deg,
            position=self.position.copy(),
            scale_known=False,
        )
        return self.last
