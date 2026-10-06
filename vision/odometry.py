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
        self.fx = None
        self.fy = None
        self.cx_ratio = 0.5
        self.cy_ratio = 0.5
        self.prev_depth = None

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

    def set_intrinsics(self, camera_matrix: np.ndarray, image_width: int | None = None, image_height: int | None = None) -> None:
        k = np.asarray(camera_matrix, dtype=np.float64)
        if k.shape != (3, 3):
            raise ValueError("camera_matrix must be 3x3")
        self.fx = float(k[0, 0])
        self.fy = float(k[1, 1])
        iw = float(image_width or self.width)
        ih = float(image_height or max(1, round(self.width * 9 / 16)))
        self.cx_ratio = float(k[0, 2]) / max(1.0, iw)
        self.cy_ratio = float(k[1, 2]) / max(1.0, ih)

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_kp = None
        self.prev_desc = None
        self.prev_depth = None
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

    def update(self, frame: np.ndarray, depth: np.ndarray | None = None, metric_depth: bool = False, imu_delta: np.ndarray | None = None) -> OdometryState:
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
        fx = self.fx if self.fx is not None else focal
        fy = self.fy if self.fy is not None else focal
        cx, cy = gray.shape[1] * self.cx_ratio, gray.shape[0] * self.cy_ratio
        K = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
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

        # Depth-assisted pose when metric depth is available. PnP converts
        # matched pixels into 3D points, giving translation in depth units.
        metric_pose = None
        metric_inliers = 0
        if metric_depth and depth is not None and self.prev_depth is not None:
            try:
                prev_d = self.prev_depth
                curr_d = depth
                if prev_d.shape != gray.shape:
                    prev_d = cv2.resize(prev_d, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
                if curr_d.shape != gray.shape:
                    curr_d = cv2.resize(curr_d, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
                obj, img = [], []
                for m in good:
                    u, v = self.prev_kp[m.queryIdx].pt
                    cu, cv = kp[m.trainIdx].pt
                    iu, iv = int(round(u)), int(round(v))
                    z = float(prev_d[np.clip(iv, 0, prev_d.shape[0]-1), np.clip(iu, 0, prev_d.shape[1]-1)])
                    if np.isfinite(z) and 0.15 < z < 8.0:
                        obj.append(((u-cx)*z/fx, (v-cy)*z/fy, z))
                        img.append((cu, cv))
                if len(obj) >= 12:
                    ok, rvec, tvec, idx = cv2.solvePnPRansac(
                        np.asarray(obj, dtype=np.float32), np.asarray(img, dtype=np.float32),
                        K, None, flags=cv2.SOLVEPNP_EPNP, reprojectionError=3.0,
                        confidence=0.995, iterationsCount=80)
                    if ok and idx is not None and len(idx) >= self.min_inliers:
                        Rm, _ = cv2.Rodrigues(rvec)
                        metric_pose = (Rm, tvec.reshape(3).astype(np.float64))
                        metric_inliers = int(len(idx))
                        if metric_inliers / max(1, len(obj)) >= 0.30:
                            tracking = True
                            confidence = max(confidence, metric_inliers / max(1, len(obj)))
            except (cv2.error, ValueError, FloatingPointError):
                metric_pose = None

        # Fuse a short IMU rotation delta with the visual estimate. IMU is
        # used only for rotation here; translation remains vision/depth based.
        if imu_delta is not None and np.asarray(imu_delta).shape == (3, 3):
            try:
                visual_rvec, _ = cv2.Rodrigues(R)
                imu_rvec, _ = cv2.Rodrigues(np.asarray(imu_delta, dtype=np.float64))
                fused_rvec = 0.70 * visual_rvec.reshape(3) + 0.30 * imu_rvec.reshape(3)
                R, _ = cv2.Rodrigues(fused_rvec.reshape(3, 1))
            except cv2.error:
                pass

        translation = t.reshape(3).astype(np.float64)
        rotation_deg = float(
            np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)))
        )

        scale_known = False
        if metric_pose is not None:
            Rm, tm = metric_pose
            step = -Rm.T @ tm
            step_norm = float(np.linalg.norm(step))
            if np.isfinite(step_norm) and 0.002 < step_norm < 2.0:
                self.position += self.rotation @ step
                self.rotation = self.rotation @ Rm.T
                translation = tm
                rotation_deg = float(np.degrees(np.arccos(np.clip((np.trace(Rm)-1.0)*0.5, -1.0, 1.0))))
                scale_known = True
        elif tracking:
            step = (-R.T @ translation)
            step_norm = np.linalg.norm(step)
            if step_norm > 1e-8:
                step /= step_norm
            step *= min(1.0, 0.15 + 0.85 * confidence)
            self.position += self.rotation @ step
            self.rotation = self.rotation @ R.T

        self.prev_gray, self.prev_kp, self.prev_desc = gray, kp, desc
        if depth is not None:
            d = depth
            if d.shape != gray.shape:
                d = cv2.resize(d, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
            self.prev_depth = d.astype(np.float32, copy=True)
        else:
            self.prev_depth = None
        self.last = OdometryState(
            initialized=True,
            tracking=tracking,
            matches=len(good),
            inliers=int(inlier_count),
            confidence=confidence,
            translation=translation,
            rotation_deg=rotation_deg,
            position=self.position.copy(),
            scale_known=scale_known,
        )
        return self.last
