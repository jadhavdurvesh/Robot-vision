from __future__ import annotations

import numpy as np

from vision.local_map import LocalOccupancyMap
from vision.odometry import VisualOdometry


def test_local_map_starts_empty():
    local = LocalOccupancyMap(size=160)
    result = local.update(
        depth=None,
        detections=[],
        position=np.zeros(3),
        frame_shape=(480, 640, 3),
        roi_top_ratio=0.38,
    )
    assert result.image.shape == (160, 160, 3)
    assert 0.0 <= result.coverage <= 1.0


def test_local_map_accepts_metric_depth():
    local = LocalOccupancyMap(size=160)
    depth = np.full((120, 160), 3.0, dtype=np.float32)
    result = local.update(
        depth=depth,
        detections=[],
        position=np.zeros(3),
        frame_shape=(120, 160, 3),
        roi_top_ratio=0.38,
        metric_depth=True,
    )
    assert result.coverage > 0.0


def test_odometry_initializes_without_crashing():
    vo = VisualOdometry(width=320)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    state = vo.update(frame)
    assert state.initialized is True
    assert state.position.shape == (3,)
