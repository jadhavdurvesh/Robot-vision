import numpy as np

from planning.local_planner import LocalPlanner
from vision.detector import Detection
from vision.obstacle_map import build_obstacle_map


def test_empty_scene_plans_forward():
    free = np.full((480, 640), 255, dtype=np.uint8)
    planner = LocalPlanner(samples=15, top_ratio=0.52, bottom_ratio=0.96, lookahead_ratio=0.72, margin=8)
    path = planner.plan(free)
    assert path.clear is True
    assert path.direction == "FORWARD"
    assert len(path.points) >= 2


def test_center_obstacle_causes_detour():
    frame_shape = (480, 640, 3)
    obstacle = Detection(None, 56, "chair", 0.95, 270, 250, 370, 440)
    obstacle_map = build_obstacle_map(frame_shape, [obstacle], {"chair"}, 0.38, 5)

    planner = LocalPlanner(samples=31, top_ratio=0.52, bottom_ratio=0.96, lookahead_ratio=0.72, margin=8)
    path = planner.plan(obstacle_map.free_mask)

    assert path.clear is True
    assert path.direction in {"LEFT", "RIGHT"}


def test_fully_blocked_scene_stops():
    free = np.zeros((480, 640), dtype=np.uint8)
    planner = LocalPlanner(samples=15, top_ratio=0.52, bottom_ratio=0.96, lookahead_ratio=0.72, margin=8)
    path = planner.plan(free)
    assert path.clear is False
    assert path.direction == "BLOCKED"


def test_default_camera_intrinsics_are_valid():
    from vision.calibration import default_intrinsics
    k = default_intrinsics(1280, 720)
    assert k.shape == (3, 3)
    assert k[0, 0] > 0 and k[1, 1] > 0
    assert k[2, 2] == 1


def test_local_map_persists_trajectory_without_jitter():
    import numpy as np
    from vision.local_map import LocalOccupancyMap
    m = LocalOccupancyMap(size=120, meters_per_cell=0.05, decay=0.997)
    shape = (270, 480, 3)
    depth = np.full((270, 480), 2.0, dtype=np.float32)
    result1 = m.update(depth, [], np.array([0., 0., 0.]), shape, 0.38, metric_depth=True)
    result2 = m.update(depth, [], np.array([0., 0., 0.]), shape, 0.38, metric_depth=True)
    assert result2.trajectory_length == 0.0
    assert result2.coverage >= result1.coverage * 0.9
