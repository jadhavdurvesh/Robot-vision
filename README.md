# Robot Vision

A college-project prototype for real-time robot vision and local path planning using a mobile phone camera connected to a PC.

## Scope

This project is intentionally separate from **DMJ Vision**. It is a college robotics/vision project and does not require a physical moving robot. The phone acts as the camera, while the operator moves it manually to simulate robot motion.

## Pipeline

mobile camera -> video input -> YOLO detection/tracking -> obstacle map -> local path planner -> live visualization

## Features

- Live camera input from a PC camera or phone stream URL.
- Real-time detection of people and common objects with YOLO.
- Persistent object IDs using YOLO tracking.
- Detection-based obstacle map with a configurable safety margin.
- Local free-space path generation toward the forward direction.
- Live overlay showing object boxes, labels, obstacle regions, path, direction, and FPS.

## Quick start

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Then:
```bash
pip install -r requirements.txt
python app.py --source 0
```

For a mobile-camera streaming app:
```bash
python app.py --source "http://PHONE_STREAM_URL/video"
```

## Controls

- Q — quit
- R — reset planner state

## Important limitation

A single monocular phone camera does not directly provide reliable metric distance. The initial planner therefore operates in image space using detected obstacle regions. A depth-estimation module can be added later without changing the planner interface.

## Structure

```text
Robot-vision/
├── app.py
├── config.yaml
├── requirements.txt
├── vision/
│   ├── __init__.py
│   ├── detector.py
│   └── obstacle_map.py
├── planning/
│   ├── __init__.py
│   └── local_planner.py
└── visualization/
    ├── __init__.py
    └── renderer.py
```
