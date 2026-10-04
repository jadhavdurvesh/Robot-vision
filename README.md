# Robot Vision

College-project prototype for real-time robot vision and local path planning using a **wireless mobile phone camera**.

This project is intentionally separate from **DMJ Vision**.

## Architecture

```text
PHONE CAMERA
    │ Wi-Fi / HTTP stream
    ▼
ROBOT VISION
    ├── YOLO detection + tracking
    ├── depth estimation module
    ├── obstacle map
    ├── local path planner
    └── live visualization
```

The operator manually moves the phone forward to simulate robot movement. No physical moving robot is required.

## Hybrid execution

The project uses short-lived sessions rather than an always-on server.

- `auto`: prefer a local GPU; otherwise check the configured cloud backend and fall back locally.
- `local`: run processing on the PC.
- `cloud`: reserved for a deployed compatible remote inference service.

Start the program when testing/demonstrating and press **Q** to stop it. Nothing runs continuously in the background.

## First local demo

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Install:
```bash
pip install -r requirements.txt
```

PC webcam:
```bash
python run.py --backend local --source 0
```

Wireless phone camera:
```bash
python run.py --backend auto --source "http://PHONE_STREAM_URL/video"
```

## Wireless phone setup

Put the phone and PC on the same Wi-Fi network. Use a camera-streaming app/page that exposes an HTTP/MJPEG stream, then pass its stream URL to `--source`.

The repository includes `phone_camera.html` as the beginning of a browser-camera transport.

## Cloud

The cloud backend is intentionally optional. Free GPU services are useful for short experiments or demonstrations, but their quotas and availability are not suitable for assuming an always-on service. The repository contains a small `/health` + `/infer` client contract so a temporary cloud inference service can be plugged in later.

## Current limitations

- A monocular phone camera does not directly provide reliable metric distance.
- The current planner is image-space/local rather than full persistent SLAM.
- Cloud mode requires a deployed compatible remote inference service.
- Wireless camera latency depends on the phone streaming method and Wi-Fi.

## Structure

```text
Robot-vision/
├── app.py
├── run.py
├── config.yaml
├── requirements.txt
├── phone_camera.html
├── hybrid/
│   ├── backend.py
│   └── cloud_client.py
├── vision/
│   ├── camera.py
│   ├── depth.py
│   ├── detector.py
│   └── obstacle_map.py
├── planning/
│   └── local_planner.py
├── visualization/
│   └── renderer.py
└── tests/
    └── test_navigation.py
```
