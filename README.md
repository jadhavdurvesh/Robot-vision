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
    ├── YOLO11s + ByteTrack
    ├── metric indoor depth
    ├── visual odometry
    ├── temporal local map
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

- Visual odometry from a single camera is scale-ambiguous by itself.
- Metric depth is used as the distance source for indoor spatial reasoning.
- The planner remains local rather than full persistent SLAM.
- Depth can be less reliable on reflective, transparent, textureless or unusual surfaces.
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
│   ├── obstacle_map.py
│   ├── odometry.py
│   └── local_map.py
├── planning/
│   └── local_planner.py
├── visualization/
│   └── renderer.py
└── tests/
    └── test_navigation.py
```

## Wireless phone-camera demo

The easiest live demo uses a phone browser as the camera and the PC as the vision computer. Both devices should be on the same Wi-Fi network.

### 1. Install

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

The first run of Ultralytics may download the YOLO model.

### 2. Start the phone session

Windows: double-click `start_phone_camera.bat`, or run:

```bash
python phone_server.py
```

The terminal prints a URL such as `https://192.168.x.x:8443/`. Open that URL on the phone. Because the PC creates a local demo certificate, the phone browser will show a certificate warning; accept it for this private LAN demo. Then press **START CAMERA** and allow camera permission.

The PC window named **Robot Vision - Phone Camera** is the processed output. Move the phone forward manually; detections, obstacle regions and the local path are recalculated from each received frame. Press **Q** in the PC vision window to stop.

### 3. If the phone cannot connect

- Confirm both devices are on the same Wi-Fi.
- Allow Python through Windows Firewall on **Private networks**.
- Make sure TCP port `8443` is not blocked.
- Do not use mobile data for this first test.

This wireless demo intentionally uses short-lived JPEG-over-WebSocket transport because it is simple and reliable for a college prototype. It is not intended as the final low-latency WebRTC transport.

### Local monitor URL

When running on a normal Windows PC, do **not** use `localhost:8080` for the project workflow. The server prints the PC's LAN address, for example:

```
LAN monitor: http://192.168.1.105:8080/
```

Open that address on the PC or another device on the same Wi-Fi. The phone camera continues to use the printed HTTPS address on port `8443`.

## Phase 4 spatial perception

Phase 4 adds lightweight monocular visual odometry (ORB + Essential Matrix), metric indoor depth, and a bounded temporal local map. The system deliberately does not claim absolute camera-motion scale from monocular odometry alone. Depth is run periodically at reduced resolution to protect live FPS.

The selected Depth Anything V2 checkpoint is the official Small indoor metric-depth model; its model card describes it as fine-tuned for indoor metric depth estimation and compatible with Transformers. citeturn0search0turn0search4
