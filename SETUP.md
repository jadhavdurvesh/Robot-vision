# Robot Vision — Setup & Calibration

## 1. Requirements

- Windows PC
- Python 3.10+ recommended
- NVIDIA GPU recommended for YOLO/depth
- Phone and PC on the same Wi-Fi
- Printed checkerboard for calibration (recommended)

## 2. First-time installation

Open PowerShell in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The first run may download the YOLO/depth models.

## 3. Start the system

Run:

```powershell
.\.venv\Scripts\python.exe phone_server.py
```

The terminal prints a Phone URL similar to:

```text
Phone URL: https://192.168.x.x:8443/
LAN monitor: http://192.168.x.x:8080/
```

On the phone:

1. Connect to the same Wi-Fi as the PC.
2. Open the printed Phone URL.
3. Accept the local certificate warning.
4. Tap **START CAMERA**.
5. Allow camera and motion/orientation permissions.
6. Keep the phone steady for a few seconds.

Two PC windows should appear:

- **Robot Vision - Phone Camera** — detections, depth, obstacles, local map and route.
- **Robot Vision - 3D Environment** — 3D points, persistent environment, trajectory and route.

Keyboard:

- **R** = reset map/odometry/navigation
- **Q** or **Esc** = quit

## 4. Camera calibration

Calibration is recommended for better 3D geometry, odometry and navigation.

### Checkerboard

Use:

- **9 × 6 inner corners**
- therefore **10 × 7 squares**

The calibration script defaults to a **24 mm / 0.024 m** square size. If your printed checkerboard uses another square size, change it in `tools/calibrate_camera.py`.

### Take photos

Create the folder:

```powershell
mkdir calibration
```

Take about **15–20 photos**.

Vary:

- distance
- angle
- left/right position
- top/bottom position
- tilt
- rotation

Keep the entire checkerboard visible and avoid blurry photos.

Example:

```text
calibration/
  img01.jpg
  img02.jpg
  ...
  img20.jpg
```

### Run calibration

From the project root:

```powershell
.\.venv\Scripts\python.exe tools\calibrate_camera.py calibration\*.jpg
```

A successful calibration creates:

```text
calibration/camera.json
```

The application automatically uses:

```yaml
calibration:
  path: calibration/camera.json
```

### RMS error

| RMS | Result |
|---:|---|
| < 0.5 | Excellent |
| 0.5–1.0 | Good |
| 1.0–2.0 | Usually usable |
| > 2.0 | Retake calibration photos |

If the error is high, use sharper images with more varied checkerboard positions and angles.

## 5. If you skip calibration

Robot Vision has an approximate fallback camera model, so the system can still start.

Calibration is recommended for:

- 3D point positions
- obstacle positions
- visual odometry
- persistent maps
- navigation geometry

## 6. Recommended demo

1. Start `phone_server.py`.
2. Open the phone URL.
3. Start the camera.
4. Hold the phone around the intended robot/camera height.
5. Wait for depth and tracking.
6. Move slowly forward.
7. Watch obstacles appear in the map.
8. Watch the 3D environment accumulate.
9. Watch the route update.
10. Press **R** before a new test.

Slow, smooth movement gives better visual odometry.

## 7. Important limitation

This uses a normal phone camera and monocular depth. The 3D environment is therefore an estimated/reconstructed environment, not a LiDAR or RGB-D scan.

Accuracy depends on depth quality, calibration, visual texture, lighting and phone motion.

The navigation system is intended for short demonstration/testing sessions.

## 8. Troubleshooting

### Phone cannot connect

Check that:

- PC and phone are on the same Wi-Fi.
- Windows Firewall allows Python on the private network.
- You use the exact IP/URL printed by the server.

### 3D window is empty

Wait for the depth model to load. The first inference can take longer.

Check the terminal for `[DEPTH]` messages.

### Tracking is unstable

Move slowly and point at textured areas. Blank walls, darkness and motion blur make visual odometry difficult.

### Route is blocked

Press **R** and start from a clear position.

## 9. Main files

```text
phone_server.py
phone_camera.html
config.yaml
tools/calibrate_camera.py

vision/
  detector.py
  depth.py
  odometry.py
  point_cloud.py
  local_map.py

planning/
  local_planner.py
  map_planner.py
  ground_navigation.py
  world_navigation.py

visualization/
  renderer.py
  scene3d.py
```
