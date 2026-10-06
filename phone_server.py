from __future__ import annotations

import argparse
import asyncio
import ipaddress
import socket
import ssl
import tempfile
import threading
import time
import traceback
import json
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from planning.local_planner import LocalPlanner
from vision.detector import ObjectDetector
from vision.depth import MonocularDepth, DepthResult
from vision.obstacle_map import build_obstacle_map
from vision.odometry import VisualOdometry, OdometryState
from vision.local_map import LocalOccupancyMap, LocalMap
from vision.imu import IMUFusion
from vision.calibration import load_intrinsics
from vision.point_cloud import PersistentPointCloud
from visualization.renderer import draw_scene
from visualization.scene3d import render_3d_scene

ROOT = Path(__file__).resolve().parent
PHONE_PAGE = ROOT / "phone_camera.html"


def local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def make_certificate(ip: str) -> tuple[str, str]:
    cert_dir = Path(tempfile.gettempdir()) / "robot-vision-certs"
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_path = cert_dir / "cert.pem"
    key_path = cert_dir / "key.pem"
    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.utcnow()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, ip)])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=30))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(ip))]), critical=False)
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return str(cert_path), str(key_path)




class OpenCVDisplay:
    """Thread-safe frame buffer; OpenCV GUI itself stays on the Windows main thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.running = True

    def show(self, frame: np.ndarray) -> None:
        with self.lock:
            self.frame = frame.copy()

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def close(self) -> None:
        self.running = False


class VisionSession:
    def __init__(self, config: dict):
        d = config["detection"]
        o = config["obstacle_map"]
        p = config["planner"]
        self.detector = ObjectDetector(d["model"], d["confidence"], d["iou"], d["device"], int(d.get("imgsz", 512)), bool(d.get("half", True)))
        self.obstacle_classes = set(o["obstacle_classes"])
        self.roi_top = float(o["roi_top_ratio"])
        self.padding = int(o["obstacle_padding_px"])
        self.planner = LocalPlanner(
            p["lateral_samples"], p["corridor_top_ratio"],
            p["corridor_bottom_ratio"], p["lookahead_ratio"],
            p["safety_margin_px"],
        )
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.frame_count = 0
        self.received_count = 0
        self.last_process_ms = 0.0
        self.fps = 0.0
        self._last_process_time = None
        self._previous_small = None
        self._last_detections = []
        self._frames_since_detection = 0
        runtime = config.get("runtime", {})
        self.adaptive_detection = bool(runtime.get("adaptive_detection", True))
        self.motion_threshold = float(runtime.get("static_motion_threshold", 0.012))
        self.static_detection_interval = max(1, int(runtime.get("static_detection_interval", 3)))
        self.motion_resize = max(64, int(runtime.get("motion_resize", 160)))

        depth_cfg = config.get("depth", {})
        self.depth_enabled = bool(depth_cfg.get("enabled", False))
        self.depth_interval = max(1, int(depth_cfg.get("interval", 5)))
        self.depth_width = max(160, int(depth_cfg.get("input_width", 384)))
        self._depth_frame_count = 0
        self._depth_result: DepthResult | None = None
        self._depth_engine = None
        self._depth_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robot-depth")
        self._depth_future: Future | None = None
        if self.depth_enabled:
            self._depth_engine = MonocularDepth(
                model_id=depth_cfg.get("model", "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"),
                device=depth_cfg.get("device", "auto"),
            )

        odo_cfg = config.get("odometry", {})
        self.odo_enabled = bool(odo_cfg.get("enabled", True))
        self.odo_interval = max(1, int(odo_cfg.get("interval", 2)))
        self._odo_frame_count = 0
        self.imu = IMUFusion()
        self.camera_calibration_path = str(config.get("calibration", {}).get("path", "calibration/camera.json"))
        self.camera_matrix, self.calibrated = load_intrinsics(
            self.camera_calibration_path, 480, 270
        )
        self._odometry = VisualOdometry(
            width=int(odo_cfg.get("input_width", 480)),
            max_features=int(odo_cfg.get("max_features", 700)),
            min_matches=int(odo_cfg.get("min_matches", 24)),
            min_inliers=int(odo_cfg.get("min_inliers", 12)),
        )
        self._odometry.set_intrinsics(self.camera_matrix, image_width=480, image_height=270)

        map_cfg = config.get("mapping", {})
        self.map_enabled = bool(map_cfg.get("enabled", True))
        self.map_interval = max(1, int(map_cfg.get("update_interval", 3)))
        self._map_frame_count = 0
        self._local_map = LocalOccupancyMap(
            size=int(map_cfg.get("size", 320)),
            meters_per_cell=float(map_cfg.get("meters_per_cell", 0.05)),
            decay=float(map_cfg.get("decay", 0.985)),
        )
        self._odo_state = self._odometry.last
        self._local_map_result = None
        self._pending_map_update = False

        cloud_cfg = config.get("point_cloud", {})
        self._point_cloud = PersistentPointCloud(
            max_points=int(cloud_cfg.get("max_points", 50000)),
            sample_step=int(cloud_cfg.get("sample_step", 12)),
            max_depth=float(cloud_cfg.get("max_depth", 6.0)),
        )
        self._trajectory_3d: list[np.ndarray] = []
        self._path_3d: np.ndarray | None = None
        self._scene3d = np.zeros((650, 900, 3), dtype=np.uint8)

    def reset(self) -> None:
        with self.lock:
            self.planner.reset()
            self._odometry.reset()
            self._local_map.reset()
            self._odo_state = self._odometry.last
            self._local_map_result = None
            self._previous_small = None
            self._last_detections = []
            self._frames_since_detection = 0
            self._depth_result = None
            self._depth_frame_count = 0
            self._odo_frame_count = 0
            self._map_frame_count = 0
            self._pending_map_update = False
            self._point_cloud.reset()
            self._trajectory_3d.clear()
            self._path_3d = None
            self._scene3d = np.zeros((650, 900, 3), dtype=np.uint8)

    def _estimate_depth_async(self, frame: np.ndarray) -> DepthResult:
        depth_frame = frame
        scale = min(1.0, self.depth_width / max(1, frame.shape[1]))
        if scale < 1.0:
            depth_frame = cv2.resize(
                frame,
                (self.depth_width, max(64, int(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return self._depth_engine.estimate(depth_frame)

    def process(self, frame: np.ndarray) -> np.ndarray:
        with self.lock:
            started = time.perf_counter()

            # Cheap motion estimate lets us avoid wasting GPU work on nearly
            # identical frames while immediately returning to full detection
            # when the phone moves.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small_h = max(64, int(self.motion_resize * frame.shape[0] / frame.shape[1]))
            small = cv2.resize(gray, (self.motion_resize, small_h), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (5, 5), 0)

            motion = 1.0
            if self._previous_small is not None:
                motion = float(cv2.absdiff(small, self._previous_small).mean()) / 255.0
            self._previous_small = small

            self._frames_since_detection += 1
            run_detection = (
                not self.adaptive_detection
                or not self._last_detections
                or motion >= self.motion_threshold
                or self._frames_since_detection >= self.static_detection_interval
            )

            if run_detection:
                detections = self.detector.detect(frame)
                self._last_detections = detections
                self._frames_since_detection = 0
            else:
                detections = self._last_detections

            self._depth_frame_count += 1

            # Depth is intentionally asynchronous. A cold model load or slow
            # depth inference must never freeze the live object/path pipeline.
            if self._depth_future is not None and self._depth_future.done():
                try:
                    depth_small = self._depth_future.result()
                    depth_map = cv2.resize(
                        depth_small.depth,
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    near_map = cv2.resize(
                        depth_small.near_mask,
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    self._depth_result = DepthResult(
                        depth=depth_map,
                        near_mask=near_map,
                        metric=depth_small.metric,
                    )
                except Exception as exc:
                    print(f"[DEPTH] disabled after error: {exc}", flush=True)
                    traceback.print_exc()
                    self.depth_enabled = False
                    self._depth_engine = None
                finally:
                    self._depth_future = None

            if (
                self.depth_enabled
                and self._depth_engine is not None
                and self._depth_future is None
                and (self._depth_result is None or self._depth_frame_count % self.depth_interval == 0)
            ):
                try:
                    self._depth_future = self._depth_executor.submit(
                        self._estimate_depth_async, frame.copy()
                    )
                except Exception as exc:
                    print(f"[DEPTH] could not schedule inference: {exc}", flush=True)
                    self.depth_enabled = False

            self._odo_frame_count += 1
            if self.odo_enabled and self._odo_frame_count % self.odo_interval == 0:
                depth_for_odo = None if self._depth_result is None else self._depth_result.depth
                self._odo_state = self._odometry.update(
                    frame, depth=depth_for_odo,
                    metric_depth=bool(self._depth_result and self._depth_result.metric),
                    imu_delta=self.imu.consume_delta(),
                )

            obstacle_map = build_obstacle_map(
                frame.shape, detections, self.obstacle_classes,
                self.roi_top, self.padding, self._depth_result
            )

            self._map_frame_count += 1
            if (
                self.map_enabled
                and self._map_frame_count % self.map_interval == 0
            ):
                depth_for_map = None if self._depth_result is None else self._depth_result.depth
                # Map update is completed below after planning so the
                # current route can also be projected into the map.
                self._pending_map_update = True

            path = self.planner.plan(obstacle_map.free_mask)

            if (
                self.map_enabled
                and self._pending_map_update
            ):
                depth_for_map = None if self._depth_result is None else self._depth_result.depth
                self._local_map_result = self._local_map.update(
                    depth_for_map,
                    detections,
                    self._odo_state.position,
                    frame.shape,
                    self.roi_top,
                    metric_depth=bool(self._depth_result and self._depth_result.metric),
                    path=path,
                    metric_scale=bool(self._odo_state.scale_known),
                    camera_matrix=self.camera_matrix,
                    camera_rotation=self._odo_state.rotation,
                )
                self._pending_map_update = False
            # Lightweight persistent 3D environment.
            if self.map_enabled and self._depth_result is not None:
                self._point_cloud.update(
                    self._depth_result.depth,
                    self.camera_matrix,
                    self._odo_state.position,
                    self._odo_state.rotation,
                    metric=bool(self._depth_result.metric),
                )
                if (
                    not self._trajectory_3d
                    or np.linalg.norm(self._odo_state.position - self._trajectory_3d[-1]) > 0.025
                ):
                    self._trajectory_3d.append(self._odo_state.position.astype(np.float32).copy())
                    if len(self._trajectory_3d) > 800:
                        self._trajectory_3d = self._trajectory_3d[-800:]

                if path.points:
                    h, w = frame.shape[:2]
                    route = []
                    for px, py in path.points:
                        z = 0.8 + max(0.0, (h - py) / max(1, h)) * 2.5
                        x = (px - w * 0.5) / max(1.0, w) * z
                        route.append(
                            self._odo_state.position
                            + self._odo_state.rotation @ np.array([x, 0.0, z], dtype=np.float64)
                        )
                    self._path_3d = np.asarray(route, dtype=np.float32)

                self._scene3d = render_3d_scene(
                    self._point_cloud.points,
                    np.asarray(self._trajectory_3d, dtype=np.float32) if self._trajectory_3d else None,
                    self._path_3d,
                )

            elapsed = time.perf_counter() - started
            self.last_process_ms = elapsed * 1000.0
            now = time.perf_counter()
            if self._last_process_time is not None:
                instant = 1.0 / max(1e-6, now - self._last_process_time)
                self.fps = instant if self.fps == 0 else self.fps * 0.8 + instant * 0.2
            self._last_process_time = now
            self.frame_count += 1
            rendered = draw_scene(
                frame, detections, obstacle_map, path,
                self.fps, self.last_process_ms, self._depth_result,
                self._odo_state, self._local_map_result
            )
            ok, encoded = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self.latest_jpeg = encoded.tobytes()
            return rendered


def load_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


async def index(request):
    return web.FileResponse(PHONE_PAGE)


async def health(request):
    return web.json_response({"ok": True, "service": "robot-vision-phone-session"})



async def view_page(request):
    return web.Response(text="""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot Vision - Live Monitor</title>
<style>
body{margin:0;background:#070b10;color:#eef4f8;font-family:system-ui,sans-serif}
header{padding:14px 18px;background:#101820;display:flex;justify-content:space-between;align-items:center}
h1{font-size:20px;margin:0}.status{color:#62e58b}
main{padding:14px;max-width:1400px;margin:auto}
#feed{width:100%;display:block;background:#000;border-radius:12px;min-height:240px;object-fit:contain}
p{color:#9eabb5}
</style></head><body>
<header><h1>Robot Vision — Live Monitor</h1><div id="status" class="status">● WAITING</div></header>
<main><img id="feed" alt="Processed Robot Vision feed"><p id="frames">Frames processed: 0</p><p>Live browser monitor for the Robot Vision engine.</p></main>
<script>
const img=document.getElementById("feed"),status=document.getElementById("status"),frames=document.getElementById("frames");
async function update(){
 try{
  const r=await fetch("/frame?ts="+Date.now(),{cache:"no-store"});
  const data=await r.json();
  frames.textContent="Frames received: "+data.frames_received+" • processed: "+data.frames_processed;
  if(data.available){
   img.src="/frame.jpg?ts="+Date.now();
   status.textContent="● LIVE";
  }else{
   status.textContent="● WAITING FOR PHONE";
  }
 }catch(e){status.textContent="● DISCONNECTED"}
}
setInterval(update,100);
update();
</script></body></html>""", content_type="text/html")


async def latest_frame(request):
    session = request.app["session"]
    frame = session.latest_jpeg
    if frame is None:
        return web.json_response({"available": False, "frames_processed": session.frame_count})
    return web.Response(
        body=frame,
        content_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


async def frame_status(request):
    session = request.app["session"]
    return web.json_response({
        "available": session.latest_jpeg is not None,
        "frames_processed": session.frame_count,
        "frames_received": session.received_count,
    })


async def view_health(request):
    session = request.app["session"]
    return web.json_response({"ok": True, "frames_processed": session.frame_count})


async def websocket(request):
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    session = request.app["session"]
    print(f"[PHONE] connected: {request.remote}", flush=True)
    display = request.app["display"]
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)

    async def inference_worker():
        while not ws.closed:
            data = await queue.get()
            try:
                arr = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                rendered = await asyncio.to_thread(session.process, frame)
                display.show(rendered)

                if session.frame_count <= 3 or session.frame_count % 30 == 0:
                    print(
                        f"[VISION] processed={session.frame_count} "
                        f"received={session.received_count} "
                        f"fps={session.fps:.1f} "
                        f"latency={session.last_process_ms:.0f}ms",
                        flush=True,
                    )
            except Exception as exc:
                # Never let one bad model/frame kill the websocket worker.
                print(f"[VISION] frame processing error: {exc}", flush=True)
                traceback.print_exc()

    worker = asyncio.create_task(inference_worker())
    try:
        async for message in ws:
            if message.type == web.WSMsgType.TEXT:
                try:
                    payload = json.loads(message.data)
                    if payload.get("type") == "imu":
                        self_imu = session.imu
                        self_imu.update(payload.get("timestamp", time.time()), payload.get("quaternion", [1,0,0,0]), payload.get("rotationRate", [0,0,0]))
                        continue
                except (ValueError, TypeError, KeyError):
                    pass
                print(f"[PHONE] message: {message.data}", flush=True)
                await ws.send_str("ACK")
            elif message.type == web.WSMsgType.BINARY:
                session.received_count += 1
                if session.received_count <= 5 or session.received_count % 20 == 0:
                    print(f"[PHONE] frame received: {session.received_count} ({len(message.data)} bytes)", flush=True)
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                try:
                    queue.put_nowait(bytes(message.data))
                except asyncio.QueueFull:
                    pass
            elif message.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                break
    finally:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
    print("[PHONE] disconnected", flush=True)
    return ws


def run_server(args, display):
    ip = local_ip()
    cert, key = make_certificate(ip)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)

    app = web.Application(client_max_size=8 * 1024 * 1024)
    app["session"] = VisionSession(load_config(args.config))
    app["display"] = display
    display.session = app["session"]
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/ws", websocket)
    app.router.add_get("/view", view_page)

    monitor = web.Application()
    monitor["session"] = app["session"]
    monitor.router.add_get("/", view_page)
    monitor.router.add_get("/frame.jpg", latest_frame)
    monitor.router.add_get("/frame", frame_status)
    monitor.router.add_get("/health", view_health)

    async def main_async():
        monitor_runner = web.AppRunner(monitor)
        await monitor_runner.setup()
        await web.TCPSite(monitor_runner, "0.0.0.0", 8080).start()

        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(
            runner, "0.0.0.0", args.port, ssl_context=context
        ).start()

        print("\nRobot Vision wireless session")
        print(f"Phone URL: https://{ip}:{args.port}/")
        print("Open the URL on the phone, accept the certificate warning, then tap START CAMERA.")
        print(f"LAN monitor: http://{ip}:8080/")
        print("OpenCV display: this Windows window")
        print("Press Q in the OpenCV window to stop.\n")

        await asyncio.Event().wait()

    asyncio.run(main_async())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--port", type=int, default=8443)
    args = parser.parse_args()

    # IMPORTANT: Windows OpenCV GUI must run on the main thread.
    display = OpenCVDisplay()
    display.session = None
    server_thread = threading.Thread(
        target=run_server, args=(args, display), daemon=True
    )
    server_thread.start()

    window = "Robot Vision - Phone Camera"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    scene_window = "Robot Vision - 3D Environment"
    cv2.namedWindow(scene_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(scene_window, 900, 650)

    try:
        while display.running:
            frame = display.latest()
            if frame is not None:
                h, w = frame.shape[:2]

                # Letterbox: preserve the COMPLETE phone frame.
                scale = min(1280 / w, 720 / h)
                rw = max(1, int(w * scale))
                rh = max(1, int(h * scale))
                resized = cv2.resize(
                    frame, (rw, rh), interpolation=cv2.INTER_AREA
                )

                display_frame = np.zeros(
                    (720, 1280, 3), dtype=np.uint8
                )
                x = (1280 - rw) // 2
                y = (720 - rh) // 2
                display_frame[y:y + rh, x:x + rw] = resized

                cv2.imshow(window, display_frame)

                session = getattr(display, "session", None)
                if session is not None and session._scene3d is not None:
                    cv2.imshow(scene_window, session._scene3d)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("r"):
                session = getattr(display, "session", None)
                if session is not None:
                    session.reset()
            if key == ord("q") or key == 27:
                display.close()
                break
    finally:
        cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
