from __future__ import annotations

import argparse
import asyncio
import ipaddress
import socket
import ssl
import tempfile
import threading
import time
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
from vision.obstacle_map import build_obstacle_map
from visualization.renderer import draw_scene

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


class VisionSession:
    def __init__(self, config: dict):
        d = config["detection"]
        o = config["obstacle_map"]
        p = config["planner"]
        self.detector = ObjectDetector(d["model"], d["confidence"], d["iou"], d["device"])
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
        self.last_process_ms = 0.0
        self.fps = 0.0
        self._last_process_time = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        with self.lock:
            started = time.perf_counter()
            detections = self.detector.detect(frame)
            obstacle_map = build_obstacle_map(
                frame.shape, detections, self.obstacle_classes,
                self.roi_top, self.padding
            )
            path = self.planner.plan(obstacle_map.free_mask)
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
                self.fps, self.last_process_ms
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
  frames.textContent="Frames processed: "+data.frames_processed;
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
    })


async def view_health(request):
    session = request.app["session"]
    return web.json_response({"ok": True, "frames_processed": session.frame_count})


async def websocket(request):
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    session = request.app["session"]

    async for message in ws:
        if message.type == web.WSMsgType.BINARY:
            arr = np.frombuffer(message.data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            rendered = await asyncio.to_thread(session.process, frame)
            display_w, display_h = 1280, 720
            h, w = rendered.shape[:2]
            scale = min(display_w / w, display_h / h)
            rw, rh = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(rendered, (rw, rh), interpolation=cv2.INTER_AREA)
            display = np.zeros((display_h, display_w, 3), dtype=np.uint8)
            x = (display_w - rw) // 2
            y = (display_h - rh) // 2
            display[y:y + rh, x:x + rw] = resized
            cv2.imshow("Robot Vision - Phone Camera", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                await ws.close()
                break
            ok, encoded = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ok:
                await ws.send_bytes(encoded.tobytes())
        elif message.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
            break
    return ws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--port", type=int, default=8443)
    args = parser.parse_args()

    ip = local_ip()
    cert, key = make_certificate(ip)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)

    app = web.Application(client_max_size=8 * 1024 * 1024)
    app["session"] = VisionSession(load_config(args.config))
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/ws", websocket)
    app.router.add_get("/view", view_page)

    # Separate browser monitor for GitHub Codespaces / remote development.
    monitor = web.Application()
    monitor["session"] = app["session"]
    monitor.router.add_get("/", view_page)
    monitor.router.add_get("/frame.jpg", latest_frame)
    monitor.router.add_get("/frame", frame_status)
    monitor.router.add_get("/health", view_health)

    async def start_monitor():
        runner = web.AppRunner(monitor)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", 8080)
        await site.start()
        return runner

    async def run_servers():
        await start_monitor()
        await web.TCPSite(
            web.AppRunner(app), "0.0.0.0", args.port
        ).start()

    url = f"https://{ip}:{args.port}/"
    cv2.namedWindow("Robot Vision - Phone Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robot Vision - Phone Camera", 1280, 720)

    print("\nRobot Vision wireless session")
    print(f"Phone URL: {url}")
    print("Open the URL on the phone, accept the local certificate warning, then tap START CAMERA.")
    print("Press Q in the OpenCV window or Ctrl+C in the terminal to stop.\n")
    print(f"LAN monitor (Codespaces/remote): http://{ip}:8080/")
    print("Vision display: OpenCV window on this PC")
    print("Codespaces monitor: http://localhost:8080/ (only when running inside Codespaces)")

    async def main_async():
        await start_monitor()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", args.port, ssl_context=context)
        await site.start()
        await asyncio.Event().wait()

    asyncio.run(main_async())


if __name__ == "__main__":
    main()
