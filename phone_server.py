from __future__ import annotations

import argparse
import asyncio
import ipaddress
import socket
import ssl
import tempfile
import threading
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

    def process(self, frame: np.ndarray) -> np.ndarray:
        with self.lock:
            detections = self.detector.detect(frame)
            obstacle_map = build_obstacle_map(
                frame.shape, detections, self.obstacle_classes,
                self.roi_top, self.padding
            )
            path = self.planner.plan(obstacle_map.free_mask)
            return draw_scene(frame, detections, obstacle_map, path, 0.0)


def load_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


async def index(request):
    return web.FileResponse(PHONE_PAGE)


async def health(request):
    return web.json_response({"ok": True, "service": "robot-vision-phone-session"})


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
            ok, encoded = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 75])
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

    url = f"https://{ip}:{args.port}/"
    print("\nRobot Vision wireless session")
    print(f"Phone URL: {url}")
    print("Open the URL on the phone, accept the local certificate warning, then tap START CAMERA.")
    print("Press Ctrl+C to stop the session.\n")
    web.run_app(app, host="0.0.0.0", port=args.port, ssl_context=context)


if __name__ == "__main__":
    main()
