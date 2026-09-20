#!/usr/bin/env python3
"""Loopback-only G1 sensor viewer; Python 3.8 standard-library HTTP server.

RGB reads the existing videohub. Depth uses the adjacent bounded V4L2 helper.
LiDAR temporarily starts its driver when needed and restores its initial state.
No movement, calibration, reset, or unrelated service changes are made.
"""
import argparse
from collections import deque
import datetime
import html
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import secrets
import select
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


STALE_SECONDS = 2.0
LEASE_SECONDS = 30.0
MAX_SESSION_SECONDS = 900
DEPTH_HEADER = struct.Struct("<4sIIIQI")
SCALE_LABEL = "原始深度单位 300–5000；未经距离标定；黑色为无有效数据"
LIDAR_DETAIL_FIELDS = ("point_count", "display_points", "frame_id", "warning", "service_state",
                       "lidar_ip", "serial_number", "diagnostic_at", "hms_codes", "hms_checked",
                       "received_frames", "received_hz", "timestamp_errors", "decode_errors",
                       "valid_points", "source_timestamp_ns")
SDK_LOCK = threading.Lock()
SDK_INTERFACE = None


def ensure_sdk(interface):
    """RGB and LiDAR share one DDS factory, including concurrent first starts."""
    global SDK_INTERFACE
    with SDK_LOCK:
        if SDK_INTERFACE is not None and SDK_INTERFACE != interface:
            raise RuntimeError("SDK interface cannot change within a viewer process")
        if SDK_INTERFACE is None:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            ChannelFactoryInitialize(0, interface)
            SDK_INTERFACE = interface


def lidar_factory(interface, lidar_ip):
    # Lazy import keeps /api/health usable without loading the SDK or hardware.
    from lidar_preview import LidarProducer
    return LidarProducer(interface, lidar_ip=lidar_ip, ensure_sdk=ensure_sdk)


class CaptureStopped(Exception):
    pass


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def usb_identity(device, sysfs_root=Path("/sys/class/video4linux")):
    if not re.fullmatch(r"/dev/video[0-9]+", device):
        raise ValueError("Depth device must be an explicit /dev/videoN node")
    link = sysfs_root / Path(device).name / "device"
    if not link.exists():
        raise RuntimeError("Depth node is missing from sysfs: " + device)
    current = link.resolve()
    for parent in (current,) + tuple(current.parents):
        vendor = parent / "idVendor"
        if vendor.is_file():
            name = (parent / "product").read_text().strip()
            serial = (parent / "serial").read_text().strip()
            if vendor.read_text().strip().lower() != "8086" or "435i" not in name.lower() or not serial:
                raise RuntimeError("Depth node is not an identifiable Intel RealSense D435i")
            return {"camera_name": name, "serial": serial, "depth_device": device,
                    "identity_source": "USB sysfs; RGB RPC does not carry a camera serial"}
    raise RuntimeError("No USB camera identity for " + device)


def discover_depth(binary, device="auto", sysfs_root=Path("/sys/class/video4linux"), runner=subprocess.run):
    candidates = [device] if device != "auto" else [
        "/dev/" + path.name for path in sorted(sysfs_root.glob("video*"))
        if re.fullmatch(r"video[0-9]+", path.name)]
    found, failures = [], []
    for candidate in candidates:
        try:
            identity = usb_identity(candidate, sysfs_root)
            reply = runner([str(binary), "--probe", candidate], capture_output=True,
                           text=True, timeout=3, check=False)
            if reply.returncode != 0:
                raise RuntimeError("Depth probe failed: " + reply.stderr[-300:].strip())
            info = json.loads(reply.stdout)
            if info.get("device") != candidate:
                raise RuntimeError("Depth probe returned a different device")
            if info.get("usable_depth") is True and info.get("z16") is True:
                found.append(identity)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            failures.append(str(error))
    if len(found) != 1:
        detail = "; ".join(failures[-2:])
        raise RuntimeError("Expected exactly one identifiable D435i Z16 node; found %d. %s" %
                           (len(found), detail))
    return found[0]


class RGBProducer:
    _client_lock = threading.Lock()
    _client = None

    def __init__(self, interface="eth0"):
        self.interface = interface

    def run(self, publish, stop, metadata):
        import cv2
        import numpy as np
        from unitree_sdk2py.go2.video.video_client import VideoClient
        ensure_sdk(self.interface)
        with self._client_lock:
            if self.__class__._client is None:
                client = VideoClient()
                client.SetTimeout(2.0)
                client.Init()
                self.__class__._client = client
            client = self.__class__._client
        while not stop.is_set():
            begin = time.monotonic()
            code, data = client.GetImageSample()
            if stop.is_set():
                break
            if code != 0 or not data:
                raise RuntimeError("RGB GetImageSample failed (code %s)" % code)
            jpeg = bytes(data)
            if len(jpeg) > 16 * 1024 * 1024:
                raise RuntimeError("RGB sample exceeds size limit")
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError("RGB image decoding failed")
            height, width = frame.shape[:2]
            if not (0 < width <= 4096 and 0 < height <= 4096):
                raise RuntimeError("Unexpected RGB dimensions")
            if not jpeg.startswith(b"\xff\xd8"):
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    raise RuntimeError("RGB JPEG encoding failed")
                jpeg = encoded.tobytes()
            publish(jpeg, width, height)
            stop.wait(max(0, 0.1 - (time.monotonic() - begin)))

    def close(self):
        # RPC polling stops; the manufacturer's videohub remains running.
        return None


def read_exact(stream, length, stop, timeout=3.0):
    chunks, remaining = [], length
    deadline = time.monotonic() + timeout
    while remaining:
        if stop.is_set():
            raise CaptureStopped()
        if time.monotonic() >= deadline:
            raise RuntimeError("Depth frame timed out")
        readable, _, _ = select.select([stream], [], [], min(0.2, max(0, deadline - time.monotonic())))
        if not readable:
            continue
        chunk = os.read(stream.fileno(), remaining)
        if not chunk:
            raise RuntimeError("Depth helper closed its frame stream")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def unpack_depth_header(header):
    if len(header) != DEPTH_HEADER.size:
        raise ValueError("Truncated depth header")
    magic, width, height, sequence, timestamp_us, length = DEPTH_HEADER.unpack(header)
    if magic != b"D16F" or not (0 < width <= 1920 and 0 < height <= 1080):
        raise ValueError("Invalid depth stream header")
    if length != width * height * 2 or timestamp_us == 0:
        raise ValueError("Invalid depth payload length or timestamp")
    return width, height, sequence, timestamp_us, length


def colorize_depth(depth):
    import cv2
    import numpy as np
    valid = (depth != 0) & (depth != 65535)
    normalized = np.clip((depth.astype(np.float32) - 300) * (255.0 / 4700), 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    color[~valid] = 0
    return color, float(valid.mean())


def depth_cleanup_warning(diagnostics, had_frames):
    if any(marker in diagnostics for marker in (
            "restore original format:", "Original format did not restore exactly",
            "VIDIOC_STREAMOFF:", "VIDIOC_REQBUFS release:", "munmap:", "close capture node:")):
        return "Depth helper reported a cleanup failure: " + diagnostics[-1500:]
    expected = ["format_restore=ok"] if had_frames or "capture_format" in diagnostics else []
    if had_frames:
        expected.append("streamoff=ok")
    if any(marker not in diagnostics for marker in expected):
        return "Depth helper cleanup confirmation is missing; original format restoration is unconfirmed"
    return None


def diagnostics_tail(stream):
    stream.seek(0, os.SEEK_END)
    stream.seek(max(0, stream.tell() - 8192))
    return stream.read().decode("utf-8", "replace").strip()


class DepthProducer:
    def __init__(self, binary, device="auto", max_seconds=MAX_SESSION_SECONDS):
        self.binary, self.device = Path(binary), device
        self.max_seconds = max_seconds
        self.process = None
        self.lock = threading.Lock()
        self.closing = False
        self.cleanup_warning = None

    def run(self, publish, stop, metadata):
        import cv2
        import numpy as np
        identity = discover_depth(self.binary, self.device)
        metadata(identity)
        with tempfile.TemporaryFile() as errors:
            with self.lock:
                if self.closing or stop.is_set():
                    return
                self.process = subprocess.Popen(
                    [str(self.binary), "--device", identity["depth_device"],
                     "--seconds", str(self.max_seconds), "--stream"],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors,
                    bufsize=0, close_fds=True)
                process = self.process
            previous_sequence = previous_timestamp = None
            had_frames = False
            try:
                while not stop.is_set():
                    values = unpack_depth_header(read_exact(process.stdout, DEPTH_HEADER.size, stop))
                    width, height, sequence, timestamp_us, length = values
                    raw = read_exact(process.stdout, length, stop)
                    had_frames = True
                    if previous_sequence is not None and (sequence == previous_sequence or timestamp_us <= previous_timestamp):
                        raise RuntimeError("Depth sequence/timestamp did not advance")
                    previous_sequence, previous_timestamp = sequence, timestamp_us
                    depth = np.frombuffer(raw, dtype="<u2").reshape((height, width))
                    color, nonzero = colorize_depth(depth)
                    ok, encoded = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ok:
                        raise RuntimeError("Depth visualization JPEG encoding failed")
                    publish(encoded.tobytes(), width, height, nonzero, timestamp_us)
            except CaptureStopped:
                pass
            except Exception as error:
                self.close()
                details = diagnostics_tail(errors)[-1500:]
                raise RuntimeError("%s%s" % (error, "; " + details if details else ""))
            finally:
                self.close()
                self.cleanup_warning = (self.cleanup_warning or
                                        depth_cleanup_warning(diagnostics_tail(errors), had_frames))
                if process.stdout:
                    process.stdout.close()

    def close(self):
        with self.lock:
            self.closing = True
            process = self.process
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
                    self.cleanup_warning = "Depth helper required forced termination; format restoration is unconfirmed"
            return self.cleanup_warning


class CameraSession:
    def __init__(self, host, factories, clock=time.monotonic, supervise=True,
                 max_seconds=MAX_SESSION_SECONDS):
        if not 1 <= max_seconds <= MAX_SESSION_SECONDS:
            raise ValueError("Session limit must be 1-900 seconds")
        self.clock, self.factories = clock, factories
        self.max_seconds = max_seconds
        self.robot = {"host": host, "camera_name": None, "serial": None, "depth_device": None}
        self.condition = threading.Condition()
        self.lifecycle = threading.Lock()
        self.running = False
        self.generation = 0
        self.started = self.heartbeat_at = 0
        self.last_error = None
        self.cleanup_errors = []
        self.streams = {name: self.empty_stream() for name in self.factories}
        self.producers, self.workers = [], []
        self.stop_event = threading.Event()
        self.shutdown_event = threading.Event()
        self.monitor = None
        if supervise:
            self.monitor = threading.Thread(target=self._monitor, name="sensor-lease", daemon=True)
            self.monitor.start()

    @staticmethod
    def empty_stream():
        return {"frame": None, "frames": 0, "times": deque(maxlen=40), "error": None, "metadata": {}}

    @property
    def cleanup_error(self):
        with self.condition:
            return "\n".join(self.cleanup_errors) or None

    def record_cleanup_error(self, message):
        with self.condition:
            if message not in self.cleanup_errors:
                self.cleanup_errors.append(message)
            self.last_error = self.cleanup_error
            self.condition.notify_all()

    def start(self):
        with self.lifecycle:
            with self.condition:
                if self.shutdown_event.is_set():
                    raise RuntimeError("Server is shutting down")
                if self.running:
                    return self.status()
                if self.cleanup_error:
                    raise RuntimeError("Previous capture cleanup is unconfirmed: " + self.cleanup_error)
                if any(worker.is_alive() for worker in self.workers):
                    raise RuntimeError("Previous capture has not stopped; restart is blocked")
                self.generation += 1
                generation = self.generation
                self.streams = {name: self.empty_stream() for name in self.factories}
                self.robot.update(camera_name=None, serial=None, depth_device=None)
                if "lidar" in self.factories:
                    self.robot.update(lidar_ip=None, lidar_serial=None, lidar_model=None)
                self.last_error = None
                self.running = True
                self.started = self.heartbeat_at = self.clock()
                self.stop_event = threading.Event()
                self.producers, self.workers = [], []
                for name in self.factories:
                    worker = threading.Thread(target=self._capture,
                                              args=(name, generation, self.stop_event), daemon=True,
                                              name="sensor-" + name)
                    self.workers.append(worker)
                for worker in self.workers:
                    worker.start()
                self.condition.notify_all()
            return self.status()

    def _capture(self, name, generation, stop):
        producer = None
        try:
            producer = self.factories[name]()
            with self.condition:
                if generation != self.generation or stop.is_set():
                    return
                self.producers.append(producer)
            producer.run(lambda *args: self.publish(name, generation, *args), stop,
                         lambda info: self.set_metadata(generation, info, name))
            if not stop.is_set():
                raise RuntimeError("Capture ended unexpectedly")
        except Exception as error:
            with self.condition:
                if generation == self.generation and not stop.is_set():
                    self.streams[name]["error"] = str(error)
                    self.last_error = name + ": " + str(error)
                    self.condition.notify_all()
        finally:
            if producer:
                try:
                    warning = producer.close()
                except Exception as error:
                    warning = "Capture cleanup failed: " + str(error)
                if warning:
                    with self.condition:
                        if generation == self.generation:
                            self.record_cleanup_error(warning)

    def set_metadata(self, generation, info, name=None):
        with self.condition:
            if generation != self.generation:
                return
            if name == "lidar":
                self.streams[name]["metadata"].update(info)
                self.robot.update({key: value for key, value in info.items()
                                   if key in ("lidar_ip", "lidar_serial", "lidar_model")})
            elif self.running:
                self.robot.update(info)

    def publish(self, name, generation, payload, width, height, nonzero=None, timestamp_us=None):
        with self.condition:
            if not self.running or generation != self.generation:
                return
            stream = self.streams[name]
            stream["frames"] += 1
            now = self.clock()
            stream["times"].append(now)
            stream["frame"] = {"payload": payload, "width": width, "height": height,
                               "nonzero_fraction": nonzero, "source_timestamp_us": timestamp_us,
                               "received": now, "utc": utc_now(), "sequence": stream["frames"],
                               "generation": generation}
            self.condition.notify_all()

    def heartbeat(self):
        self.enforce_deadlines()
        with self.condition:
            if self.running:
                self.heartbeat_at = self.clock()
        return self.status()

    def stop(self, reason=None, deadline=False):
        with self.lifecycle:
            with self.condition:
                if deadline:
                    if not self.running:
                        return self.status()
                    now = self.clock()
                    reason = ("Maximum %d-second session ended" % self.max_seconds
                              if now - self.started >= self.max_seconds else
                              "Page heartbeat expired; capture stopped"
                              if now - self.heartbeat_at >= LEASE_SECONDS else None)
                    if reason is None:
                        return self.status()
                self.running = False
                self.stop_event.set()
                producers, workers = list(self.producers), list(self.workers)
                for stream in self.streams.values():
                    stream["frame"] = None
                if reason:
                    self.last_error = reason
                self.condition.notify_all()
            for producer in producers:
                try:
                    warning = producer.close()
                    if warning:
                        self.record_cleanup_error(warning)
                except Exception as error:
                    self.record_cleanup_error("Capture cleanup failed: " + str(error))
            join_deadline = time.monotonic() + 8
            for worker in workers:
                worker.join(timeout=max(0, join_deadline - time.monotonic()))
            with self.condition:
                if any(worker.is_alive() for worker in workers):
                    self.record_cleanup_error("Capture thread has not exited; new sessions are blocked")
            return self.status()

    def enforce_deadlines(self):
        # Inspect expiry while holding the lifecycle lock, so an old expiry cannot
        # stop a new session that started while capture cleanup was in progress.
        self.stop(deadline=True)

    def _monitor(self):
        while not self.shutdown_event.wait(0.5):
            self.enforce_deadlines()

    def close(self):
        self.shutdown_event.set()
        self.stop()
        if self.monitor and self.monitor is not threading.current_thread():
            self.monitor.join(timeout=1)

    def status(self):
        with self.condition:
            now = self.clock()
            result = {"running": self.running, "robot": dict(self.robot), "last_error": self.last_error,
                      "cleanup_error": self.cleanup_error,
                      "session": {"remaining_s": max(0, round(self.max_seconds - (now - self.started), 1))
                                  if self.running else 0, "id": self.generation}}
            for name, stream in self.streams.items():
                frame, timestamps = stream["frame"], stream["times"]
                age = max(0, (now - frame["received"]) * 1000) if frame else None
                state = ("idle" if not self.running else "stale" if age is not None and age > STALE_SECONDS * 1000
                         else "error" if stream["error"] else "live" if frame else "starting")
                fps = ((len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
                       if len(timestamps) > 1 and timestamps[-1] > timestamps[0] else 0)
                result[name] = {"state": state, "frames": stream["frames"], "fps": round(fps, 2),
                                "age_ms": round(age, 1) if age is not None else None,
                                "width": frame["width"] if frame else None,
                                "height": frame["height"] if frame else None, "error": stream["error"]}
                if name == "depth":
                    result[name].update(nonzero_fraction=frame["nonzero_fraction"] if frame else None,
                                        scale_label=SCALE_LABEL)
                if name == "lidar":
                    details = dict(frame["payload"]) if frame else {}
                    details.update(stream["metadata"])
                    result[name].update({key: details.get(key) for key in LIDAR_DETAIL_FIELDS})
                    result[name].update(display_frames=result[name]["frames"], display_fps=result[name]["fps"])
                    received_frames, received_hz = details.get("received_frames"), details.get("received_hz")
                    if type(received_frames) is int and received_frames >= 0:
                        result[name]["frames"] = received_frames
                    if (type(received_hz) in (int, float) and math.isfinite(received_hz) and received_hz >= 0):
                        result[name]["fps"] = round(received_hz, 2)
            return result

    def current_frame(self, name):
        with self.condition:
            stream = self.streams.get(name)
            if stream is None:
                return None
            frame = stream["frame"]
            if (not self.running or frame is None or frame["generation"] != self.generation
                    or stream["error"] or self.clock() - frame["received"] > STALE_SECONDS):
                return None
            return dict(frame)


class SensorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port, session, web_root, token=None, instance_id=None):
        self.session, self.web_root = session, Path(web_root)
        self.token = token or secrets.token_hex(32)
        self.instance_id = instance_id or secrets.token_hex(16)
        super().__init__(("127.0.0.1", port), SensorHandler)


class SensorHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, *_args):
        # Do not log request tokens or image URLs.
        pass

    def safe_request(self, require_token=False):
        host = self.headers.get("Host", "")
        try:
            parsed = urlsplit("//" + host)
            if len(self.headers.get_all("Host", [])) != 1:
                return False
            if parsed.hostname not in ("localhost", "127.0.0.1", "::1") or parsed.username or parsed.password:
                return False
            if parsed.path or parsed.query or parsed.fragment or (parsed.port is not None and not 1 <= parsed.port <= 65535):
                return False
        except ValueError:
            return False
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + host:
            return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            return False
        supplied_token = self.headers.get("X-Request-Token", "")
        return not require_token or (supplied_token.isascii() and
                                     secrets.compare_digest(supplied_token, self.server.token))

    def common_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Connection", "close")
        self.close_connection = True

    def reply(self, code, body, content_type="application/json; charset=utf-8", extra=None):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.common_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, str(value))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.safe_request():
            return self.reply(403, {"error": "Local same-origin requests only"})
        path = urlsplit(self.path).path
        if path == "/api/health":
            return self.reply(200, {"ok": True, "instance_id": self.server.instance_id})
        if path == "/api/status":
            return self.reply(200, self.server.session.status())
        if path == "/api/lidar":
            frame = self.server.session.current_frame("lidar")
            if frame is None:
                return self.reply(409, {"error": "No fresh point cloud in the active session"})
            cloud = frame["payload"]
            result = {"session_id": frame["generation"], "sequence": frame["sequence"],
                      "received_at": frame["utc"], "age_ms": max(0, round(
                          (self.server.session.clock() - frame["received"]) * 1000, 1))}
            result.update({key: cloud.get(key) for key in ("points",) + LIDAR_DETAIL_FIELDS})
            return self.reply(200, result)
        if path == "/":
            try:
                page = (self.server.web_root / "index.html").read_text(encoding="utf-8")
            except OSError:
                return self.reply(503, {"error": "Viewer index.html is unavailable"})
            page = page.replace("{{REQUEST_TOKEN}}", self.server.token)
            page = page.replace("{{ROBOT_HOST}}", html.escape(self.server.session.robot["host"], quote=True))
            return self.reply(200, page.encode("utf-8"), "text/html; charset=utf-8",
                              {"Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self'; frame-ancestors 'none'"})
        names = {"/api/snapshot/rgb.jpg": "rgb", "/api/snapshot/depth.jpg": "depth"}
        if path in names:
            name = names[path]
            frame = self.server.session.current_frame(name)
            if frame is None:
                return self.reply(409, {"error": "No fresh frame in the active session"})
            filename = "%s-session%d-frame%d.jpg" % (name, frame["generation"], frame["sequence"])
            return self.reply(200, frame["payload"], "image/jpeg",
                              {"Content-Disposition": 'attachment; filename="%s"' % filename,
                               "X-Frame-Time": frame["utc"], "X-Frame-Age-Ms":
                               round((self.server.session.clock() - frame["received"]) * 1000, 1)})
        streams = {"/stream/rgb.mjpg": "rgb", "/stream/depth.mjpg": "depth"}
        if path in streams:
            return self.mjpeg(streams[path])
        return self.reply(404, {"error": "Not found"})

    def do_POST(self):
        if not self.safe_request(require_token=True):
            return self.reply(403, {"error": "Valid request token and same origin required"})
        # Requests carry no body. Close the connection rather than retaining unread input.
        path = urlsplit(self.path).path
        try:
            if path == "/api/start":
                return self.reply(200, self.server.session.start())
            if path == "/api/stop":
                return self.reply(200, self.server.session.stop())
            if path == "/api/heartbeat":
                return self.reply(200, self.server.session.heartbeat())
        except RuntimeError as error:
            return self.reply(409, {"error": str(error)})
        return self.reply(404, {"error": "Not found"})

    def mjpeg(self, name):
        session = self.server.session
        with session.condition:
            if not session.running:
                return self.reply(409, {"error": "Capture is not running"})
            generation = session.generation
        self.send_response(200)
        self.common_headers()
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        last_sequence = None
        try:
            while True:
                with session.condition:
                    if not session.running or generation != session.generation:
                        break
                    frame = session.current_frame(name)
                    if frame is None or frame["sequence"] == last_sequence:
                        session.condition.wait(timeout=0.5)
                        continue
                age = max(0, (session.clock() - frame["received"]) * 1000)
                header = ("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n"
                          "X-Frame-Time: %s\r\nX-Frame-Age-Ms: %.1f\r\n\r\n" %
                          (len(frame["payload"]), frame["utc"], age)).encode("ascii")
                self.wfile.write(header + frame["payload"] + b"\r\n")
                self.wfile.flush()
                last_sequence = frame["sequence"]
            self.wfile.write(b"--frame--\r\n")
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-label", default="G1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--depth-binary", default=str(Path(__file__).with_name("depth_frame_check")))
    parser.add_argument("--depth-device", default="auto")
    parser.add_argument("--web-root", default=str(Path(__file__).parent))
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--lidar-ip", default="192.168.123.120")
    parser.add_argument("--max-session-seconds", type=int, default=MAX_SESSION_SECONDS)
    parser.add_argument("--instance-id", default=secrets.token_hex(16))
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("port must be 1024-65535")
    if args.depth_device != "auto" and not re.fullmatch(r"/dev/video[0-9]+", args.depth_device):
        parser.error("depth-device must be auto or /dev/videoN")
    if not 1 <= args.max_session_seconds <= MAX_SESSION_SECONDS:
        parser.error("max-session-seconds must be 1-900")
    try:
        ipaddress.IPv4Address(args.lidar_ip)
    except ipaddress.AddressValueError:
        parser.error("lidar-ip must be an IPv4 address")
    return args


def close_server(session, server):
    """Preserve cleanup failures after HTTP has closed, including launcher exit."""
    try:
        session.close()
    except Exception as error:
        session.record_cleanup_error("Viewer shutdown failed: " + str(error))
    finally:
        server.server_close()
    if session.cleanup_error:
        print(json.dumps({"kind": "cleanup_error", "error": session.cleanup_error}, ensure_ascii=False),
              file=sys.stderr, flush=True)
        return 1
    return 0


def main(argv=None):
    args = parse_args(argv)
    session = CameraSession(args.host_label, {
        "rgb": lambda: RGBProducer(args.interface),
        "depth": lambda: DepthProducer(args.depth_binary, args.depth_device, args.max_session_seconds),
        "lidar": lambda: lidar_factory(args.interface, args.lidar_ip)},
        max_seconds=args.max_session_seconds)
    server = SensorHTTPServer(args.port, session, args.web_root, instance_id=args.instance_id)
    def stop_server(_signal, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGHUP, stop_server)
    print(json.dumps({"kind": "ready", "bind": "127.0.0.1", "port": args.port}), flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        exit_status = close_server(session, server)
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
