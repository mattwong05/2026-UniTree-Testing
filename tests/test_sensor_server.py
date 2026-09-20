"""Viewer tests use loopback HTTP and fake producers; never open robot devices."""
import contextlib
import http.client
import importlib.util
import io
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location(
    "sensor_server", Path(__file__).resolve().parents[1] / "web/sensor_server.py")
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)
JPEG = b"\xff\xd8test-frame\xff\xd9"


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeProducer:
    def __init__(self):
        self.ready = threading.Event()
        self.publish = None
        self.closed = False

    def run(self, publish, stop, metadata):
        self.publish = publish
        self.metadata = metadata
        self.ready.set()
        stop.wait()

    def close(self):
        self.closed = True


class SessionFixture(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.created = {"rgb": [], "depth": []}
        def factory(name):
            def create():
                producer = FakeProducer()
                self.created[name].append(producer)
                return producer
            return create
        self.session = viewer.CameraSession(
            'G1 <test> "host"', {name: factory(name) for name in self.created},
            clock=self.clock, supervise=False)
        self.addCleanup(self.session.close)

    def start_capture(self):
        status = self.session.start()
        deadline = time.monotonic() + 2
        while any(not items or not items[-1].ready.is_set() for items in self.created.values()):
            if time.monotonic() > deadline:
                self.fail("Fake producers did not start")
            time.sleep(0.005)
        return status

    def publish(self, name="rgb", jpeg=JPEG):
        self.created[name][-1].publish(jpeg, 640, 480, 0.5 if name == "depth" else None)


class SessionTests(SessionFixture):
    def test_start_is_idempotent_and_stop_discards_old_session_images(self):
        self.start_capture()
        first_generation = self.session.generation
        old_publish = self.created["rgb"][-1].publish
        self.publish()
        self.assertEqual(self.session.current_frame("rgb")["payload"], JPEG)
        self.clock.now += 5
        self.session.start()
        self.assertEqual(self.session.generation, first_generation)
        self.assertEqual(self.session.started, 1000)
        self.assertEqual([len(items) for items in self.created.values()], [1, 1])
        self.session.stop()
        self.assertTrue(all(items[-1].closed for items in self.created.values()))
        self.assertIsNone(self.session.current_frame("rgb"))
        self.start_capture()
        old_publish(b"old-image", 10, 10)
        self.assertIsNone(self.session.current_frame("rgb"))
        self.assertEqual(self.session.status()["rgb"]["frames"], 0)
        self.assertEqual(self.session.status()["rgb"]["state"], "starting")

    def test_freshness_fps_and_error_do_not_offer_old_frames(self):
        self.start_capture()
        self.publish("depth")
        self.clock.now += 0.1
        self.publish("depth")
        status = self.session.status()["depth"]
        self.assertEqual(status["state"], "live")
        self.assertEqual(status["fps"], 10)
        self.assertEqual(status["nonzero_fraction"], 0.5)
        self.clock.now += 2.1
        self.assertEqual(self.session.status()["depth"]["state"], "stale")
        self.assertIsNone(self.session.current_frame("depth"))
        self.publish()
        self.session.streams["rgb"]["error"] = "RPC failed"
        self.assertEqual(self.session.status()["rgb"]["state"], "error")
        self.assertIsNone(self.session.current_frame("rgb"))

    def test_usb_identity_metadata_reaches_status_and_is_cleared_on_restart(self):
        self.start_capture()
        identity = {"serial": "TEST_USB", "usb_vid": "8086", "usb_pid": "0b3a",
                    "usb_speed_mbps": 5000.0, "usb_bus_path": "2-2.2", "usb_sysfs_path": "/sys/example/2-2.2"}
        self.created["depth"][-1].metadata(identity)
        for key, value in identity.items():
            self.assertEqual(self.session.status()["robot"][key], value)
        self.session.robot.update({key: "old" for key in viewer.RGB_IDENTITY_FIELDS})
        self.session.stop()
        self.start_capture()
        for key in viewer.USB_IDENTITY_FIELDS + viewer.RGB_IDENTITY_FIELDS:
            self.assertIsNone(self.session.status()["robot"][key])

    def test_status_does_not_extend_lease_and_late_heartbeat_cannot_revive(self):
        self.start_capture()
        self.clock.now += 29
        self.session.status()
        self.assertEqual(self.session.heartbeat_at, 1000)
        self.session.heartbeat()
        self.clock.now += 29
        self.session.enforce_deadlines()
        self.assertTrue(self.session.running)
        self.clock.now += 1
        result = self.session.heartbeat()
        self.assertFalse(result["running"])
        self.assertIn("heartbeat expired", result["last_error"])

    def test_heartbeat_never_extends_absolute_session_limit(self):
        self.start_capture()
        for _ in range(44):
            self.clock.now += 20
            self.session.heartbeat()
        self.assertEqual(self.session.status()["session"]["remaining_s"], 20)
        self.clock.now += 20
        self.session.heartbeat()
        self.assertFalse(self.session.running)
        self.assertIn("900-second", self.session.last_error)

    def test_short_limit_shutdown_and_factory_error(self):
        self.session.max_seconds = 15
        self.start_capture()
        self.clock.now += 15
        self.session.enforce_deadlines()
        self.assertFalse(self.session.running)
        self.assertIn("15-second", self.session.last_error)
        self.session.factories["rgb"] = mock.Mock(side_effect=RuntimeError("SDK unavailable"))
        self.session.start()
        with self.session.condition:
            self.session.condition.wait_for(lambda: self.session.streams["rgb"]["error"] is not None, 2)
        self.assertEqual(self.session.status()["rgb"]["state"], "error")
        self.assertIn("SDK unavailable", self.session.last_error)
        self.session.close()
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            self.session.start()

    def test_cleanup_failure_persists_blocks_restart_and_exits_nonzero(self):
        self.start_capture()
        self.created["depth"][-1].close = mock.Mock(return_value="Original format restoration is unconfirmed")
        status = self.session.stop()
        self.assertIn("restoration is unconfirmed", status["cleanup_error"])
        self.assertEqual(status["last_error"], status["cleanup_error"])
        with self.assertRaisesRegex(RuntimeError, "cleanup is unconfirmed"):
            self.session.start()
        log = io.StringIO()
        server = mock.Mock()
        with contextlib.redirect_stderr(log):
            self.assertEqual(viewer.close_server(self.session, server), 1)
        server.server_close.assert_called_once_with()
        self.assertEqual(json.loads(log.getvalue())["kind"], "cleanup_error")
        self.assertEqual(len(self.session.cleanup_errors), 1)

    def test_normal_lease_expiry_is_not_a_cleanup_error(self):
        self.start_capture()
        self.clock.now += 31
        self.session.enforce_deadlines()
        self.assertIn("heartbeat expired", self.session.last_error)
        self.assertIsNone(self.session.cleanup_error)
        log = io.StringIO()
        with contextlib.redirect_stderr(log):
            self.assertEqual(viewer.close_server(self.session, mock.Mock()), 0)
        self.assertEqual(log.getvalue(), "")


class HTTPTests(SessionFixture):
    def setUp(self):
        super().setUp()
        self.web = tempfile.TemporaryDirectory()
        self.addCleanup(self.web.cleanup)
        Path(self.web.name, "index.html").write_text(
            '<b>{{ROBOT_HOST}}</b><meta content="{{REQUEST_TOKEN}}">', encoding="utf-8")
        self.server = viewer.SensorHTTPServer(0, self.session, self.web.name, token="a" * 64)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.port = self.server.server_address[1]

    def stop_server(self):
        self.session.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def request(self, path, method="GET", headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        try:
            connection.request(method, path, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def post(self, path, headers=None):
        return self.request(path, "POST", headers or {"X-Request-Token": self.server.token})

    def test_health_status_and_template_never_start_hardware(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        code, headers, body = self.request("/api/health")
        self.assertEqual((code, json.loads(body)), (200, {"ok": True, "instance_id": self.server.instance_id}))
        self.assertIn("no-store", headers["Cache-Control"])
        self.request("/api/status")
        code, headers, body = self.request("/")
        self.assertEqual(code, 200)
        self.assertIn(b"G1 &lt;test&gt; &quot;host&quot;", body)
        self.assertIn(self.server.token.encode(), body)
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertEqual(self.created, {"rgb": [], "depth": []})

    def test_health_distinguishes_different_server_instances(self):
        first = json.loads(self.request("/api/health")[2])["instance_id"]
        second = viewer.SensorHTTPServer(0, self.session, self.web.name, instance_id="different-robot-session")
        self.addCleanup(second.server_close)
        self.assertNotEqual(first, second.instance_id)
        self.assertEqual(second.instance_id, "different-robot-session")

    def test_csrf_host_origin_and_token_guards(self):
        for headers in ({}, {"X-Request-Token": "wrong"}, {"X-Request-Token": "\u00e9"},
                        {"X-Request-Token": self.server.token, "Origin": "http://evil.example"},
                        {"X-Request-Token": self.server.token, "Sec-Fetch-Site": "cross-site"},
                        {"X-Request-Token": self.server.token, "Host": "evil.example"},
                        {"X-Request-Token": self.server.token, "Host": "localhost@evil.example"},
                        {"X-Request-Token": self.server.token, "Origin": "null"},
                        {"X-Request-Token": self.server.token, "Origin": "https://127.0.0.1:%d" % self.port}):
            with self.subTest(headers=tuple(headers.keys())):
                self.assertEqual(self.request("/api/start", "POST", headers)[0], 403)
        self.assertEqual(self.request("/api/health", headers={"Host": "evil.example"})[0], 403)
        headers = {"X-Request-Token": self.server.token, "Host": "localhost:9876",
                   "Origin": "http://localhost:9876"}
        self.assertEqual(self.post("/api/start", headers)[0], 200)
        self.assertTrue(self.session.running)
        self.assertEqual(self.post("/api/stop")[0], 200)

    def test_snapshots_require_fresh_active_session_and_include_time(self):
        self.assertEqual(self.request("/api/snapshot/rgb.jpg")[0], 409)
        self.start_capture()
        self.publish()
        code, headers, body = self.request("/api/snapshot/rgb.jpg")
        self.assertEqual((code, body), (200, JPEG))
        self.assertIn("X-Frame-Time", headers)
        self.assertEqual(float(headers["X-Frame-Age-Ms"]), 0)
        self.assertIn("session1-frame1.jpg", headers["Content-Disposition"])
        self.clock.now += 2.01
        self.assertEqual(self.request("/api/snapshot/rgb.jpg")[0], 409)
        self.assertEqual(self.post("/api/stop")[0], 200)
        self.assertEqual(self.request("/api/snapshot/rgb.jpg")[0], 409)

    def test_mjpeg_sends_new_frames_and_ends_on_stop(self):
        self.assertEqual(self.request("/stream/rgb.mjpg")[0], 409)
        self.start_capture()
        self.publish()
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        self.addCleanup(connection.close)
        connection.request("GET", "/stream/rgb.mjpg")
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIn("multipart/x-mixed-replace", response.getheader("Content-Type"))
        self.assertEqual(response.readline(), b"--frame\r\n")
        lines = []
        while True:
            line = response.readline()
            if line == b"\r\n":
                break
            lines.append(line)
        self.assertIn(b"X-Frame-Time:", b"".join(lines))
        self.assertEqual(response.read(len(JPEG) + 2), JPEG + b"\r\n")
        self.session.stop()
        self.assertEqual(response.read(), b"--frame--\r\n")

    def test_lidar_shares_session_and_rejects_stale_or_previous_cloud(self):
        self.created["lidar"] = []
        def create_lidar():
            producer = FakeProducer()
            self.created["lidar"].append(producer)
            return producer
        self.session.factories["lidar"] = create_lidar
        self.assertEqual(self.request("/api/lidar")[0], 409)
        self.start_capture()
        producer = self.created["lidar"][-1]
        service_state = {"name": "lidar_driver", "initial": 1, "current": 0, "owned_start": True,
                         "restore_required": True, "restore_confirmed": False, "phase": "running"}
        producer.metadata({"lidar_ip": "192.168.123.120", "lidar_serial": "TEST_LIDAR",
                           "service_state": service_state, "warning": "Window dirty"})
        cloud = {"frame_id": "lidar", "points": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                 "point_count": 20000, "display_points": 2,
                 "service_state": service_state, "warning": "Window dirty", "hms_checked": True,
                 "hms_codes": ["0x01040002"], "diagnostic_at": "2026-09-20T00:00:00Z",
                 "received_frames": 23, "received_hz": 9.99, "timestamp_errors": 0,
                 "decode_errors": 0, "valid_points": 19000}
        producer.publish(cloud, 0, 0)
        code, headers, body = self.request("/api/lidar")
        self.assertEqual(code, 200)
        data = json.loads(body)
        self.assertEqual(data["points"], cloud["points"])
        self.assertTrue(data["hms_checked"])
        self.assertEqual(data["service_state"], service_state)
        self.assertEqual((data["session_id"], data["sequence"], data["age_ms"]), (1, 1, 0))
        self.assertIn("received_at", data)
        status = self.session.status()
        self.assertEqual(status["lidar"]["state"], "live")
        self.assertEqual(status["lidar"]["point_count"], 20000)
        self.assertEqual((status["lidar"]["frames"], status["lidar"]["fps"]), (23, 9.99))
        self.assertEqual((status["lidar"]["display_frames"], status["lidar"]["display_fps"]), (1, 0))
        self.assertEqual((data["received_frames"], data["received_hz"], data["valid_points"]), (23, 9.99, 19000))
        self.assertEqual(status["lidar"]["hms_codes"], ["0x01040002"])
        self.assertEqual(status["robot"]["lidar_serial"], "TEST_LIDAR")
        self.assertIsNone(status["robot"]["serial"])
        self.clock.now += 2.1
        self.assertEqual(self.request("/api/lidar")[0], 409)
        self.assertEqual(self.session.status()["lidar"]["state"], "stale")
        self.session.stop()
        producer.metadata({"service_state": "restored"})
        self.assertEqual(self.session.status()["lidar"]["service_state"], "restored")
        self.assertTrue(all(items[-1].closed for items in self.created.values()))
        self.start_capture()
        producer.publish(cloud, 0, 0)
        producer.metadata({"service_state": "OLD", "lidar_serial": "OLD"})
        self.assertEqual(self.request("/api/lidar")[0], 409)
        self.assertIsNone(self.session.status()["robot"]["lidar_serial"])
        self.assertEqual(self.session.status()["lidar"]["frames"], 0)

    def test_lidar_endpoint_requires_loopback_origin_even_with_cloud(self):
        self.assertEqual(self.request("/api/lidar", headers={"Origin": "http://evil.example"})[0], 403)


class DepthProtocolTests(unittest.TestCase):
    def test_header_rejects_wrong_magic_dimensions_length_and_timestamp(self):
        header = viewer.DEPTH_HEADER.pack(b"D16F", 640, 480, 42, 123456, 640 * 480 * 2)
        self.assertEqual(viewer.unpack_depth_header(header), (640, 480, 42, 123456, 614400))
        invalid = [header[:-1], viewer.DEPTH_HEADER.pack(b"BAD!", 640, 480, 1, 1, 614400),
                   viewer.DEPTH_HEADER.pack(b"D16F", 99999, 480, 1, 1, 614400),
                   viewer.DEPTH_HEADER.pack(b"D16F", 640, 480, 1, 1, 8),
                   viewer.DEPTH_HEADER.pack(b"D16F", 640, 480, 1, 0, 614400)]
        for raw in invalid:
            with self.subTest(raw=raw[:16]), self.assertRaises(ValueError):
                viewer.unpack_depth_header(raw)

    def test_pipe_read_exact_handles_split_reads_eof_timeout_and_stop(self):
        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd, "rb", buffering=0) as stream:
            os.write(write_fd, b"abc")
            self.assertEqual(viewer.read_exact(stream, 3, threading.Event()), b"abc")
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                viewer.read_exact(stream, 1, threading.Event(), timeout=0.01)
            stop = threading.Event()
            stop.set()
            with self.assertRaises(viewer.CaptureStopped):
                viewer.read_exact(stream, 1, stop)
            os.close(write_fd)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                viewer.read_exact(stream, 1, threading.Event())

    def test_depth_close_waits_for_own_process_and_reports_forced_kill(self):
        producer = viewer.DepthProducer("unused")
        process = mock.Mock()
        process.poll.return_value = None
        producer.process = process
        self.assertIsNone(producer.close())
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=4)
        process.kill.assert_not_called()
        producer = viewer.DepthProducer("unused")
        producer.process = process = mock.Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("helper", 4), 0]
        self.assertIn("restoration is unconfirmed", producer.close())
        process.kill.assert_called_once_with()

    def test_depth_cleanup_requires_restore_evidence_after_capture(self):
        self.assertIsNone(viewer.depth_cleanup_warning("streamoff=ok\nformat_restore=ok\n", True))
        self.assertIsNone(viewer.depth_cleanup_warning("open capture node: Device busy", False))
        self.assertIn("unconfirmed", viewer.depth_cleanup_warning("streamoff=ok", True))
        self.assertIn("unconfirmed", viewer.depth_cleanup_warning("capture_format width=640", False))
        self.assertIn("failure", viewer.depth_cleanup_warning("restore original format: Input/output error", False))

    def test_depth_color_direction_and_invalid_pixels(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("Optional runtime cv2/numpy are not installed locally")
        color, fraction = viewer.colorize_depth(np.array([[0, 300, 5000, 65535]], dtype=np.uint16))
        self.assertEqual(fraction, 0.5)
        self.assertTrue((color[0, 0] == 0).all())
        self.assertTrue((color[0, 3] == 0).all())
        self.assertGreater(int(color[0, 1, 0]), int(color[0, 1, 2]))  # Near: blue in BGR.
        self.assertGreater(int(color[0, 2, 2]), int(color[0, 2, 0]))  # Far: red.


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sysfs = self.root / "video4linux"
        self.sysfs.mkdir()
        self.usb = self.root / "1-3"
        self.usb.mkdir()
        for key, value in (("idVendor", "8086"), ("idProduct", "0b3a"), ("speed", "480"),
                           ("product", "Intel RealSense D435i"), ("serial", "TEST123")):
            (self.usb / key).write_text(value)
        for node in ("video0", "video1"):
            (self.sysfs / node).mkdir()
            (self.sysfs / node / "device").symlink_to(self.usb, target_is_directory=True)

    @staticmethod
    def probe(args, **kwargs):
        node = args[-1]
        return types.SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(
            {"device": node, "usable_depth": node == "/dev/video0", "z16": node == "/dev/video0"}))

    def test_discovery_requires_usb_identity_and_readonly_z16_probe(self):
        runner = mock.Mock(side_effect=self.probe)
        identity = viewer.discover_depth("/tmp/helper", sysfs_root=self.sysfs, runner=runner)
        self.assertEqual(identity["depth_device"], "/dev/video0")
        self.assertEqual(identity["serial"], "TEST123")
        self.assertEqual((identity["usb_vid"], identity["usb_pid"]), ("8086", "0b3a"))
        self.assertEqual((identity["usb_speed_mbps"], identity["usb_bus_path"]), (480, "1-3"))
        self.assertEqual(identity["usb_sysfs_path"], str(self.usb.resolve()))
        for call in runner.call_args_list:
            self.assertEqual(call.args[0][:2], ["/tmp/helper", "--probe"])
        (self.usb / "idVendor").write_text("0000")
        runner.reset_mock()
        with self.assertRaisesRegex(RuntimeError, "identifiable"):
            viewer.discover_depth("/tmp/helper", sysfs_root=self.sysfs, runner=runner)
        runner.assert_not_called()

    def test_discovery_rejects_ambiguity_wrong_reply_and_unusable_explicit_node(self):
        def both(args, **kwargs):
            return types.SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(
                {"device": args[-1], "usable_depth": True, "z16": True}))
        with self.assertRaisesRegex(RuntimeError, "found 2"):
            viewer.discover_depth("helper", sysfs_root=self.sysfs, runner=both)
        with self.assertRaisesRegex(RuntimeError, "found 0"):
            viewer.discover_depth("helper", "/dev/video1", self.sysfs, self.probe)
        reply = types.SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(
            {"device": "/dev/video99", "usable_depth": True, "z16": True}))
        with self.assertRaisesRegex(RuntimeError, "different device"):
            viewer.discover_depth("helper", "/dev/video0", self.sysfs, mock.Mock(return_value=reply))
        with self.assertRaises(ValueError):
            viewer.usb_identity("/tmp/fake-device", self.sysfs)

    def test_missing_or_empty_serial_has_actionable_enumeration_message(self):
        serial = self.usb / "serial"
        serial.unlink()
        with self.assertRaises(viewer.USBEnumerationError) as raised:
            viewer.usb_identity("/dev/video0", self.sysfs)
        message = str(raised.exception)
        self.assertIn("serial", message)
        self.assertIn("USB 3", message)
        self.assertNotIn("No such file", message)
        self.assertNotIn(str(self.root), message)
        serial.write_text("")
        with self.assertRaisesRegex(viewer.USBEnumerationError, "序列号为空"):
            viewer.usb_identity("/dev/video0", self.sysfs)

    def test_vanished_node_and_reenumeration_during_probe_are_explicit(self):
        link = self.sysfs / "video0" / "device"
        link.unlink()
        with self.assertRaisesRegex(viewer.USBEnumerationError, "节点已消失"):
            viewer.usb_identity("/dev/video0", self.sysfs)
        link.symlink_to(self.usb, target_is_directory=True)
        def remove_during_probe(args, **kwargs):
            link.unlink()
            return self.probe(args, **kwargs)
        with self.assertRaisesRegex(viewer.USBEnumerationError, "USB 枚举"):
            viewer.discover_depth("helper", "/dev/video0", self.sysfs, remove_during_probe, attempts=1)

    def test_usb3_reenumeration_recovers_and_still_selects_only_z16(self):
        (self.usb / "serial").unlink()
        stop = mock.Mock()
        stop.is_set.return_value = False
        def reenumerate(_delay):
            usb3 = self.root / "2-2.2"
            usb3.mkdir()
            for key, value in (("idVendor", "8086"), ("idProduct", "0b3a"), ("speed", "5000"),
                               ("product", "Intel RealSense D435i"), ("serial", "TEST_USB3")):
                (usb3 / key).write_text(value)
            for index in range(6):
                node = self.sysfs / ("video%d" % index)
                node.mkdir(exist_ok=True)
                link = node / "device"
                if link.is_symlink():
                    link.unlink()
                link.symlink_to(usb3, target_is_directory=True)
            return False
        stop.wait.side_effect = reenumerate
        runner = mock.Mock(side_effect=self.probe)
        identity = viewer.discover_depth("helper", sysfs_root=self.sysfs, runner=runner, stop=stop)
        self.assertEqual(identity["depth_device"], "/dev/video0")
        self.assertEqual((identity["serial"], identity["usb_speed_mbps"], identity["usb_bus_path"]),
                         ("TEST_USB3", 5000, "2-2.2"))
        self.assertEqual(identity["camera_name"], "Intel RealSense D435i")
        self.assertEqual(runner.call_count, 6)
        stop.wait.assert_called_once_with(0.4)

    def test_unreadable_candidate_cannot_make_another_node_look_unique(self):
        other = self.root / "2-2.2"
        other.mkdir()
        for key, value in (("idVendor", "8086"), ("product", "Intel RealSense D435i")):
            (other / key).write_text(value)
        link = self.sysfs / "video1" / "device"
        link.unlink()
        link.symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(viewer.USBEnumerationError, "serial"):
            viewer.discover_depth("helper", sysfs_root=self.sysfs, runner=self.probe, attempts=1)

    def test_enumeration_retry_is_cancellable_before_and_during_probe(self):
        stop = threading.Event()
        stop.set()
        runner = mock.Mock(side_effect=self.probe)
        with self.assertRaises(viewer.CaptureStopped):
            viewer.discover_depth("helper", sysfs_root=self.sysfs, runner=runner, stop=stop)
        runner.assert_not_called()
        stop.clear()
        def cancel_probe(args, **kwargs):
            stop.set()
            return self.probe(args, **kwargs)
        with self.assertRaises(viewer.CaptureStopped):
            viewer.discover_depth("helper", sysfs_root=self.sysfs, runner=cancel_probe, stop=stop)
        (self.usb / "serial").unlink()
        waiting = mock.Mock()
        waiting.is_set.return_value = False
        waiting.wait.return_value = True
        with self.assertRaises(viewer.CaptureStopped):
            viewer.discover_depth("helper", sysfs_root=self.sysfs, runner=self.probe, stop=waiting)
        waiting.wait.assert_called_once_with(0.4)

    def test_enumeration_retry_budget_is_bounded_and_does_not_accept_another_model(self):
        clock = [100.0]
        stop = mock.Mock()
        stop.is_set.return_value = False
        def wait(delay):
            clock[0] += delay
            return False
        stop.wait.side_effect = wait
        def timeout_probe(args, **kwargs):
            self.assertLessEqual(kwargs["timeout"], 1)
            clock[0] += kwargs["timeout"]
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        with mock.patch.object(viewer.time, "monotonic", side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(viewer.USBEnumerationError, "超时"):
                viewer.discover_depth("helper", sysfs_root=self.sysfs, runner=timeout_probe, stop=stop)
        self.assertLessEqual(clock[0], 105.001)
        self.assertLessEqual(stop.wait.call_count, 2)
        (self.usb / "product").write_text("Intel RealSense D455")
        runner = mock.Mock(side_effect=self.probe)
        with self.assertRaisesRegex(RuntimeError, "identifiable"):
            viewer.discover_depth("helper", sysfs_root=self.sysfs, runner=runner)
        runner.assert_not_called()

    def test_cli_limits_are_bounded(self):
        self.assertEqual(viewer.parse_args([]).max_session_seconds, 900)
        self.assertEqual(viewer.parse_args(["--max-session-seconds", "15"]).max_session_seconds, 15)
        self.assertEqual(viewer.parse_args([]).lidar_ip, "192.168.123.120")
        for argv in (["--max-session-seconds", "901"], ["--port", "80"], ["--depth-device", "/tmp/file"],
                     ["--lidar-ip", "not-an-ip"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                viewer.parse_args(argv)


class RGBAndProcessTests(unittest.TestCase):
    def test_rgb_client_and_channel_are_reused_across_sessions(self):
        stop = threading.Event()
        client = mock.Mock()
        def image_sample():
            stop.set()
            return 0, JPEG
        client.GetImageSample.side_effect = image_sample
        client_type = mock.Mock(return_value=client)
        channel_init = mock.Mock()
        modules = {name: types.ModuleType(name) for name in (
            "cv2", "numpy", "unitree_sdk2py", "unitree_sdk2py.core", "unitree_sdk2py.core.channel",
            "unitree_sdk2py.go2", "unitree_sdk2py.go2.video", "unitree_sdk2py.go2.video.video_client",
            "video_service")}
        modules["unitree_sdk2py.core.channel"].ChannelFactoryInitialize = channel_init
        modules["unitree_sdk2py.go2.video.video_client"].VideoClient = client_type
        def lease():
            service = mock.Mock()
            service.start.side_effect = lambda _stop, identity_check: identity_check()
            service.status.return_value = {"phase": "running"}
            service.close.return_value = None
            return service
        modules["video_service"].VideoServiceLease = mock.Mock(side_effect=lease)
        modules["video_service"].configured_camera_identity = mock.Mock(
            return_value={"rgb_device": "/dev/video4", "rgb_serial": "TEST_RGB"})
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(viewer, "SDK_INTERFACE", None), \
                mock.patch.object(viewer.RGBProducer, "_client", None):
            for _ in range(3):
                stop.clear()
                producer = viewer.RGBProducer("eth0")
                producer.run(mock.Mock(), stop, mock.Mock())
                producer.close()
            channel_init.assert_called_once_with(0, "eth0")
            client_type.assert_called_once_with()
            client.Init.assert_called_once_with()
            self.assertEqual(client.GetImageSample.call_count, 3)
            stop.clear()
            with self.assertRaisesRegex(RuntimeError, "interface cannot change"):
                viewer.RGBProducer("eth1").run(mock.Mock(), stop, mock.Mock())

    def test_rgb_and_lidar_initialize_dds_only_once_under_concurrency(self):
        channel = types.ModuleType("unitree_sdk2py.core.channel")
        channel.ChannelFactoryInitialize = mock.Mock(side_effect=lambda *_args: time.sleep(0.02))
        with mock.patch.dict(sys.modules, {"unitree_sdk2py.core.channel": channel}), \
                mock.patch.object(viewer, "SDK_INTERFACE", None):
            workers = [threading.Thread(target=viewer.ensure_sdk, args=("eth0",)) for _ in range(8)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(1)
            channel.ChannelFactoryInitialize.assert_called_once_with(0, "eth0")
            self.assertTrue(all(not worker.is_alive() for worker in workers))

    def test_lidar_factory_passes_shared_sdk_initializer_without_calling_it(self):
        module = types.ModuleType("lidar_preview")
        module.LidarProducer = mock.Mock()
        with mock.patch.dict(sys.modules, {"lidar_preview": module}), \
                mock.patch.object(viewer, "ensure_sdk") as initializer:
            viewer.lidar_factory("eth0", "192.168.123.120")
            module.LidarProducer.assert_called_once_with(
                "eth0", lidar_ip="192.168.123.120", ensure_sdk=initializer)
            initializer.assert_not_called()

    def test_server_main_exits_cleanly_on_hangup_without_loading_sdk(self):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        process = subprocess.Popen(
            [sys.executable, str(Path(viewer.__file__).resolve()), "--port", str(port),
             "--max-session-seconds", "15", "--instance-id", "signal-test"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertTrue(select.select([process.stdout], [], [], 3)[0], "Server did not become ready")
            ready = process.stdout.readline()
            self.assertEqual(json.loads(ready)["kind"], "ready")
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            try:
                connection.request("GET", "/api/health")
                response = connection.getresponse()
                self.assertEqual(json.loads(response.read()), {"ok": True, "instance_id": "signal-test"})
            finally:
                connection.close()
            process.send_signal(signal.SIGHUP)
            stdout, stderr = process.communicate(timeout=3)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertNotIn("unitree_sdk", stdout + stderr)
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=3)


if __name__ == "__main__":
    unittest.main()
