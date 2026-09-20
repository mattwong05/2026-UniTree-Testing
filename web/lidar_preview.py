"""Bounded G1 MID360 preview; only the named lidar_driver service is switched.

RobotStateClient contract: unitreerobotics/unitree_sdk2_python,
unitree_sdk2py/go2/robot_state/robot_state_client.py. Service status 0 means
running, 1 means stopped; ServiceSwitch takes True to start, False to stop.
DDS is initialized by the caller's shared ensure_sdk callback, never here.
"""
from collections import deque
import datetime
import importlib.util
import ipaddress
import json
from pathlib import Path
import re
import socket
import threading
import time


TOPIC = "rt/utlidar/cloud_livox_mid360"
SERVICE = "lidar_driver"
MAX_POINTS = 300000
MAX_DISPLAY_POINTS = 6000
MAX_CLOUD_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 512 * 1024
NO_FRAME_SECONDS = 5.0
RESTORE_SECONDS = 20.0
_helpers = {}


def _helper(name):
    """Use the same checked-in helper locally and beside the uploaded server."""
    if name not in _helpers:
        here = Path(__file__).resolve().parent
        candidates = (here / (name + ".py"), here.parent / "scripts" / (name + ".py"))
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise RuntimeError("Required sensor helper is missing: " + name)
        spec = importlib.util.spec_from_file_location("sensor_preview_" + name, source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _helpers[name] = module
    return _helpers[name]


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def model_from_product_info(product_info):
    """Recognize an explicit model token, never a build/production date.

    MID360 protocol key 0x8001 documents the legacy "Mid-360 2021/12/01"
    format. Current hardware also returns "DevType:Mid-360 FmType:...".
    Other tokens and conflicting/repeated DevType fields remain unknown.
    """
    if not isinstance(product_info, str):
        return "unknown"
    text = product_info.strip()
    if not text or "\0" in text:
        return "unknown"
    model_names = {"mid-360": "MID360", "mid360": "MID360",
                   "mid-360s": "MID360s", "mid360s": "MID360s"}
    fields = re.findall(r"(?:^|\s)DevType\s*:", text, flags=re.IGNORECASE)
    if fields:
        match = re.match(r"DevType\s*:\s*(\S+)", text, flags=re.IGNORECASE)
        if len(fields) != 1 or match is None:
            return "unknown"
        token = match.group(1)
    else:
        token = text.split()[0]
    return model_names.get(token.casefold(), "unknown")


def read_hms(lidar_ip):
    """One allowlisted query from an ephemeral port; never claim a cached result.

    The existing helper validates CRC, command, sequence, exact source IP/port,
    and parameter lengths. No discovery broadcast, SDK initialization, binding
    to driver ports, configuration, or work-mode command is performed.
    """
    probe = _helper("livox_readonly_probe")
    keys = (0x8000, 0x8001, 0x8006, 0x8011)
    sequence = 1 + time.monotonic_ns() % 0xFFFF
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        connection.bind(("", 0))
        payload, _evidence = probe.exchange(connection, lidar_ip, probe.COMMAND_PORT,
                                            probe.QUERY, sequence, 0.45, keys)
    values = probe.parse_query(payload, keys)
    serial = values["serial_number"]
    if not serial:
        raise ValueError("HMS reply has no LiDAR serial number")
    product_info = values["product_info"]
    model = model_from_product_info(product_info)
    codes = [item["code"] for item in values["hms_nonzero"]]
    warnings = []
    for code in codes:
        # This meaning is documented by the MID360-specific HMS table. A topic
        # name, another model, or an unrecognized product must not select it.
        if model == "MID360" and code == "0x01040002":
            warnings.append("光学窗口脏污警告 0x01040002，可能影响点云可靠性")
        else:
            warnings.append("HMS 告警 " + code + "，需查该型号厂家说明")
    warning = "；".join(warnings) if warnings else "本次只读查询未发现非零 HMS 告警"
    if model == "unknown":
        warning += "；雷达型号未确认"
    return {"hms_checked": True, "hms_codes": codes, "serial_number": serial,
            "lidar_model": model, "product_info": product_info,
            "diagnostic_at": _utc_now(), "warning": warning}


def validate_message(message):
    count = message.width * message.height
    if not (0 < message.width <= MAX_POINTS and 0 < message.height <= MAX_POINTS and
            0 < count <= MAX_POINTS and 0 < message.point_step <= 256):
        raise ValueError("Point cloud dimensions exceed preview limits")
    if not (message.width * message.point_step <= message.row_step and
            message.row_step * message.height <= MAX_CLOUD_BYTES and
            len(message.data) == message.row_step * message.height):
        raise ValueError("Point cloud byte length or row stride is invalid")
    if len(message.fields) > 64:
        raise ValueError("Point cloud has too many fields")
    frame_id = message.header.frame_id
    if not isinstance(frame_id, str) or not frame_id or len(frame_id) > 128:
        raise ValueError("Point cloud frame_id is invalid")
    stamp = message.header.stamp
    if stamp.sec < 0 or not 0 <= stamp.nanosec < 1000000000:
        raise ValueError("Point cloud timestamp is invalid")
    timestamp = stamp.sec * 1000000000 + stamp.nanosec
    if timestamp <= 0:
        raise ValueError("Point cloud timestamp is zero")
    return timestamp


def cloud_payload(message):
    import numpy as np
    validate_message(message)
    xyz = _helper("lidar_cloud_check").extract_xyz(message)
    valid = np.isfinite(xyz).all(axis=1) & (xyz != 0).any(axis=1)
    usable = xyz[valid]
    if len(usable) > MAX_DISPLAY_POINTS:
        indices = np.linspace(0, len(usable) - 1, MAX_DISPLAY_POINTS, dtype=np.int64)
        usable = usable[indices]
    return {"points": usable.tolist(), "point_count": int(len(xyz)),
            "valid_points": int(valid.sum()), "display_points": int(len(usable)),
            "frame_id": message.header.frame_id}


class LidarProducer:
    # SDK RPC clients have no public Close; reuse one response channel per server
    # process. The server permits only one active LiDAR session at a time.
    _client_lock = threading.Lock()
    _shared_client = None
    _shared_interface = None

    def __init__(self, interface, lidar_ip="192.168.123.120", ensure_sdk=None):
        if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,32}", interface):
            raise ValueError("Invalid DDS interface")
        address = ipaddress.IPv4Address(lidar_ip)
        if (address.is_multicast or address.is_unspecified or address.is_loopback or
                int(address) == 0xFFFFFFFF):
            raise ValueError("LiDAR IP must be a unicast LAN address")
        if not callable(ensure_sdk):
            raise ValueError("A shared ensure_sdk(interface) callback is required")
        self.interface, self.lidar_ip, self.ensure_sdk = interface, str(address), ensure_sdk
        self._lifecycle = threading.RLock()
        self._data_lock = threading.Lock()
        self._closing = threading.Event()
        self._new_frame = threading.Event()
        self._closed = False
        self._client = self._reader = self._diagnostics = None
        self._metadata = None
        self._cleanup_warning = None
        self._latest = self._error = None
        self._previous_timestamp = None
        self._last_received = None
        self._frames = self._timestamp_errors = self._decode_errors = 0
        self._receive_times = deque(maxlen=100)
        self._serial = None
        self._hms = {"hms_checked": False, "hms_codes": [], "serial_number": None,
                     "lidar_model": "unknown", "product_info": None,
                     "diagnostic_at": None, "warning": "当前 HMS 未检查"}
        self._service = {"name": SERVICE, "initial": None, "current": None,
                         "owned_start": False, "restore_required": False,
                         "restore_confirmed": False, "phase": "checking"}

    def _make_client(self):
        from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
        with self._client_lock:
            cls = self.__class__
            if cls._shared_client is None:
                client = RobotStateClient()
                client.Init()
                cls._shared_client, cls._shared_interface = client, self.interface
            elif cls._shared_interface != self.interface:
                raise RuntimeError("Shared LiDAR RPC client belongs to another DDS interface")
            cls._shared_client.SetTimeout(1.0)
            return cls._shared_client

    def _open_reader(self):
        from cyclonedds.qos import Qos, Policy
        from unitree_sdk2py.core.channel import ChannelFactory
        from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
        channel = ChannelFactory().CreateChannel(TOPIC, PointCloud2_)
        try:
            channel.SetReader(Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(10)),
                              self._on_sample, 0)
        except Exception:
            channel.CloseReader()
            raise
        return channel

    def _read_service(self):
        code, services = self._client.ServiceList()
        if code != 0 or services is None:
            raise RuntimeError("ServiceList failed (code %s)" % code)
        matches = [service for service in services if service.name == SERVICE]
        if len(matches) != 1 or matches[0].status not in (0, 1):
            raise RuntimeError("lidar_driver service state is missing, duplicated, or unsupported")
        self._service["current"] = int(matches[0].status)
        return self._service["current"]

    def _on_sample(self, message):
        if self._closing.is_set():
            return
        info = getattr(message, "sample_info", None)
        if info is not None and not info.valid_data:
            return
        now = time.monotonic()
        with self._data_lock:
            self._frames += 1
            self._receive_times.append(now)
            try:
                timestamp = validate_message(message)
                if self._previous_timestamp is not None and timestamp <= self._previous_timestamp:
                    self._timestamp_errors += 1
                    raise ValueError("Point cloud timestamp did not advance")
                self._previous_timestamp = timestamp
                self._last_received = now
                # One reference only; replace a pending frame with the newest.
                self._latest = (message, timestamp)
            except (AttributeError, TypeError, ValueError) as error:
                self._error = str(error)
            self._new_frame.set()

    def _diagnostic_loop(self):
        while not self._closing.is_set():
            try:
                result = read_hms(self.lidar_ip)
            except Exception as error:
                result = {"hms_checked": False, "hms_codes": [], "serial_number": None,
                          "lidar_model": "unknown", "product_info": None,
                          "diagnostic_at": None,
                          "warning": "当前 HMS 未检查：" + str(error)[:200]}
            with self._data_lock:
                serial = result["serial_number"]
                if serial and self._serial and serial != self._serial:
                    self._error = "LiDAR serial changed during this preview; reconnect and verify identity"
                    self._new_frame.set()
                if serial:
                    self._serial = serial
                self._hms = result
            if self._closing.wait(2.0):
                break

    def _status(self):
        with self._lifecycle:
            service = dict(self._service)
        with self._data_lock:
            times = list(self._receive_times)
            result = dict(self._hms)
            result.update(received_frames=self._frames, timestamp_errors=self._timestamp_errors,
                          decode_errors=self._decode_errors,
                          received_hz=round((len(times) - 1) / (times[-1] - times[0]), 3)
                          if len(times) > 1 and times[-1] > times[0] else 0)
        result.update(lidar_ip=self.lidar_ip, lidar_serial=result["serial_number"],
                      service_state=service)
        return result

    def _emit_metadata(self):
        if self._metadata:
            self._metadata(self._status())

    def run(self, publish, stop, metadata):
        self._metadata = metadata
        try:
            if self._closing.is_set() or stop.is_set():
                return
            self._emit_metadata()
            self.ensure_sdk(self.interface)
            with self._lifecycle:
                if self._closing.is_set() or stop.is_set():
                    return
                self._client = self._make_client()
                self._service["initial"] = self._read_service()
                if self._closing.is_set() or stop.is_set():
                    return
                self._reader = self._open_reader()
                if self._closing.is_set() or stop.is_set():
                    return
                if self._service["initial"] == 1:
                    # Mark before RPC: a lost ACK does not prove start was unapplied.
                    self._service.update(owned_start=True, restore_required=True, phase="starting")
                    code = self._client.ServiceSwitch(SERVICE, True)
                    if code != 0:
                        raise RuntimeError("Starting lidar_driver failed or is unconfirmed (code %s)" % code)
                self._diagnostics = threading.Thread(target=self._diagnostic_loop,
                                                     name="lidar-hms", daemon=True)
                self._diagnostics.start()
            deadline = time.monotonic() + 6.0
            while not self._closing.is_set() and not stop.is_set():
                with self._lifecycle:
                    if self._closing.is_set():
                        return
                    if self._read_service() == 0:
                        self._service["phase"] = "running"
                        break
                if time.monotonic() >= deadline:
                    raise RuntimeError("lidar_driver start was not confirmed within 6 seconds")
                stop.wait(0.1)
            if self._closing.is_set() or stop.is_set():
                return
            self._emit_metadata()
            waiting_since = time.monotonic()
            while not stop.is_set() and not self._closing.is_set():
                self._new_frame.wait(0.1)
                with self._data_lock:
                    self._new_frame.clear()
                    error, latest = self._error, self._latest
                    self._latest = None
                    last = self._last_received
                if error:
                    raise RuntimeError(error)
                if time.monotonic() - (last if last is not None else waiting_since) > NO_FRAME_SECONDS:
                    raise RuntimeError("No new valid LiDAR point cloud for more than 5 seconds")
                if latest is None:
                    self._emit_metadata()
                    continue
                if stop.is_set() or self._closing.is_set():
                    continue
                message, timestamp = latest
                try:
                    cloud = cloud_payload(message)
                    cloud.update(self._status(), source_timestamp_ns=timestamp)
                    if not cloud["display_points"]:
                        cloud["warning"] += "；本帧没有有限且非零的 XYZ 点"
                    if len(json.dumps(cloud, allow_nan=False, separators=(",", ":")).encode("utf-8")) > MAX_JSON_BYTES:
                        raise ValueError("Point cloud preview JSON exceeds size limit")
                except (AttributeError, TypeError, ValueError) as error:
                    with self._data_lock:
                        self._decode_errors += 1
                    raise RuntimeError("Point cloud decode failed: " + str(error)) from error
                self._emit_metadata()
                if not stop.is_set() and not self._closing.is_set():
                    publish(cloud, 0, 0)
        finally:
            self.close()

    def close(self):
        self._closing.set()
        self._new_frame.set()
        with self._lifecycle:
            if self._closed:
                return self._cleanup_warning
            warnings = []
            if self._reader:
                try:
                    self._reader.CloseReader()
                except Exception as error:
                    warnings.append("LiDAR subscriber cleanup failed: " + str(error))
                self._reader = None
            if self._service["restore_required"]:
                self._service["phase"] = "restoring"
                deadline = time.monotonic() + RESTORE_SECONDS
                last_error = None
                try:
                    # SDK _CallBase can wait once sending and once for a reply.
                    self._client.SetTimeout(min(0.5, max(0.001, (deadline - time.monotonic()) / 2)))
                    code = self._client.ServiceSwitch(SERVICE, False)
                    if code != 0:
                        last_error = "stop request code %s" % code
                except Exception as error:
                    last_error = str(error)
                while time.monotonic() < deadline:
                    try:
                        self._client.SetTimeout(min(0.5, max(0.001, (deadline - time.monotonic()) / 2)))
                        if self._read_service() == 1:
                            self._service.update(restore_confirmed=True, restore_required=False, phase="restored")
                            break
                    except Exception as error:
                        last_error = str(error)
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        time.sleep(min(0.1, remaining))
                if not self._service["restore_confirmed"]:
                    self._service["phase"] = "restore_unconfirmed"
                    warnings.append("lidar_driver original stopped state was not confirmed within %g seconds" %
                                    RESTORE_SECONDS +
                                    (": " + last_error if last_error else ""))
            else:
                self._service["phase"] = "unchanged"
            self._cleanup_warning = "; ".join(warnings) if warnings else None
            self._closed = True
        if self._diagnostics and self._diagnostics is not threading.current_thread():
            self._diagnostics.join(timeout=0.7)
        try:
            self._emit_metadata()
        except Exception:
            # Cleanup truth is independent of the UI consumer still being alive.
            pass
        return self._cleanup_warning
