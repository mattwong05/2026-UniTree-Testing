"""Bounded ownership of the manufacturer's head video service, never chest/motion."""
import json
from pathlib import Path
import re
import shlex
import threading
import time


SERVICE = "video_hub_pc4"
RESTORE_SECONDS = 12.0


def configured_rgb_node(command):
    """Accept only the known direct command or manufacturer's daemon wrapper."""
    if not isinstance(command, str):
        raise RuntimeError("Head video service has an unsupported camera configuration")
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError:
        raise RuntimeError("Head video service has an unsupported camera configuration") from None
    if tokens[:3] == ["export", "CYCLONEDDS_URI=/unitree/module/video_hub_pc4/cyclonedds.xml", ";"]:
        tokens = tokens[3:]
    if tokens and tokens[-1] == ";":
        tokens = tokens[:-1]
    if tokens and tokens[0] == "/unitree/sbin/start-stop-daemon":
        expected = {"--start", "--background", "--make-pidfile",
                    "--pidfile=/unitree/var/run/videohub_pc4.pid"}
        if len(tokens) != 8 or set(tokens[1:5]) != expected or tokens[5] != "--exec":
            raise RuntimeError("Head video service has an unsupported camera configuration")
        tokens = tokens[6:]
    if (len(tokens) != 2 or tokens[0] != "/unitree/module/video_hub_pc4/videohub_pc4" or
            not re.fullmatch(r"/dev/video[0-9]+", tokens[1])):
        raise RuntimeError("Head video service has an unsupported camera configuration")
    return tokens[1]


def configured_camera_identity(identify, config=Path("/unitree/etc/master_service/service/video_hub_pc4"),
                               sysfs=Path("/sys/class/video4linux")):
    """Validate the configured head RGB node before starting a stopped service."""
    try:
        command = json.loads(config.read_text())["Start"]["Cmd"]
    except (OSError, ValueError, KeyError, TypeError):
        raise RuntimeError("Head video service camera configuration is missing or invalid") from None
    device = configured_rgb_node(command)
    identity = identify(device)
    interface = (sysfs / Path(device).name / "device" / "bInterfaceNumber").read_text().strip()
    if identity.get("usb_pid") != "0b3a" or interface.lower() != "03":
        raise RuntimeError("Configured head video node is not the D435i RGB interface")
    if identify(device) != identity:
        raise RuntimeError("Configured head video camera identity changed during verification")
    return {"rgb_device": device, "rgb_serial": identity["serial"],
            "rgb_source": "head video RPC (configured D435i RGB node verified)"}


class VideoServiceLease:
    _client_lock = threading.Lock()
    _shared_client = None

    def __init__(self):
        self.lock = threading.RLock()
        self.closing = threading.Event()
        self.closed = False
        self.client = None
        self.warning = None
        self.state = dict(name=SERVICE, initial=None, current=None, owned_start=False,
                          restore_required=False, restore_confirmed=False, phase="checking")

    def _make_client(self):
        from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
        with self._client_lock:
            if self.__class__._shared_client is None:
                client = RobotStateClient()
                client.Init()
                self.__class__._shared_client = client
            return self.__class__._shared_client

    def status(self):
        with self.lock:
            return dict(self.state)

    def _read(self):
        code, services = self.client.ServiceList()
        matches = [item for item in (services or []) if item.name == SERVICE]
        if code != 0 or len(matches) != 1 or matches[0].status not in (0, 1):
            raise RuntimeError("Head video service state is unavailable or unsupported")
        self.state["current"] = int(matches[0].status)
        return self.state["current"]

    def start(self, stop, verify_start):
        with self.lock:
            if self.closing.is_set() or stop.is_set():
                return {}
            self.client = self._make_client()
            self.client.SetTimeout(0.5)
            self.state["initial"] = self._read()
            identity = {}
            if self.state["initial"] == 1:
                identity = verify_start()
                if self.closing.is_set() or stop.is_set():
                    return {}
                # Lost start ACK still requires restoration of the original state.
                self.state.update(owned_start=True, restore_required=True, phase="starting")
                code = self.client.ServiceSwitch(SERVICE, True)
                if code != 0:
                    raise RuntimeError("Head video start failed or is unconfirmed (code %s)" % code)
        deadline = time.monotonic() + 6
        while not stop.is_set() and not self.closing.is_set():
            with self.lock:
                if self.closing.is_set():
                    return identity
                if self._read() == 0:
                    self.state["phase"] = "running"
                    return identity
            if time.monotonic() >= deadline:
                raise RuntimeError("Head video service start was not confirmed within 6 seconds")
            stop.wait(0.1)
        return identity

    def close(self):
        self.closing.set()
        with self.lock:
            if self.closed:
                return self.warning
            if self.state["restore_required"]:
                self.state["phase"] = "restoring"
                deadline = time.monotonic() + RESTORE_SECONDS
                last_error = None
                try:
                    self.client.SetTimeout(0.3)
                    code = self.client.ServiceSwitch(SERVICE, False)
                    if code != 0:
                        last_error = "stop request code %s" % code
                except Exception as error:
                    last_error = str(error)
                while time.monotonic() < deadline:
                    try:
                        self.client.SetTimeout(min(0.3, max(0.001, (deadline - time.monotonic()) / 2)))
                        if self._read() == 1:
                            self.state.update(restore_required=False, restore_confirmed=True, phase="restored")
                            break
                    except Exception as error:
                        last_error = str(error)
                    time.sleep(min(0.1, max(0, deadline - time.monotonic())))
                if not self.state["restore_confirmed"]:
                    self.state["phase"] = "restore_unconfirmed"
                    self.warning = "Head video service original stopped state was not confirmed" + (
                        ": " + last_error if last_error else "")
            else:
                self.state["phase"] = "unchanged"
            self.closed = True
            return self.warning
