#!/usr/bin/env python3
"""Run the sensor page through SSH, without installing robot dependencies."""
import argparse
import atexit
import ipaddress
import json
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser


ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.123.164")
    parser.add_argument("--user", default="unitree")
    parser.add_argument("--name", help="device label; separate SSH host-key record for each robot")
    parser.add_argument("--port", type=int, default=8765, help="local browser port")
    parser.add_argument("--remote-port", type=int, default=8765)
    parser.add_argument("--depth-device", default="auto")
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--lidar-ip", default="192.168.123.120")
    parser.add_argument("--ssh-control-path", help="reuse an already authenticated SSH session")
    parser.add_argument("--no-open", action="store_true", help="do not launch the browser automatically")
    args = parser.parse_args(argv)
    try:
        address = ipaddress.IPv4Address(args.host)
    except ipaddress.AddressValueError:
        parser.error("--host must be the robot's IPv4 address")
    if address.is_multicast or address.is_unspecified or int(address) == 0xFFFFFFFF:
        parser.error("--host must be a unicast robot address")
    try:
        lidar = ipaddress.IPv4Address(args.lidar_ip)
    except ipaddress.AddressValueError:
        parser.error("--lidar-ip must be an IPv4 address")
    if lidar.is_multicast or lidar.is_unspecified or int(lidar) == 0xFFFFFFFF:
        parser.error("--lidar-ip must be a unicast sensor address")
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_-]{0,31}", args.user):
        parser.error("invalid SSH username")
    if args.name and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,39}", args.name):
        parser.error("--name accepts 1-40 letters, digits, underscores and hyphens")
    if not all(1024 <= port <= 65535 for port in (args.port, args.remote_port)):
        parser.error("ports must be 1024-65535")
    if args.depth_device != "auto" and not re.fullmatch(r"/dev/video[0-9]+", args.depth_device):
        parser.error("--depth-device must be auto or /dev/videoN")
    if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,32}", args.interface):
        parser.error("invalid DDS interface")
    if args.ssh_control_path and args.name:
        parser.error("--name cannot change the identity of an existing SSH control session")
    return args


def remote_server_command(remote_dir, args):
    command = ["python3", "-u", remote_dir + "/sensor_server.py",
               "--host-label", args.host, "--port", str(args.remote_port),
               "--instance-id", Path(remote_dir).name,
               "--depth-binary", remote_dir + "/depth_frame_check",
               "--depth-device", args.depth_device, "--interface", args.interface,
               "--lidar-ip", args.lidar_ip,
               "--web-root", remote_dir]
    # PID belongs to this shell and is preserved by exec. No unrelated PID is killed.
    return "umask 077; printf '%s\\n' \"$$\" > " + shlex.quote(remote_dir + "/server.pid") + "; exec " + shlex.join(command)


def cleanup_program(remote_dir):
    # The PID file is accepted only with the exact task's script in /proc/cmdline.
    return "\n".join([
        "import os,signal,time,pathlib",
        "root=pathlib.Path(" + repr(remote_dir) + ")",
        "try: pid=int((root/'server.pid').read_text())",
        "except (OSError,ValueError): raise SystemExit(0)",
        "def belongs():",
        "    try: args=pathlib.Path('/proc/%s/cmdline'%pid).read_bytes().split(bytes([0]))",
        "    except OSError: return False",
        "    return str(root/'sensor_server.py').encode() in args",
        "if pid>1 and belongs():",
        "    os.kill(pid,signal.SIGTERM)",
        "    for _ in range(400):",
        "        if not belongs(): break",
        "        time.sleep(.1)",
        "    if belongs():",
        "        print('Server shutdown is unconfirmed; inspect the robot before reconnecting.')",
        "        raise SystemExit(1)",
    ])


class Launcher:
    def __init__(self, args):
        self.args = args
        self.target = args.user + "@" + args.host
        self.temp = Path(tempfile.mkdtemp(prefix="g1-sensor-web-"))
        self.control = args.ssh_control_path or str(self.temp / "ssh.sock")
        self.owns_master = not bool(args.ssh_control_path)
        self.remote_dir = "/tmp/" + self.temp.name
        self.forward = "127.0.0.1:%d:127.0.0.1:%d" % (args.port, args.remote_port)
        self.process = None
        self.master_ready = False
        self.uploaded = False
        self.cleaned = False
        self.cleaning = False
        self.cleanup_failed = False
        self.options = ["-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5",
                        "-o", "ServerAliveCountMax=3", "-o", "ControlPath=" + self.control]
        if args.name:
            state = ROOT / ".sensor-web"
            state.mkdir(mode=0o700, exist_ok=True)
            self.options += ["-o", "HostKeyAlias=g1-sensors-" + args.name,
                             "-o", "UserKnownHostsFile=" + str(state / "known_hosts")]

    def run_ssh(self, remote_command, **kwargs):
        return subprocess.run(["ssh", *self.options, "-o", "BatchMode=yes",
                               self.target, remote_command], check=True, **kwargs)

    def start(self):
        for binary in ("ssh", "scp"):
            if not shutil.which(binary):
                raise RuntimeError("Required local command missing: " + binary)
        files = [ROOT / "web/sensor_server.py", ROOT / "web/index.html",
                 ROOT / "web/lidar_preview.py", ROOT / "scripts/depth_frame_check.c",
                 ROOT / "scripts/lidar_cloud_check.py", ROOT / "scripts/livox_readonly_probe.py"]
        for file in files:
            if not file.is_file():
                raise RuntimeError("Missing project file: " + str(file))
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", self.args.port))
            except OSError as error:
                raise RuntimeError("Local port %d is in use; choose --port." % self.args.port) from error
        print("连接 %s。首次连接需核对 SSH 指纹；密码只交给 SSH。" % self.target, flush=True)
        if self.owns_master:
            subprocess.run(["ssh", *self.options, "-o", "ControlPersist=no",
                            "-M", "-N", "-f", self.target], check=True)
        else:
            subprocess.run(["ssh", *self.options, "-O", "check", self.target], check=True)
        self.master_ready = True
        print("检查机载依赖并准备临时采集服务……", flush=True)
        identity_check = ("import os,getpass; "
                          "peer=os.environ.get('SSH_CONNECTION','').split(); "
                          "assert len(peer)==4 and peer[2]==%r and getpass.getuser()==%r, "
                          "'SSH session target differs from requested robot'; "
                          "import cv2,numpy,unitree_sdk2py; "
                          "print('Robot connection identity and existing dependencies verified')") % (self.args.host, self.args.user)
        self.run_ssh("python3 -c " + shlex.quote(identity_check) +
                     " && command -v gcc", timeout=15)
        self.run_ssh("umask 077; mkdir -p " + shlex.quote(self.remote_dir), timeout=12)
        self.uploaded = True
        subprocess.run(["scp", "-o", "ControlPath=" + self.control, "-o", "BatchMode=yes",
                        *(str(file) for file in files),
                        self.target + ":" + self.remote_dir + "/"], check=True, timeout=30)
        self.run_ssh(shlex.join(["gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Wpedantic",
                                 self.remote_dir + "/depth_frame_check.c", "-o",
                                 self.remote_dir + "/depth_frame_check"]), timeout=30)
        self.process = subprocess.Popen([
            "ssh", *self.options, "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
            "-L", self.forward, self.target, remote_server_command(self.remote_dir, self.args)],
            start_new_session=True)
        url = "http://127.0.0.1:%d" % self.args.port
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("The remote page service exited; see its error above.")
            try:
                with opener.open(url + "/api/health", timeout=1) as response:
                    health = json.load(response)
                if health.get("ok") and health.get("instance_id") == Path(self.remote_dir).name:
                    break
            except (OSError, ValueError, urllib.error.URLError):
                pass
            time.sleep(0.25)
        else:
            raise RuntimeError("Page service did not become ready within 20 seconds.")
        print("\n实时检查页面：%s\n点击网页“连接预览”开始；此终端按 Ctrl+C 完整退出。\n" % url, flush=True)
        if not self.args.no_open:
            webbrowser.open(url)
        return self.process.wait()

    def cleanup(self):
        if self.cleaned or self.cleaning:
            return
        self.cleaning = True
        previous_signals = {}
        try:
            # A second Ctrl+C must not interrupt cleanup after capture is signalled.
            for number in (signal.SIGINT, signal.SIGTERM):
                previous_signals[number] = signal.signal(number, signal.SIG_IGN)
            try:
                if self.master_ready and self.uploaded:
                    try:
                        self.run_ssh("python3 -c " + shlex.quote(cleanup_program(self.remote_dir)), timeout=45)
                    except (OSError, subprocess.SubprocessError, KeyboardInterrupt):
                        self.cleanup_failed = True
                        print("SSH 清理未确认；请核对机上采集和雷达驱动状态。", file=sys.stderr)
            finally:
                try:
                    if self.process and self.process.poll() is None:
                        try:
                            self.process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            self.cleanup_failed = True
                            self.process.terminate()
                            try:
                                self.process.wait(timeout=2)
                            except subprocess.TimeoutExpired:
                                self.process.kill()
                                self.process.wait(timeout=2)
                    if self.process and isinstance(self.process.returncode, int) and self.process.returncode != 0:
                        self.cleanup_failed = True
                except (OSError, subprocess.SubprocessError, KeyboardInterrupt):
                    self.cleanup_failed = True
                finally:
                    if self.master_ready:
                        command = ["ssh", *self.options]
                        if self.owns_master:
                            command += ["-O", "exit", self.target]
                        else:
                            command += ["-O", "cancel", "-L", self.forward, self.target]
                        try:
                            subprocess.run(command, timeout=5, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
                        except (OSError, subprocess.SubprocessError, KeyboardInterrupt):
                            pass
        finally:
            # Only our locally created temporary socket directory is removed.
            shutil.rmtree(self.temp, ignore_errors=True)
            self.cleaned, self.cleaning = True, False
            for number, handler in previous_signals.items():
                signal.signal(number, handler)


def main(argv=None):
    launcher = Launcher(parse_args(argv))
    atexit.register(launcher.cleanup)

    def interrupted(_number, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        result = launcher.start()
    except KeyboardInterrupt:
        print("\n正在停止本次采集并关闭 SSH 转发……", flush=True)
        result = 0
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print("启动失败：%s" % error, file=sys.stderr)
        result = 1
    finally:
        launcher.cleanup()
    return 1 if launcher.cleanup_failed else result


if __name__ == "__main__":
    sys.exit(main())
