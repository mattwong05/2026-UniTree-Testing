"""Launcher boundary tests; no network, credentials, or robot access."""
import argparse
import ast
import contextlib
import getpass
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "start_sensor_web", Path(__file__).resolve().parents[1] / "scripts/start_sensor_web.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_target_and_options_are_validated_before_connecting(self):
        bad = [
            ["--host", "-oProxyCommand=anything"], ["--host", "230.1.1.1"],
            ["--host", "0.0.0.0"], ["--host", "255.255.255.255"],
            ["--user", "unitree;id"], ["--name", "../next"],
            ["--port", "22"], ["--remote-port", "65536"],
            ["--depth-device", "/dev/null"], ["--interface", "eth0;id"],
            ["--lidar-ip", "230.1.1.1"], ["--lidar-ip", "not-an-ip"],
            ["--name", "g1-02", "--ssh-control-path", "/tmp/old-session"],
        ]
        for args in bad:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    launcher.parse_args(args)

    def test_remote_exec_preserves_quoted_script_arguments(self):
        args = launcher.parse_args(["--depth-device", "/dev/video2", "--remote-port", "8766"])
        directory = "/tmp/g1 sensor 'probe'"
        runtime = {"python": "/usr/bin/python3", "sdk_parent": "/home/unitree/SDK's copy"}
        command = launcher.remote_server_command(directory, args, runtime)
        tokens = shlex.split(command.split("; exec ", 1)[1])
        self.assertEqual(tokens[:3], [runtime["python"], "-u", directory + "/sensor_server.py"])
        self.assertEqual(tokens[tokens.index("--depth-binary") + 1], directory + "/depth_frame_check")
        self.assertEqual(tokens[tokens.index("--depth-device") + 1], "/dev/video2")
        self.assertEqual(tokens[tokens.index("--port") + 1], "8766")
        self.assertEqual(tokens[tokens.index("--instance-id") + 1], Path(directory).name)
        self.assertEqual(tokens[tokens.index("--lidar-ip") + 1], "192.168.123.120")

    def test_runtime_record_requires_concrete_absolute_paths(self):
        valid = {"python": "/usr/bin/python3", "sdk_parent": "/home/unitree/SDK's copy"}
        record = launcher.RUNTIME_MARKER + json.dumps(valid)
        self.assertEqual(launcher.parse_remote_runtime("import message\n" + record + "\n"), valid)
        invalid = [{}, [], {**valid, "python": "python3"},
                   {**valid, "sdk_parent": ""}, {**valid, "sdk_parent": "relative"},
                   {**valid, "sdk_parent": "/sdk:other"}, {**valid, "sdk_parent": "/sdk\nother"},
                   {**valid, "python": "/usr/bin/py\x00thon"}, {**valid, "sdk_parent": None}]
        for runtime in invalid:
            with self.subTest(runtime=runtime), self.assertRaises(RuntimeError):
                launcher.parse_remote_runtime(launcher.RUNTIME_MARKER + json.dumps(runtime))
        for output in ("", record + "\n" + record, launcher.RUNTIME_MARKER + "{"):
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                launcher.parse_remote_runtime(output)

    def test_probe_checks_real_capture_modules_without_starting_clients(self):
        source = launcher.dependency_probe_program("192.168.123.164", "unitree")
        ast.parse(source, feature_version=(3, 8))
        self.assertIn("unitree_sdk2py.core.channel import ChannelFactory,ChannelFactoryInitialize", source)
        self.assertIn("unitree_sdk2py.go2.video.video_client import VideoClient", source)
        self.assertIn("unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient", source)
        self.assertIn("unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_", source)
        self.assertIn("cyclonedds.qos import Qos,Policy", source)
        self.assertNotIn("ChannelFactoryInitialize(", source)
        self.assertNotIn("ServiceSwitch(", source)

    @staticmethod
    def make_dependency_fixture(root):
        modules = {
            "cv2.py": "", "numpy.py": "", "cyclonedds/qos.py": "Qos=Policy=object\n",
            "unitree_sdk2py/core/channel.py": "ChannelFactory=ChannelFactoryInitialize=object\n",
            "unitree_sdk2py/go2/video/video_client.py": "VideoClient=object\n",
            "unitree_sdk2py/go2/robot_state/robot_state_client.py": "RobotStateClient=object\n",
            "unitree_sdk2py/idl/sensor_msgs/msg/dds_.py": "PointCloud2_=object\n",
        }
        for name, body in modules.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
            parent = path.parent
            while parent != root:
                (parent / "__init__.py").touch()
                parent = parent.parent

    def test_cwd_only_sdk_is_preserved_for_absolute_server_script(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            home = root / "robot home's $SDK"
            self.make_dependency_fixture(home)
            remote = root / "temporary service's files"
            remote.mkdir()
            script = remote / "sensor_server.py"
            script.write_text("import json,os,sys,unitree_sdk2py\n"
                              "print(json.dumps({'sdk':unitree_sdk2py.__file__,"
                              "'python':sys.executable,'pythonpath':os.environ['PYTHONPATH']}))\n")
            env = dict(os.environ, SSH_CONNECTION="10.0.0.1 1234 192.168.123.164 22")
            env.pop("PYTHONPATH", None)
            env["PYTHONNOUSERSITE"] = "1"
            env.pop("PYTHONSAFEPATH", None)
            probe = subprocess.run([sys.executable, "-c", launcher.dependency_probe_program(
                "192.168.123.164", getpass.getuser())], cwd=home, env=env,
                capture_output=True, text=True, check=True)
            runtime = launcher.parse_remote_runtime(probe.stdout)
            self.assertEqual(runtime["sdk_parent"], str(home.resolve()))
            self.assertEqual(runtime["python"], sys.executable)
            original = subprocess.run([sys.executable, str(script)], cwd=home, env=env,
                                      capture_output=True, text=True)
            self.assertNotEqual(original.returncode, 0)
            self.assertIn("No module named 'unitree_sdk2py'", original.stderr)
            existing = root / "existing library path"
            env["PYTHONPATH"] = str(existing)
            command = launcher.remote_server_command(str(remote), launcher.parse_args([]), runtime)
            result = subprocess.run(["/bin/sh", "-c", command], cwd=home, env=env,
                                    capture_output=True, text=True, check=True)
            loaded = json.loads(result.stdout)
            self.assertEqual(Path(loaded["sdk"]).resolve(), (home / "unitree_sdk2py/__init__.py").resolve())
            self.assertEqual(loaded["python"], sys.executable)
            self.assertEqual(loaded["pythonpath"], str(home.resolve()) + ":" + str(existing))

    def test_probe_rejects_incomplete_sdk_before_service_starts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_dependency_fixture(root)
            (root / "unitree_sdk2py/go2/robot_state/robot_state_client.py").unlink()
            env = dict(os.environ, SSH_CONNECTION="10.0.0.1 1234 192.168.123.164 22")
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONSAFEPATH", None)
            result = subprocess.run([sys.executable, "-c", launcher.dependency_probe_program(
                "192.168.123.164", getpass.getuser())], cwd=root, env=env,
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("robot_state_client", result.stderr)
            self.assertNotIn(launcher.RUNTIME_MARKER, result.stdout)

    def test_cleanup_program_requires_exact_task_script_in_cmdline(self):
        source = launcher.cleanup_program("/tmp/g1-sensor-web-example")
        ast.parse(source, feature_version=(3, 8))
        self.assertIn("str(root/'sensor_server.py').encode() in args", source)
        self.assertIn("pid>1 and belongs()", source)
        self.assertNotIn("SIGKILL", source)
        self.assertNotIn("pkill", source)

    def test_existing_master_is_not_closed_by_cleanup(self):
        args = launcher.parse_args(["--ssh-control-path", "/tmp/existing-control"])
        instance = launcher.Launcher(args)
        instance.master_ready = True
        with mock.patch.object(launcher.subprocess, "run") as run:
            instance.cleanup()
        command = run.call_args.args[0]
        self.assertIn("cancel", command)
        self.assertNotIn("exit", command)
        self.assertFalse(instance.temp.exists())

    def test_owned_master_is_closed_and_cleanup_is_idempotent(self):
        instance = launcher.Launcher(launcher.parse_args([]))
        instance.master_ready = True
        with mock.patch.object(launcher.subprocess, "run") as run:
            instance.cleanup()
            instance.cleanup()
        run.assert_called_once()
        self.assertIn("exit", run.call_args.args[0])
        self.assertFalse(instance.temp.exists())

    def test_cleanup_continues_after_repeated_interrupt_and_process_exit_race(self):
        for remote_error, process_error in ((KeyboardInterrupt(), None),
                                             (None, ProcessLookupError())):
            instance = launcher.Launcher(launcher.parse_args([]))
            instance.master_ready = instance.uploaded = True
            instance.process = mock.Mock()
            instance.process.poll.return_value = None
            if process_error:
                instance.process.terminate.side_effect = process_error
                instance.process.wait.side_effect = launcher.subprocess.TimeoutExpired("ssh", 3)
            with mock.patch.object(instance, "run_ssh", side_effect=remote_error), \
                 mock.patch.object(launcher.subprocess, "run") as run, \
                 contextlib.redirect_stderr(io.StringIO()):
                instance.cleanup()
            run.assert_called_once()
            self.assertIn("exit", run.call_args.args[0])
            self.assertFalse(instance.temp.exists())

    def test_main_reports_cleanup_failure(self):
        instance = mock.Mock()
        instance.start.return_value = 0
        instance.cleanup_failed = True
        with mock.patch.object(launcher, "Launcher", return_value=instance), \
             mock.patch.object(launcher.atexit, "register"), \
             mock.patch.object(launcher.signal, "signal"):
            self.assertEqual(launcher.main([]), 1)
        instance.cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
