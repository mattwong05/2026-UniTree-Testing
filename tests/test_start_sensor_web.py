"""Launcher boundary tests; no network, credentials, or robot access."""
import argparse
import ast
import contextlib
import importlib.util
import io
from pathlib import Path
import shlex
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
        command = launcher.remote_server_command(directory, args)
        tokens = shlex.split(command.split("; exec ", 1)[1])
        self.assertEqual(tokens[:3], ["python3", "-u", directory + "/sensor_server.py"])
        self.assertEqual(tokens[tokens.index("--depth-binary") + 1], directory + "/depth_frame_check")
        self.assertEqual(tokens[tokens.index("--depth-device") + 1], "/dev/video2")
        self.assertEqual(tokens[tokens.index("--port") + 1], "8766")
        self.assertEqual(tokens[tokens.index("--instance-id") + 1], Path(directory).name)
        self.assertEqual(tokens[tokens.index("--lidar-ip") + 1], "192.168.123.120")

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
