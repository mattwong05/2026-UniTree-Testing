"""Head video ownership and configuration tests; no SDK, USB or robot access."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location(
    "video_service", Path(__file__).resolve().parents[1] / "web/video_service.py")
video = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video)

DIRECT = "/unitree/module/video_hub_pc4/videohub_pc4 /dev/video4"
WRAPPED = ("export CYCLONEDDS_URI=/unitree/module/video_hub_pc4/cyclonedds.xml;"
           "/unitree/sbin/start-stop-daemon --start --background --make-pidfile "
           "--pidfile=/unitree/var/run/videohub_pc4.pid --exec " + DIRECT)


class Client:
    def __init__(self, initial=1, start_code=0, restore=True):
        self.state, self.start_code, self.restore = initial, start_code, restore
        self.switches = []
        self.timeouts = []

    def SetTimeout(self, value):
        self.timeouts.append(value)

    def ServiceList(self):
        return 0, [SimpleNamespace(name="video_hub_pc4_chest", status=0),
                   SimpleNamespace(name=video.SERVICE, status=self.state)]

    def ServiceSwitch(self, name, enabled):
        self.switches.append((name, enabled))
        if enabled:
            self.state = 0
            if isinstance(self.start_code, Exception):
                raise self.start_code
            return self.start_code
        if self.restore:
            self.state = 1
        return 0


class LifecycleTests(unittest.TestCase):
    def make(self, client=None):
        lease = video.VideoServiceLease()
        client = client or Client()
        lease._make_client = mock.Mock(return_value=client)
        return lease, client

    def test_preexisting_running_service_is_never_switched(self):
        lease, client = self.make(Client(initial=0))
        verify = mock.Mock(side_effect=AssertionError("must not verify a start that is not needed"))
        self.assertEqual(lease.start(threading.Event(), verify), {})
        self.assertEqual(lease.status()["phase"], "running")
        self.assertIsNone(lease.close())
        self.assertEqual(client.switches, [])
        self.assertEqual(client.state, 0)
        self.assertEqual(lease.status()["phase"], "unchanged")
        verify.assert_not_called()

    def test_stopped_service_starts_after_verification_and_restores_once(self):
        lease, client = self.make()
        identity = {"rgb_device": "/dev/video4", "rgb_serial": "camera-02"}
        def verify():
            self.assertEqual(client.switches, [])
            return identity
        self.assertEqual(lease.start(threading.Event(), verify), identity)
        self.assertTrue(lease.status()["owned_start"])
        self.assertTrue(lease.status()["restore_required"])
        self.assertEqual(lease.status()["initial"], 1)
        self.assertEqual(lease.status()["phase"], "running")
        self.assertIsNone(lease.close())
        self.assertIsNone(lease.close())
        self.assertEqual(client.switches, [(video.SERVICE, True), (video.SERVICE, False)])
        self.assertEqual(client.state, 1)
        self.assertFalse(lease.status()["restore_required"])
        self.assertTrue(lease.status()["restore_confirmed"])
        self.assertEqual(lease.status()["phase"], "restored")

    def test_lost_start_ack_still_restores_an_applied_start(self):
        for reply in (99, TimeoutError("start reply lost")):
            with self.subTest(reply=reply):
                lease, client = self.make(Client(start_code=reply))
                with self.assertRaises((RuntimeError, TimeoutError)):
                    lease.start(threading.Event(), lambda: {})
                self.assertTrue(lease.status()["restore_required"])
                self.assertIsNone(lease.close())
                self.assertEqual(client.switches, [(video.SERVICE, True), (video.SERVICE, False)])
                self.assertTrue(lease.status()["restore_confirmed"])

    def test_failed_identity_or_service_state_never_switches(self):
        lease, client = self.make()
        with self.assertRaisesRegex(RuntimeError, "wrong RGB node"):
            lease.start(threading.Event(), mock.Mock(side_effect=RuntimeError("wrong RGB node")))
        self.assertIsNone(lease.close())
        self.assertEqual(client.switches, [])
        for response in ((1, []), (0, []),
                         (0, [SimpleNamespace(name=video.SERVICE, status=2)]),
                         (0, [SimpleNamespace(name=video.SERVICE, status=1)] * 2)):
            with self.subTest(response=response):
                lease, client = self.make()
                client.ServiceList = mock.Mock(return_value=response)
                verify = mock.Mock()
                with self.assertRaisesRegex(RuntimeError, "state is unavailable"):
                    lease.start(threading.Event(), verify)
                lease.close()
                verify.assert_not_called()
                self.assertEqual(client.switches, [])

    def test_stop_or_close_before_start_does_not_create_client(self):
        for close_first in (False, True):
            with self.subTest(close_first=close_first):
                lease, client = self.make()
                stop = threading.Event()
                if close_first:
                    lease.close()
                else:
                    stop.set()
                verify = mock.Mock()
                self.assertEqual(lease.start(stop, verify), {})
                lease.close()
                lease._make_client.assert_not_called()
                verify.assert_not_called()
                self.assertEqual(client.switches, [])

    def test_stop_during_verification_does_not_start_service(self):
        lease, client = self.make()
        stop = threading.Event()
        def verify():
            stop.set()
            return {"rgb_serial": "camera-02"}
        self.assertEqual(lease.start(stop, verify), {})
        self.assertIsNone(lease.close())
        self.assertEqual(client.switches, [])

    def test_stop_after_start_still_requires_restore(self):
        lease, client = self.make()
        stop = threading.Event()
        original = client.ServiceSwitch
        def switch(name, enabled):
            result = original(name, enabled)
            if enabled:
                stop.set()
            return result
        client.ServiceSwitch = switch
        lease.start(stop, lambda: {})
        self.assertTrue(lease.status()["restore_required"])
        self.assertIsNone(lease.close())
        self.assertEqual(client.switches, [(video.SERVICE, True), (video.SERVICE, False)])

    def test_close_racing_with_start_restores_exactly_once(self):
        lease, client = self.make()
        entered, release = threading.Event(), threading.Event()
        original = client.ServiceSwitch
        def switch(name, enabled):
            result = original(name, enabled)
            if enabled:
                entered.set()
                if not release.wait(2):
                    raise RuntimeError("test release timed out")
            return result
        client.ServiceSwitch = switch
        errors, cleanup = [], []
        def run():
            try:
                lease.start(threading.Event(), lambda: {})
            except Exception as error:
                errors.append(error)
        worker = threading.Thread(target=run)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            closer = threading.Thread(target=lambda: cleanup.append(lease.close()))
            closer.start()
            self.assertTrue(lease.closing.wait(1))
        finally:
            release.set()
            worker.join(2)
        closer.join(2)
        self.assertFalse(worker.is_alive())
        self.assertFalse(closer.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(cleanup, [None])
        self.assertEqual(client.switches, [(video.SERVICE, True), (video.SERVICE, False)])

    def test_restore_ack_without_stopped_state_leaves_persistent_warning(self):
        lease, client = self.make(Client(restore=False))
        lease.start(threading.Event(), lambda: {})
        clock = [10.0]
        with mock.patch.object(video.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(video.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)):
            warning = lease.close()
        self.assertIn("original stopped state was not confirmed", warning)
        self.assertEqual(lease.close(), warning)
        self.assertEqual(lease.status()["phase"], "restore_unconfirmed")
        self.assertTrue(lease.status()["restore_required"])
        self.assertFalse(lease.status()["restore_confirmed"])
        self.assertEqual(client.switches, [(video.SERVICE, True), (video.SERVICE, False)])
        self.assertLessEqual(clock[0] - 10.0, video.RESTORE_SECONDS + 0.001)

    def test_lost_stop_ack_is_success_only_when_stopped_state_is_observed(self):
        lease, client = self.make()
        lease.start(threading.Event(), lambda: {})
        original = client.ServiceSwitch
        def switch(name, enabled):
            original(name, enabled)
            raise TimeoutError("stop reply lost")
        client.ServiceSwitch = switch
        self.assertIsNone(lease.close())
        self.assertTrue(lease.status()["restore_confirmed"])


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "service.json"
        self.sysfs = self.root / "video4linux"
        self.interface = self.sysfs / "video4/device/bInterfaceNumber"
        self.interface.parent.mkdir(parents=True)
        self.interface.write_text("03\n")
        self.identify = mock.Mock(return_value={"usb_pid": "0b3a", "serial": "camera-02"})

    def read(self, command=WRAPPED):
        self.config.write_text(json.dumps({"Start": {"Cmd": command}}))
        return video.configured_camera_identity(self.identify, self.config, self.sysfs)

    def test_official_wrapper_and_direct_head_command_accept_verified_rgb(self):
        for command in (DIRECT, WRAPPED, WRAPPED + ";"):
            with self.subTest(command=command):
                identity = self.read(command)
                self.assertEqual(identity["rgb_device"], "/dev/video4")
                self.assertEqual(identity["rgb_serial"], "camera-02")
        self.assertEqual(self.identify.call_count, 6)
        self.identify.assert_called_with("/dev/video4")

    def test_unsupported_command_and_non_head_service_are_rejected(self):
        commands = [None, [], "echo " + DIRECT, "false;" + DIRECT,
                    DIRECT + " && echo done", DIRECT.replace("/dev/video4", "/dev/null"),
                    DIRECT.replace("videohub_pc4 ", "videohub_pc4_ch "),
                    WRAPPED.replace("--start", "--stop"),
                    WRAPPED.replace("videohub_pc4.pid", "videohub_pc4_ch.pid"),
                    WRAPPED.replace("--exec", "--exec /bin/echo --exec"),
                    '"unterminated']
        for command in commands:
            with self.subTest(command=command), self.assertRaisesRegex(RuntimeError, "unsupported"):
                self.read(command)
        self.identify.assert_not_called()

    def test_missing_or_malformed_configuration_is_rejected(self):
        for value in ("{", "null", "{}", '{"Start": {}}', '{"Start": null}'):
            self.config.write_text(value)
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "missing or invalid"):
                video.configured_camera_identity(self.identify, self.config, self.sysfs)
        self.config.unlink()
        with self.assertRaisesRegex(RuntimeError, "missing or invalid"):
            video.configured_camera_identity(self.identify, self.config, self.sysfs)
        self.identify.assert_not_called()

    def test_non_rgb_interface_wrong_pid_and_identity_changes_are_rejected(self):
        self.interface.write_text("00\n")
        with self.assertRaisesRegex(RuntimeError, "not the D435i RGB"):
            self.read()
        self.interface.write_text("03\n")
        self.identify.return_value = {"usb_pid": "0ad3", "serial": "other-camera"}
        with self.assertRaisesRegex(RuntimeError, "not the D435i RGB"):
            self.read()
        self.identify.side_effect = [{"usb_pid": "0b3a", "serial": "camera-02"},
                                     {"usb_pid": "0b3a", "serial": "changed-camera"}]
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.read()


if __name__ == "__main__":
    unittest.main()
