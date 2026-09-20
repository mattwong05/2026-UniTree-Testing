"""Offline LiDAR lifecycle, protocol, and cloud tests; never contact a device."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

spec = importlib.util.spec_from_file_location(
    "lidar_preview", Path(__file__).resolve().parents[1] / "web/lidar_preview.py")
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


def cloud(timestamp=1, points=((1, 2, 3),)):
    return SimpleNamespace(
        width=len(points), height=1, point_step=12, row_step=len(points) * 12,
        fields=[SimpleNamespace(name=name, offset=index * 4, count=1, datatype=7)
                for index, name in enumerate(("x", "y", "z"))],
        is_bigendian=False, data=b"".join(struct.pack("<3f", *point) for point in points),
        header=SimpleNamespace(frame_id="livox_frame",
                               stamp=SimpleNamespace(sec=timestamp, nanosec=0)))


def diagnostic(serial="current-lidar", codes=()):
    return {"hms_checked": True, "hms_codes": list(codes), "serial_number": serial,
            "lidar_model": "MID360", "product_info": "Mid-360 2021/12/01",
            "diagnostic_at": "2026-09-20T00:00:00+00:00", "warning": "current query only"}


class Client:
    def __init__(self, initial=1, start_code=0, restore=True):
        self.state, self.start_code, self.restore = initial, start_code, restore
        self.switches = []
        self.timeouts = []

    def SetTimeout(self, timeout):
        self.timeouts.append(timeout)

    def ServiceList(self):
        return 0, [SimpleNamespace(name="ai_sport", status=1),
                   SimpleNamespace(name="unitree_slam", status=1),
                   SimpleNamespace(name="lidar_driver", status=self.state)]

    def ServiceSwitch(self, name, enabled):
        self.switches.append((name, enabled))
        if enabled:
            self.state = 0
            return self.start_code
        if self.restore:
            self.state = 1
        return 0


class LifecycleTests(unittest.TestCase):
    def test_repeated_sessions_share_one_rpc_client_and_check_interface(self):
        module_name = "unitree_sdk2py.go2.robot_state.robot_state_client"
        module = ModuleType(module_name)
        module.RobotStateClient = mock.Mock()
        with mock.patch.dict(sys.modules, {module_name: module}), \
                mock.patch.object(preview.LidarProducer, "_shared_client", None), \
                mock.patch.object(preview.LidarProducer, "_shared_interface", None):
            first = preview.LidarProducer("eth0", ensure_sdk=mock.Mock())._make_client()
            second = preview.LidarProducer("eth0", ensure_sdk=mock.Mock())._make_client()
            self.assertIs(first, second)
            module.RobotStateClient.assert_called_once()
            first.Init.assert_called_once()
            self.assertEqual(first.SetTimeout.call_count, 2)
            with self.assertRaisesRegex(RuntimeError, "another DDS interface"):
                preview.LidarProducer("eth1", ensure_sdk=mock.Mock())._make_client()

    def make(self, client=None, sample=True):
        producer = preview.LidarProducer("eth0", ensure_sdk=mock.Mock())
        client = client or Client()
        reader = mock.Mock()
        producer._make_client = mock.Mock(return_value=client)
        def open_reader():
            if sample:
                producer._on_sample(cloud())
            return reader
        producer._open_reader = mock.Mock(side_effect=open_reader)
        return producer, client, reader

    def run_one(self, producer):
        stop = threading.Event()
        received, metadata = [], []
        def publish(value, width, height):
            received.append((value, width, height))
            stop.set()
        payload = {"points": [[1, 2, 3]], "point_count": 1, "valid_points": 1,
                   "display_points": 1, "frame_id": "livox_frame"}
        with mock.patch.object(preview, "read_hms", return_value=diagnostic()), \
                mock.patch.object(preview, "cloud_payload", return_value=payload):
            producer.run(publish, stop, metadata.append)
        return received, metadata

    def test_owned_driver_is_restored_and_close_is_idempotent(self):
        producer, client, reader = self.make()
        received, metadata = self.run_one(producer)
        producer.ensure_sdk.assert_called_once_with("eth0")
        self.assertEqual(client.switches, [("lidar_driver", True), ("lidar_driver", False)])
        self.assertEqual(client.state, 1)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0][1:], (0, 0))
        self.assertTrue(metadata[-1]["service_state"]["restore_confirmed"])
        self.assertEqual(metadata[-1]["service_state"]["phase"], "restored")
        self.assertIsNone(producer.close())
        reader.CloseReader.assert_called_once()
        self.assertEqual(len(client.switches), 2)

    def test_preexisting_running_service_is_never_switched(self):
        producer, client, _reader = self.make(Client(initial=0))
        _received, metadata = self.run_one(producer)
        self.assertEqual(client.switches, [])
        self.assertEqual(client.state, 0)
        self.assertFalse(metadata[-1]["service_state"]["owned_start"])
        self.assertEqual(metadata[-1]["service_state"]["phase"], "unchanged")

    def test_stop_or_close_before_run_has_no_sdk_or_service_side_effect(self):
        for close_first in (False, True):
            producer, client, _reader = self.make()
            stop = threading.Event()
            if close_first:
                producer.close()
            else:
                stop.set()
            producer.run(mock.Mock(), stop, mock.Mock())
            producer.ensure_sdk.assert_not_called()
            producer._make_client.assert_not_called()
            self.assertEqual(client.switches, [])

    def test_lost_start_ack_still_restores_an_applied_start(self):
        producer, client, reader = self.make(Client(start_code=99))
        with self.assertRaisesRegex(RuntimeError, "unconfirmed"):
            producer.run(mock.Mock(), threading.Event(), mock.Mock())
        self.assertEqual(client.switches, [("lidar_driver", True), ("lidar_driver", False)])
        self.assertEqual(client.state, 1)
        reader.CloseReader.assert_called_once()
        self.assertIsNone(producer.close())

    def test_close_racing_with_inflight_start_restores_exactly_once(self):
        producer, client, reader = self.make(sample=False)
        entered, release = threading.Event(), threading.Event()
        original_switch = client.ServiceSwitch
        def blocked_switch(name, enabled):
            result = original_switch(name, enabled)
            if enabled:
                entered.set()
                if not release.wait(2):
                    raise RuntimeError("test release timed out")
            return result
        client.ServiceSwitch = blocked_switch
        errors, cleanup = [], []
        def run():
            try:
                producer.run(mock.Mock(), threading.Event(), mock.Mock())
            except Exception as error:
                errors.append(error)
        with mock.patch.object(preview, "read_hms", return_value=diagnostic()):
            worker = threading.Thread(target=run)
            worker.start()
            self.assertTrue(entered.wait(1))
            closer = threading.Thread(target=lambda: cleanup.append(producer.close()))
            closer.start()
            self.assertTrue(producer._closing.wait(1))
            release.set()
            closer.join(2); worker.join(2)
        self.assertFalse(closer.is_alive())
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(cleanup, [None])
        self.assertEqual(client.switches, [("lidar_driver", True), ("lidar_driver", False)])
        self.assertEqual(client.state, 1)
        reader.CloseReader.assert_called_once()

    def test_restore_ack_without_stopped_status_is_a_persistent_warning(self):
        producer, client, _reader = self.make(Client(restore=False))
        with mock.patch.object(preview, "RESTORE_SECONDS", 0.025):
            _received, metadata = self.run_one(producer)
        warning = producer.close()
        self.assertIn("not confirmed", warning)
        self.assertEqual(metadata[-1]["service_state"]["phase"], "restore_unconfirmed")
        self.assertTrue(metadata[-1]["service_state"]["restore_required"])
        self.assertEqual(len(client.switches), 2)

    def test_delayed_service_shutdown_is_confirmed_beyond_six_seconds(self):
        producer, client, reader = self.make()
        producer._client, producer._reader = client, reader
        producer._service.update(initial=1, current=0, owned_start=True,
                                 restore_required=True, phase="running")
        client.state = 0
        clock = [100.0]
        pending_until = [None]
        def switch(name, enabled):
            client.switches.append((name, enabled))
            self.assertFalse(enabled)
            pending_until[0] = clock[0] + 7.5
            return 0
        def states():
            if clock[0] >= pending_until[0]:
                client.state = 1
            return 0, [SimpleNamespace(name="lidar_driver", status=client.state)]
        client.ServiceSwitch, client.ServiceList = switch, states
        with mock.patch.object(preview.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(preview.time, "sleep", side_effect=lambda delay: clock.__setitem__(0, clock[0] + delay)):
            warning = producer.close()
        self.assertIsNone(warning)
        self.assertGreaterEqual(clock[0] - 100, 7.5)
        self.assertLess(clock[0] - 100, preview.RESTORE_SECONDS)
        self.assertEqual(client.switches, [("lidar_driver", False)])
        self.assertTrue(producer._service["restore_confirmed"])
        self.assertEqual(producer._service["phase"], "restored")
        reader.CloseReader.assert_called_once()

    def test_restore_deadline_includes_rpc_waits_and_reports_configured_limit(self):
        producer, client, _reader = self.make(Client(restore=False))
        producer._client = client
        producer._service.update(initial=1, current=0, owned_start=True,
                                 restore_required=True, phase="running")
        client.state = 0
        clock = [100.0]
        original_states = client.ServiceList
        original_switch = client.ServiceSwitch
        def rpc_delay():
            # SDK may wait separately to send and receive, each with SetTimeout.
            clock[0] += min(0.8, 2 * client.timeouts[-1])
        def states():
            rpc_delay()
            return original_states()
        def switch(name, enabled):
            rpc_delay()
            return original_switch(name, enabled)
        client.ServiceList, client.ServiceSwitch = states, switch
        with mock.patch.object(preview.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(preview.time, "sleep", side_effect=lambda delay: clock.__setitem__(0, clock[0] + delay)):
            warning = producer.close()
        self.assertIn("within %g seconds" % preview.RESTORE_SECONDS, warning)
        self.assertGreaterEqual(clock[0] - 100, preview.RESTORE_SECONDS)
        self.assertLessEqual(clock[0] - 100, preview.RESTORE_SECONDS + 0.0021)
        self.assertEqual(client.switches, [("lidar_driver", False)])
        self.assertFalse(producer._service["restore_confirmed"])
        self.assertEqual(producer._service["phase"], "restore_unconfirmed")
        self.assertEqual(producer.close(), warning)
        self.assertEqual(len(client.switches), 1)

    def test_no_frame_and_oversized_json_fail_and_restore(self):
        producer, client, _reader = self.make(sample=False)
        with mock.patch.object(preview, "read_hms", return_value=diagnostic()), \
                mock.patch.object(preview, "NO_FRAME_SECONDS", 0.025):
            with self.assertRaisesRegex(RuntimeError, "No new valid"):
                producer.run(mock.Mock(), threading.Event(), mock.Mock())
        self.assertEqual(client.state, 1)
        producer, client, _reader = self.make()
        with mock.patch.object(preview, "MAX_JSON_BYTES", 10):
            with self.assertRaisesRegex(RuntimeError, "JSON exceeds"):
                self.run_one(producer)
        self.assertEqual(client.state, 1)

    def test_all_reception_stats_are_kept_while_only_latest_frame_is_cached(self):
        producer, _client, _reader = self.make()
        with mock.patch.object(preview.time, "monotonic", side_effect=[100 + i / 10 for i in range(20)]):
            for i in range(20):
                producer._on_sample(cloud(i + 1))
        status = producer._status()
        self.assertEqual(status["received_frames"], 20)
        self.assertEqual(status["received_hz"], 10)
        self.assertEqual(producer._latest[1], 20 * 1000000000)

    def test_duplicate_or_regressing_timestamp_is_rejected(self):
        for next_stamp in (2, 1):
            producer, _client, _reader = self.make()
            producer._on_sample(cloud(2))
            producer._on_sample(cloud(next_stamp))
            self.assertIn("did not advance", producer._error)
            self.assertEqual(producer._status()["timestamp_errors"], 1)


class DiagnosticTests(unittest.TestCase):
    def query_fixture(self, product_info, codes=(0x01040002,)):
        probe = preview._helper("livox_readonly_probe")
        serial = b"current-lidar"
        entries = [(0x8000, serial), (0x8001, product_info.encode("ascii")),
                   (0x8006, bytes([1])),
                   (0x8011, struct.pack("<8I", *(tuple(codes) + (0,) * (8 - len(codes)))))]
        payload = bytes([0]) + struct.pack("<H", len(entries)) + b"".join(
            struct.pack("<HH", key, len(value)) + value for key, value in entries)
        connection = mock.Mock()
        context = mock.MagicMock()
        context.__enter__.return_value = connection
        with mock.patch.object(preview.socket, "socket", return_value=context), \
                mock.patch.object(probe, "exchange", return_value=(payload, {})) as exchange:
            result = preview.read_hms("192.168.123.120")
        connection.bind.assert_called_once_with(("", 0))
        self.assertEqual(exchange.call_args.args[1:4], ("192.168.123.120", 56100, probe.QUERY))
        self.assertEqual(exchange.call_args.args[-1], (0x8000, 0x8001, 0x8006, 0x8011))
        return result

    def test_hms_uses_only_query_and_an_ephemeral_port(self):
        product_info = "DevType:Mid-360 FmType:App FmVer:13180240 BuildTime:2024/12/17"
        result = self.query_fixture(product_info)
        self.assertEqual(result["hms_codes"], ["0x01040002"])
        self.assertEqual(result["serial_number"], "current-lidar")
        self.assertEqual(result["lidar_model"], "MID360")
        self.assertEqual(result["product_info"], product_info)
        self.assertIn("脏污", result["warning"])

    def test_explicit_product_models_support_current_and_legacy_formats(self):
        cases = {
            "Mid-360 2021/12/01": "MID360",
            "MID360 2026/09/20": "MID360",
            "DevType:Mid-360 FmType:App FmVer:13180240 BuildTime:2024/12/17": "MID360",
            "DevType:Mid-360s FmType:App BuildTime:2021/01/01": "MID360s",
            "MID360s 2026/09/20": "MID360s",
            " DevType : MID360S ": "MID360s",
        }
        for value, expected in cases.items():
            with self.subTest(product_info=value):
                self.assertEqual(preview.model_from_product_info(value), expected)

    def test_dates_unknown_tokens_and_conflicting_fields_do_not_identify_models(self):
        for value in (None, "", "2026/09/20", "BuildTime:2024/12/17", "Mid-360-plus 2026/09/20",
                      "DevType:Unknown FmType:Mid-360", "Unknown Mid-360 2024/12/17",
                      "Mid-360 DevType:Mid-360s", "DevType:Mid-360 DevType:Mid-360s",
                      "DevType:Mid-360\0 FmType:App", "DevType:Mid-360s2"):
            with self.subTest(product_info=value):
                self.assertEqual(preview.model_from_product_info(value), "unknown")

    def test_unknown_model_preserves_raw_product_and_does_not_apply_mid360_hms_meaning(self):
        product_info = "DevType:Other FmType:App BuildTime:2024/12/17"
        result = self.query_fixture(product_info)
        self.assertEqual(result["lidar_model"], "unknown")
        self.assertEqual(result["product_info"], product_info)
        self.assertEqual(result["hms_codes"], ["0x01040002"])
        self.assertNotIn("脏污", result["warning"])
        self.assertIn("型号未确认", result["warning"])
        self.assertIn("0x01040002", result["warning"])

    def test_mid360s_keeps_its_distinct_model_without_an_unverified_hms_mapping(self):
        result = self.query_fixture("DevType:Mid-360s FmType:App")
        self.assertEqual(result["lidar_model"], "MID360s")
        self.assertNotIn("脏污", result["warning"])
        self.assertIn("0x01040002", result["warning"])

    def test_initial_and_refreshed_metadata_use_only_current_query_model(self):
        producer = preview.LidarProducer("eth0", ensure_sdk=mock.Mock())
        self.assertEqual(producer._status()["lidar_model"], "unknown")
        self.assertIsNone(producer._status()["product_info"])
        current = self.query_fixture("Mid-360 2021/12/01")
        with mock.patch.object(preview, "read_hms", return_value=current), \
                mock.patch.object(producer._closing, "wait", return_value=True):
            producer._diagnostic_loop()
        self.assertEqual(producer._status()["lidar_model"], "MID360")
        self.assertEqual(producer._status()["product_info"], "Mid-360 2021/12/01")

    def test_failed_fresh_query_does_not_reuse_previous_alarm(self):
        producer = preview.LidarProducer("eth0", ensure_sdk=mock.Mock())
        with mock.patch.object(preview, "read_hms", side_effect=[
                diagnostic(codes=["0x01040002"]), OSError("query timed out")]), \
                mock.patch.object(producer._closing, "wait", side_effect=[False, True]):
            producer._diagnostic_loop()
        self.assertFalse(producer._hms["hms_checked"])
        self.assertEqual(producer._hms["hms_codes"], [])
        self.assertIsNone(producer._hms["serial_number"])
        self.assertIsNone(producer._hms["diagnostic_at"])
        self.assertEqual(producer._status()["lidar_model"], "unknown")
        self.assertIsNone(producer._status()["product_info"])
        self.assertIn("当前 HMS 未检查", producer._hms["warning"])

    def test_serial_change_requires_a_new_verified_session(self):
        producer = preview.LidarProducer("eth0", ensure_sdk=mock.Mock())
        with mock.patch.object(preview, "read_hms", side_effect=[diagnostic("first"), diagnostic("second")]), \
                mock.patch.object(producer._closing, "wait", side_effect=[False, True]):
            producer._diagnostic_loop()
        self.assertIn("serial changed", producer._error)

    def test_payload_dimensions_and_timestamp_are_bounded_before_decode(self):
        for change in ({"width": 300001}, {"point_step": 257}, {"row_step": 0},
                       {"data": bytes(11)}):
            sample = cloud()
            for name, value in change.items():
                setattr(sample, name, value)
            with self.assertRaises(ValueError):
                preview.validate_message(sample)
        with self.assertRaises(ValueError):
            preview.validate_message(cloud(0))


@unittest.skipIf(np is None, "NumPy is required for PointCloud2 preview fixtures")
class CloudTests(unittest.TestCase):
    def test_nonfinite_and_zero_points_are_removed_then_downsampled_to_6000(self):
        points = [(float(i), 1, 2) for i in range(13000)]
        points += [(0, 0, 0), (float("nan"), 1, 2), (float("inf"), 1, 2)]
        result = preview.cloud_payload(cloud(points=points))
        self.assertEqual(result["point_count"], 13003)
        self.assertEqual(result["valid_points"], 13000)
        self.assertEqual(result["display_points"], 6000)
        self.assertEqual(result["points"][0], [0, 1, 2])
        self.assertEqual(result["points"][-1], [12999, 1, 2])
        self.assertLess(len(json.dumps(result, allow_nan=False).encode()), preview.MAX_JSON_BYTES)

    def test_all_invalid_points_produce_an_empty_cloud_not_nan_json(self):
        result = preview.cloud_payload(cloud(points=[(0, 0, 0), (float("nan"), 0, 0)]))
        self.assertEqual(result["points"], [])
        self.assertEqual(result["display_points"], 0)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
