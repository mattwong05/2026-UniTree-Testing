"""Offline protocol fixtures; never connect to a sensor."""
import binascii
import contextlib
import importlib.util
import io
from pathlib import Path
import socket
import struct
import unittest
from unittest import mock
import zlib


spec = importlib.util.spec_from_file_location(
    "livox_probe", Path(__file__).resolve().parents[1] / "scripts/livox_readonly_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
SOURCE = ("192.168.123.120", 56100)


def response(payload, command=probe.QUERY, sequence=7, kind=1, sender=1):
    header = struct.pack("<BBHIHBB6s", 0xAA, 0, 24 + len(payload), sequence,
                         command, kind, sender, bytes(6))
    return header + struct.pack("<HI", binascii.crc_hqx(header, 0xFFFF),
                                zlib.crc32(payload)) + payload


def tlv(key, value):
    return struct.pack("<HH", key, len(value)) + value


class LivoxReadOnlyTests(unittest.TestCase):
    def parse(self, packet, source=SOURCE):
        return probe.parse_response(packet, source, SOURCE, probe.QUERY, 7)

    def test_official_crc_check_vectors(self):
        self.assertEqual(binascii.crc_hqx(b"123456789", 0xFFFF), 0x29B1)
        self.assertEqual(zlib.crc32(b"123456789"), 0xCBF43926)
        packet = probe.build_request(probe.DISCOVERY, 7)
        self.assertEqual(len(packet), 24)
        self.assertEqual(packet[20:], bytes(4))
        self.assertEqual(struct.unpack_from("<H", packet, 18)[0],
                         binascii.crc_hqx(packet[:18], 0xFFFF))

    def test_request_allowlist_and_query_layout(self):
        packet = probe.build_request(probe.QUERY, 7, (0x8000, 0x8006))
        self.assertEqual(packet[24:], bytes.fromhex("0200000000800680"))
        for command, keys in ((0x0100, (0x8006,)), (0x0200, ()),
                              (probe.DISCOVERY, (0x8006,)),
                              (probe.QUERY, ()), (probe.QUERY, (0xFFFF,)),
                              (probe.QUERY, (0x8006, 0x8006))):
            with self.subTest(command=command, keys=keys), self.assertRaises(ValueError):
                probe.build_request(command, 7, keys)

    def test_source_command_sequence_and_ack_validation(self):
        valid = response(b"\0\0\0")
        self.assertEqual(self.parse(valid), b"\0\0\0")
        for source in (("192.168.123.121", 56100), (SOURCE[0], 56000)):
            with self.assertRaisesRegex(probe.ProtocolError, "source"):
                self.parse(valid, source)
        for kw in ({"command": 0x0100}, {"sequence": 8}, {"kind": 0}, {"sender": 0}):
            with self.assertRaisesRegex(probe.ProtocolError, "match"):
                self.parse(response(b"\0\0\0", **kw))

    def test_corrupted_crc_and_length_are_rejected(self):
        packet = response(b"\0\0\0")
        for index in (12, 18, 20, 24):
            changed = bytearray(packet)
            changed[index] ^= 1
            with self.subTest(index=index), self.assertRaises(probe.ProtocolError):
                self.parse(changed)
        for changed in (packet[:23], packet[:-1], packet + b"\0"):
            with self.assertRaises(probe.ProtocolError):
                self.parse(changed)

    def test_error_return_and_empty_response(self):
        with self.assertRaisesRegex(probe.ProtocolError, "error code 3"):
            self.parse(response(b"\3"))
        with self.assertRaisesRegex(probe.ProtocolError, "missing"):
            self.parse(response(b""))

    def test_discovery_identity_and_endpoint_validation(self):
        body = struct.pack("<BB16s4sH", 0, 9, b"TEST_SERIAL", socket.inet_aton(SOURCE[0]), 56100)
        result = probe.parse_discovery(body, SOURCE[0])
        self.assertEqual(result["model"], "MID360")
        self.assertEqual(result["serial_number"], "TEST_SERIAL")
        for malformed in (body[:-1], body + b"\0", body[:1] + b"\1" + body[2:],
                          body[:-2] + struct.pack("<H", 56000)):
            with self.assertRaises(probe.ProtocolError):
                probe.parse_discovery(malformed, SOURCE[0])
        with self.assertRaisesRegex(probe.ProtocolError, "differs"):
            probe.parse_discovery(body, "192.168.123.121")

    def test_query_decodes_parameters_and_rejects_ambiguous_data(self):
        keys = (0x0006, 0x8006)
        destination = socket.inet_aton("192.168.123.164") + struct.pack("<HH", 56301, 56300)
        body = b"\0\2\0" + tlv(0x0006, destination) + tlv(0x8006, b"\1")
        result = probe.parse_query(body, keys)
        self.assertEqual(result["current_work_state_name"], "sampling")
        self.assertEqual(result["point_data_destination"]["host_port"], 56301)
        malformed = (body[:-1], body + b"\0", b"\0\1\0" + body[3:],
                     b"\0\2\0" + tlv(0x8006, b"\1") * 2,
                     b"\0\2\0" + tlv(0x8006, b"\1\2") + tlv(0x0006, destination),
                     b"\0\2\0" + tlv(0x8007, bytes(4)) + tlv(0x0006, destination))
        for candidate in malformed:
            with self.assertRaises(probe.ProtocolError):
                probe.parse_query(candidate, keys)

    def test_mid360_temperature_and_dirty_window_warning(self):
        body = (b"\0\2\0" + tlv(0x8007, struct.pack("<i", 6410)) +
                tlv(0x8011, struct.pack("<8I", 0x01040002, 0, 0, 0, 0, 0, 0, 0)))
        result = probe.parse_query(body, (0x8007, 0x8011))
        self.assertEqual(result["core_temperature_c"], 64.1)
        self.assertEqual(result["hms_nonzero"], [dict(
            code="0x01040002", abnormal_id="0x0104", level=2,
            meaning="window_dirty_point_cloud_reliability_warning")])

    def test_exchange_ignores_foreign_packet_but_sends_only_once(self):
        sock = mock.Mock()
        packet = response(b"\0\0\0")
        sock.recvfrom.side_effect = [(packet, ("192.168.123.121", 56100)), (packet, SOURCE)]
        payload, info = probe.exchange(sock, SOURCE[0], 56100, probe.QUERY, 7, 1, (0x8006,))
        self.assertEqual(payload, b"\0\0\0")
        self.assertEqual(info["rejected_datagrams"], 1)
        sock.sendto.assert_called_once()
        sock.recvfrom.side_effect = socket.timeout
        with self.assertRaises(probe.ProbeTimeout) as caught:
            probe.exchange(sock, SOURCE[0], 56100, probe.QUERY, 7, 1, (0x8006,))
        self.assertIn("request_hex", caught.exception.evidence)
        self.assertEqual(caught.exception.evidence["destination"], list(SOURCE))

    def test_broadcast_discovery_is_single_and_target_filtered(self):
        sender, receiver = mock.Mock(), mock.Mock()
        packet = response(b"\0", command=probe.DISCOVERY)
        receiver.recvfrom.side_effect = [(packet, ("192.168.123.121", 56000)),
                                        (packet, (SOURCE[0], 56000))]
        with mock.patch.object(probe.select, "select", return_value=([receiver], [], [])):
            _, info = probe.exchange(sender, SOURCE[0], 56000, probe.DISCOVERY, 7, 1,
                                     broadcast=True, receive_sockets=(sender, receiver))
        sender.sendto.assert_called_once_with(probe.build_request(probe.DISCOVERY, 7),
                                              ("255.255.255.255", 56000))
        self.assertEqual(info["rejected_datagrams"], 1)
        self.assertEqual(info["last_rejected_source"], ["192.168.123.121", 56000])
        with self.assertRaisesRegex(ValueError, "only for discovery"):
            probe.exchange(sender, SOURCE[0], 56100, probe.QUERY, 7, 1, (0x8006,), broadcast=True)

    def test_cli_rejects_broadcast_and_unbounded_timeout(self):
        base = ["--host-ip", "192.168.123.164", "--lidar-ip", SOURCE[0]]
        self.assertEqual(probe.parse_args(base).timeout, 3)
        for args in (base + ["--timeout", "nan"], base + ["--timeout", "11"],
                     base + ["--lidar-ip", "255.255.255.255"],
                     base + ["--lidar-ip", "230.1.1.1"],
                     base + ["--lidar-ip", "192.168.123.164"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                probe.parse_args(args)


if __name__ == "__main__":
    unittest.main()
