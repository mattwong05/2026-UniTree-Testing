#!/usr/bin/env python3
"""Bounded Livox SDK2 discovery and parameter reads; Python 3.8 stdlib only.

Sends only discovery (0x0000) and parameter query (0x0101). Does not use the
SDK initializer, whose normal driver path can configure data destinations.
Protocol sources in https://github.com/Livox-SDK/Livox-SDK2/tree/master:
  sdk_core/comm/{define.h,sdk_protocol.h,sdk_protocol.cpp}
  sdk_core/command_handler/{command_impl.cpp,general_command_handler.cpp}
  include/livox_lidar_def.h and 3rdparty/FastCRC/FastCRCsw.cpp
MID360 state/temperature/HMS definitions:
  https://livox-wiki-en.readthedocs.io/en/latest/tutorials/new_product/mid360/livox_eth_protocol_mid360.html
  https://livox-wiki-en.readthedocs.io/en/latest/tutorials/new_product/mid360/hms_code_mid360.html

Successful reads establish identity/state, not optical ranging performance.
"""
import argparse
import binascii
import contextlib
import ipaddress
import json
import math
import select
import socket
import struct
import sys
import time
import zlib


DISCOVERY = 0x0000
QUERY = 0x0101
DISCOVERY_PORT = 56000
COMMAND_PORT = 56100
MODELS = {9: "MID360", 35: "MID360s"}
# MID360-specific protocol states, not the SDK's cross-model enum labels.
WORK_STATES = {1: "sampling", 2: "idle", 4: "error",
               5: "self_check", 6: "motor_starting", 8: "upgrade", 9: "ready"}
PARAMETERS = {
    0x0006: "point_data_destination", 0x0007: "imu_data_destination",
    0x001A: "target_work_mode", 0x001C: "imu_data_enabled",
    0x8000: "serial_number", 0x8001: "product_info", 0x8002: "app_version",
    0x8005: "mac_address", 0x8006: "current_work_state",
    0x8007: "core_temperature_raw", 0x8008: "power_up_count",
    0x800E: "diagnostic_status_raw", 0x8011: "hms_codes_hex",
}
QUERY_KEYS = tuple(sorted(PARAMETERS))


class ProtocolError(ValueError):
    pass


class ProbeTimeout(TimeoutError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def build_request(command, sequence, keys=()):
    """The allowlist prevents this builder from producing control commands."""
    if not 0 <= sequence <= 0xFFFF:
        raise ValueError("sequence must fit the SDK's 16-bit sequence range")
    if command == DISCOVERY and not keys:
        payload = b""
    elif command == QUERY and keys and len(keys) == len(set(keys)):
        if any(key not in PARAMETERS for key in keys):
            raise ValueError("unsupported read-only query key")
        payload = struct.pack("<HH", len(keys), 0)
        payload += struct.pack("<" + "H" * len(keys), *keys)
    else:
        raise ValueError("only empty discovery and allowlisted queries are permitted")
    header = struct.pack("<BBHIHBB6s", 0xAA, 0, 24 + len(payload), sequence,
                         command, 0, 0, b"\0" * 6)
    return header + struct.pack("<HI", binascii.crc_hqx(header, 0xFFFF),
                                zlib.crc32(payload) & 0xFFFFFFFF) + payload


def parse_response(data, source, expected_source, command, sequence):
    if source != expected_source:
        raise ProtocolError("unexpected UDP source")
    if len(data) < 24:
        raise ProtocolError("truncated packet header")
    sof, version, length, seq, cmd, kind, sender = struct.unpack_from("<BBHIHBB", data)
    if sof != 0xAA or version != 0 or length != len(data):
        raise ProtocolError("invalid SOF, version, or packet length")
    if seq != sequence or cmd != command or kind != 1 or sender != 1:
        raise ProtocolError("response does not match requested command/sequence/ACK")
    crc16, crc32 = struct.unpack_from("<HI", data, 18)
    if crc16 != binascii.crc_hqx(data[:18], 0xFFFF):
        raise ProtocolError("header CRC16 mismatch")
    payload = data[24:]
    if crc32 != zlib.crc32(payload) & 0xFFFFFFFF:
        raise ProtocolError("payload CRC32 mismatch")
    if not payload:
        raise ProtocolError("missing response return code")
    if payload[0] != 0:
        raise ProtocolError("device returned error code %d" % payload[0])
    return payload


def parse_discovery(payload, expected_ip):
    if len(payload) != 24 or payload[0] != 0:
        raise ProtocolError("invalid discovery response")
    _, device_type, serial, ip_bytes, command_port = struct.unpack("<BB16s4sH", payload)
    address = socket.inet_ntoa(ip_bytes)
    if address != expected_ip:
        raise ProtocolError("advertised LiDAR IP differs from target")
    if device_type not in MODELS or command_port != COMMAND_PORT:
        raise ProtocolError("device is not a supported MID360/MID360s command endpoint")
    return {"model": MODELS[device_type], "device_type": device_type,
            "serial_number": serial.split(b"\0", 1)[0].decode("ascii", "replace"),
            "lidar_ip": address, "command_port": command_port}


def decode_parameter(key, value):
    expected_lengths = {0x0006: 8, 0x0007: 8, 0x001A: 1, 0x001C: 1,
                        0x8002: 4, 0x8005: 6, 0x8006: 1, 0x8007: 4,
                        0x8008: 4, 0x800E: 2, 0x8011: 32}
    if key in expected_lengths and len(value) != expected_lengths[key]:
        raise ProtocolError("unexpected value length for 0x%04x" % key)
    if key in (0x0006, 0x0007):
        host_port, lidar_port = struct.unpack_from("<HH", value, 4)
        return {"host_ip": socket.inet_ntoa(value[:4]), "host_port": host_port,
                "lidar_port": lidar_port}
    if key in (0x8000, 0x8001):
        if not 1 <= len(value) <= (16 if key == 0x8000 else 64):
            raise ProtocolError("invalid identity string length")
        return value.split(b"\0", 1)[0].decode("ascii", "replace")
    if key == 0x8002:
        return list(value)
    if key == 0x8005:
        return ":".join("%02x" % byte for byte in value)
    if key == 0x8011:
        return ["0x%08x" % number for number in struct.unpack("<8I", value)]
    if key == 0x8007:
        return struct.unpack("<i", value)[0]
    return int.from_bytes(value, "little")


def parse_query(payload, keys):
    if len(payload) < 3 or payload[0] != 0:
        raise ProtocolError("invalid query response")
    count = struct.unpack_from("<H", payload, 1)[0]
    if count != len(keys):
        raise ProtocolError("response parameter count differs from request")
    offset, seen, result = 3, set(), {}
    for _ in range(count):
        if offset + 4 > len(payload):
            raise ProtocolError("truncated parameter header")
        key, length = struct.unpack_from("<HH", payload, offset)
        offset += 4
        if key not in keys or key in seen or offset + length > len(payload):
            raise ProtocolError("unexpected, duplicated, or truncated parameter")
        seen.add(key)
        value = payload[offset:offset + length]
        offset += length
        result[PARAMETERS[key]] = decode_parameter(key, value)
    if offset != len(payload):
        raise ProtocolError("unexpected trailing query data")
    if "current_work_state" in result:
        result["current_work_state_name"] = WORK_STATES.get(result["current_work_state"], "unknown")
    if "core_temperature_raw" in result:
        result["core_temperature_c"] = result["core_temperature_raw"] / 100.0
    if "hms_codes_hex" in result:
        result["hms_nonzero"] = []
        for text in result["hms_codes_hex"]:
            code = int(text, 16)
            if not code:
                continue
            abnormal_id, level = code >> 16, code & 0xFF
            result["hms_nonzero"].append(dict(
                code=text, abnormal_id="0x%04x" % abnormal_id, level=level,
                meaning=("window_dirty_point_cloud_reliability_warning"
                         if code == 0x01040002 else "consult_model_HMS_table")))
    return result


def exchange(sock, lidar_ip, port, command, sequence, timeout, keys=(),
             broadcast=False, receive_sockets=None):
    if broadcast and (command != DISCOVERY or port != DISCOVERY_PORT):
        raise ValueError("broadcast is permitted only for discovery")
    request = build_request(command, sequence, keys)
    destination = ("255.255.255.255" if broadcast else lidar_ip, port)
    sock.sendto(request, destination)
    evidence = {"request_hex": request.hex(), "destination": list(destination),
                "expected_source": [lidar_ip, port]}
    deadline = time.monotonic() + timeout
    rejected, last_error = 0, None
    while time.monotonic() < deadline:
        sock.settimeout(max(0.001, deadline - time.monotonic()))
        try:
            receiver = sock
            if receive_sockets:
                readable, _, _ = select.select(receive_sockets, [], [],
                                                max(0, deadline - time.monotonic()))
                if not readable:
                    break
                receiver = readable[0]
            data, source = receiver.recvfrom(65535)
        except socket.timeout:
            break
        try:
            payload = parse_response(data, source, (lidar_ip, port), command, sequence)
        except ProtocolError as error:
            rejected += 1
            last_error = str(error)
            evidence["last_rejected_source"] = list(source)
            evidence["last_rejected_prefix_hex"] = data[:96].hex()
            continue
        evidence.update(response_hex=data.hex(), source=list(source), rejected_datagrams=rejected)
        return payload, evidence
    evidence.update(rejected_datagrams=rejected, last_error=last_error)
    raise ProbeTimeout("no valid matching response; rejected=%d; last_error=%s" %
                       (rejected, last_error), evidence)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-ip", required=True, help="existing local IPv4 address")
    parser.add_argument("--lidar-ip", required=True, help="single target IPv4 address")
    parser.add_argument("--timeout", type=float, default=3.0, help="1-10 seconds per request")
    parser.add_argument("--broadcast-discovery", action="store_true",
                        help="required by MID360: one broadcast, accepting only the target IP")
    args = parser.parse_args(argv)
    for name in ("host_ip", "lidar_ip"):
        try:
            address = ipaddress.IPv4Address(getattr(args, name))
        except ipaddress.AddressValueError:
            parser.error("%s must be an IPv4 address" % name)
        if address.is_multicast or address.is_unspecified or address.is_loopback or int(address) == 0xFFFFFFFF:
            parser.error("%s must be a unicast LAN IPv4 address" % name)
    if args.host_ip == args.lidar_ip:
        parser.error("host and LiDAR IP must differ")
    if not math.isfinite(args.timeout) or not 1 <= args.timeout <= 10:
        parser.error("timeout must be 1-10 seconds")
    return args


def emit(kind, **data):
    print(json.dumps(dict(kind=kind, **data), ensure_ascii=False), flush=True)


def main(argv=None):
    args = parse_args(argv)
    sequence = 1 + time.monotonic_ns() % 0xFFFF
    try:
        with contextlib.ExitStack() as stack:
            sock = stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
            # No address reuse: do not share or replace an existing driver's socket.
            sock.bind((args.host_ip, DISCOVERY_PORT))
            receivers = None
            if args.broadcast_discovery:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                # SDK2 also listens on the limited-broadcast address on Linux.
                # Use a separate socket without sharing any existing driver's port.
                broadcast_sock = stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
                broadcast_sock.bind(("255.255.255.255", DISCOVERY_PORT))
                receivers = (sock, broadcast_sock)
            payload, packet = exchange(sock, args.lidar_ip, DISCOVERY_PORT,
                                       DISCOVERY, sequence, args.timeout,
                                       broadcast=args.broadcast_discovery,
                                       receive_sockets=receivers)
            identity = parse_discovery(payload, args.lidar_ip)
            emit("discovery", identity=identity, packet=packet)
            payload, packet = exchange(sock, args.lidar_ip, identity["command_port"],
                                       QUERY, sequence % 0xFFFF + 1, args.timeout, QUERY_KEYS)
            values = parse_query(payload, QUERY_KEYS)
            if values["serial_number"] != identity["serial_number"]:
                raise ProtocolError("queried serial does not match discovered device")
            emit("query", values=values, packet=packet,
                 limitation="identity/state only; no optical data collected; HMS warnings require separate review")
    except (OSError, ProtocolError) as error:
        emit("error", reason=str(error), packet=getattr(error, "evidence", None))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
