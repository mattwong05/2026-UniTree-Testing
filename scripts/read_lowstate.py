"""Bounded, subscriber-only G1 motor telemetry. Run on the robot's existing SDK."""
import argparse
import dataclasses
import datetime
import json
import math
import time
from collections import Counter

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC

NAMES = [
    "LeftHipPitch", "LeftHipRoll", "LeftHipYaw", "LeftKnee",
    "LeftAnklePitch", "LeftAnkleRoll", "RightHipPitch", "RightHipRoll",
    "RightHipYaw", "RightKnee", "RightAnklePitch", "RightAnkleRoll",
    "WaistYaw", "WaistRoll", "WaistPitch", "LeftShoulderPitch",
    "LeftShoulderRoll", "LeftShoulderYaw", "LeftElbow", "LeftWristRoll",
    "LeftWristPitch", "LeftWristYaw", "RightShoulderPitch", "RightShoulderRoll",
    "RightShoulderYaw", "RightElbow", "RightWristRoll", "RightWristPitch",
    "RightWristYaw",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 120:
        parser.error("seconds must be between 1 and 120")
    ChannelFactoryInitialize(0, args.interface)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init()
    crc = CRC()
    start = time.monotonic()
    count = invalid = nonfinite = 0
    previous_tick = None
    ticks = Counter()
    modes = set()
    axes = []
    next_snapshot = start
    max_gap = 0
    previous_received = None
    try:
        while time.monotonic() - start < args.seconds:
            msg = sub.Read(1)
            now = time.monotonic()
            if msg is None:
                continue
            if crc.Crc(msg) != msg.crc:
                invalid += 1
                continue
            if not axes:
                axes = [dict(index=i, name=NAMES[i] if i < 29 else "UnusedSlot",
                             mode_values=set(), error_values=set(), ranges={})
                        for i in range(len(msg.motor_state))]
            count += 1
            modes.add((msg.mode_machine, msg.mode_pr))
            if previous_received is not None:
                max_gap = max(max_gap, now - previous_received)
            previous_received = now
            if previous_tick is not None:
                ticks[(msg.tick - previous_tick) & 0xFFFFFFFF] += 1
            previous_tick = msg.tick
            for axis, motor in zip(axes, msg.motor_state):
                axis["mode_values"].add(motor.mode)
                axis["error_values"].add(motor.motorstate)
                fields = dict(q=motor.q, dq=motor.dq, tau_est=motor.tau_est,
                              temperature_0=motor.temperature[0],
                              temperature_1=motor.temperature[1], voltage=motor.vol)
                for key, value in fields.items():
                    if not math.isfinite(value):
                        nonfinite += 1
                        continue
                    old = axis["ranges"].setdefault(key, [value, value])
                    old[0], old[1] = min(old[0], value), max(old[1], value)
            if now >= next_snapshot:
                print(json.dumps(dict(kind="snapshot", elapsed_s=now-start,
                      utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      state=dataclasses.asdict(msg)),
                      default=lambda value: list(value) if isinstance(value, bytes)
                      else str(value)), flush=True)
                next_snapshot = now + 1
    finally:
        sub.Close()
    elapsed = time.monotonic() - start
    for axis in axes:
        for field in ("mode_values", "error_values"):
            axis[field] = sorted(axis[field])
    print(json.dumps(dict(kind="summary", elapsed_s=elapsed, valid_samples=count,
          crc_errors=invalid, nonfinite_fields=nonfinite,
          received_hz=count/elapsed, max_receive_gap_s=max_gap,
          tick_delta_counts=dict(ticks), modes=sorted(modes), axes=axes)), flush=True)
    if not count or invalid or nonfinite:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
