"""Subscribe to the G1 documented MID360 PointCloud2 topic; no control writes."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time


def extract_xyz(message):
    import numpy as np
    if not message.width or not message.height or not message.point_step:
        raise ValueError("empty cloud dimensions")
    if message.row_step < message.width * message.point_step:
        raise ValueError("row stride smaller than point data")
    raw = bytes(message.data)
    if len(raw) != message.row_step * message.height:
        raise ValueError("cloud payload length does not match row stride")
    fields = {}
    for field in message.fields:
        if field.name in fields:
            raise ValueError("duplicate field name")
        fields[field.name] = field
    axes = []
    for name in ("x", "y", "z"):
        field = fields.get(name)
        if field is None or field.count != 1 or field.datatype not in (7, 8):
            raise ValueError("XYZ must be scalar FLOAT32 or FLOAT64")
        size = 4 if field.datatype == 7 else 8
        if field.offset < 0 or field.offset + size > message.point_step:
            raise ValueError("XYZ offset exceeds point stride")
        dtype = (">" if message.is_bigendian else "<") + "f" + str(size)
        axes.append(np.ndarray((message.height, message.width), dtype=dtype,
                               buffer=raw, offset=field.offset,
                               strides=(message.row_step, message.point_step)).reshape(-1))
    return np.column_stack(axes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=20, choices=range(1, 31))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    import numpy as np
    from cyclonedds.domain import Domain, DomainParticipant
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from cyclonedds.qos import Qos, Policy
    from unitree_sdk2py.core.channel_config import ChannelConfigHasInterface
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    domain = Domain(0, ChannelConfigHasInterface.replace('$__IF_NAME__$', 'eth0'))
    participant = DomainParticipant(0)
    topic = Topic(participant, "rt/utlidar/cloud_livox_mid360", PointCloud2_)
    reader = DataReader(participant, topic, Qos(Policy.Reliability.BestEffort,
                                               Policy.History.KeepLast(10)))
    start = time.monotonic()
    count = errors = total_points = finite_points = nonzero_points = regressions = duplicates = 0
    first = last = previous_stamp = None
    max_gap = 0.0
    hashes, frame_ids = set(), set()
    points_per_frame, ranges = [], []
    print(json.dumps(dict(kind="start", utc=datetime.datetime.now(
        datetime.timezone.utc).isoformat(), seconds=args.seconds)), flush=True)
    while time.monotonic() - start < args.seconds:
        for sample in reader.take(10):
            if not sample.sample_info.valid_data:
                continue
            now = time.monotonic()
            try:
                xyz = extract_xyz(sample)
            except ValueError as error:
                errors += 1
                print(json.dumps(dict(kind="decode_error", reason=str(error))), flush=True)
                continue
            count += 1
            stamp = sample.header.stamp.sec * 1000000000 + sample.header.stamp.nanosec
            if previous_stamp is not None:
                regressions += int(stamp < previous_stamp)
                duplicates += int(stamp == previous_stamp)
            previous_stamp = stamp
            if last is not None:
                max_gap = max(max_gap, now - last)
            if first is None:
                first = now
            last = now
            mask = np.isfinite(xyz).all(axis=1)
            good = xyz[mask]
            distances = np.linalg.norm(good, axis=1)
            nonzero = distances > 0
            total_points += len(xyz)
            finite_points += len(good)
            nonzero_points += int(nonzero.sum())
            points_per_frame.append(len(xyz))
            if nonzero.any():
                ranges.append([float(distances[nonzero].min()), float(distances.max())])
            hashes.add(hashlib.sha256(bytes(sample.data)).hexdigest())
            frame_ids.add(sample.header.frame_id)
            if count == 1:
                np.save(output / "lidar-first.npy", xyz)
                print(json.dumps(dict(kind="first_frame", width=sample.width,
                    height=sample.height, point_step=sample.point_step, row_step=sample.row_step,
                    frame_id=sample.header.frame_id,
                    fields=[dict(name=f.name, offset=f.offset, datatype=f.datatype,
                                 count=f.count) for f in sample.fields])), flush=True)
        time.sleep(0.002)
    summary = dict(kind="summary", seconds=time.monotonic() - start, frames=count,
                   unique_payloads=len(hashes), decode_errors=errors,
                   received_hz=(count - 1) / (last - first) if count > 1 and last > first else None,
                   max_receive_gap_s=max_gap, timestamp_regressions=regressions,
                   duplicate_timestamps=duplicates, frame_ids=sorted(frame_ids),
                   total_points=total_points, finite_points=finite_points,
                   nonzero_points=nonzero_points,
                   points_per_frame_range=[min(points_per_frame), max(points_per_frame)] if count else None,
                   coordinate_distance_range=[min(x[0] for x in ranges), max(x[1] for x in ranges)] if ranges else None,
                   limitation="received point cloud only; no calibrated accuracy or long-duration test")
    print(json.dumps(summary), flush=True)
    return 0 if count >= 2 and len(hashes) >= 2 and nonzero_points and not (errors or regressions or duplicates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
