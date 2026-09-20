"""Bounded image reads from the existing Unitree videohub service (on robot).

Does not open the camera device or restart the service. Received-image rate is
the polling rate, not a measurement of the camera's native frame rate.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=20, choices=range(1, 31))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    import cv2
    import numpy as np
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.video.video_client import VideoClient

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ChannelFactoryInitialize(0, "eth0")
    client = VideoClient()
    client.SetTimeout(2.0)
    client.Init()
    start = time.monotonic()
    count = failures = 0
    hashes, dimensions = set(), set()
    previous = None
    max_gap = 0.0
    brightness, contrast = [], []
    last = None
    print(json.dumps(dict(kind="start", utc=datetime.datetime.now(
        datetime.timezone.utc).isoformat(), seconds=args.seconds)), flush=True)
    while time.monotonic() - start < args.seconds:
        tick = time.monotonic()
        code, data = client.GetImageSample()
        now = time.monotonic()
        if code != 0 or not data:
            failures += 1
            print(json.dumps(dict(kind="read_error", code=code)), flush=True)
            break
        image = cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            failures += 1
            print(json.dumps(dict(kind="decode_error")), flush=True)
            break
        count += 1
        hashes.add(hashlib.sha256(image.tobytes()).hexdigest())
        dimensions.add(tuple(image.shape))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness.append(float(gray.mean()))
        contrast.append(float(gray.std()))
        if previous is not None:
            max_gap = max(max_gap, now - previous)
        previous, last = now, image
        if count == 1 and not cv2.imwrite(str(output / "rgb-first.png"), image):
            raise RuntimeError("could not save first image")
        time.sleep(max(0.0, 0.1 - (time.monotonic() - tick)))
    if last is not None and not cv2.imwrite(str(output / "rgb-last.png"), last):
        raise RuntimeError("could not save last image")
    elapsed = time.monotonic() - start
    summary = dict(kind="summary", elapsed_s=elapsed, decoded_images=count,
                   unique_decoded_images=len(hashes), failures=failures,
                   dimensions=sorted(dimensions), received_images_per_s=count / elapsed,
                   max_receive_gap_s=max_gap,
                   brightness_range=[min(brightness), max(brightness)] if brightness else None,
                   contrast_range=[min(contrast), max(contrast)] if contrast else None,
                   limitation="polled RGB only; not native FPS, depth, or calibrated accuracy")
    print(json.dumps(summary), flush=True)
    return 0 if count >= args.seconds * 5 and len(hashes) >= 2 and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
