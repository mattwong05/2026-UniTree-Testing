"""Offline PointCloud2 fixtures; no DDS dependencies or device connections."""
import importlib.util
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest

try:
    import numpy as np
except ImportError:
    np = None


spec = importlib.util.spec_from_file_location(
    "lidar_cloud_check",
    Path(__file__).resolve().parents[1] / "scripts/lidar_cloud_check.py",
)
cloud_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloud_check)


def field(name, offset, datatype=7, count=1):
    return SimpleNamespace(name=name, offset=offset, datatype=datatype, count=count)


def cloud(payload, *, width=1, height=1, point_step=12, row_step=None,
          fields=None, is_bigendian=False):
    return SimpleNamespace(
        width=width,
        height=height,
        point_step=point_step,
        row_step=width * point_step if row_step is None else row_step,
        fields=fields if fields is not None else [
            field("x", 0), field("y", 4), field("z", 8)],
        is_bigendian=is_bigendian,
        data=payload,
    )


@unittest.skipIf(np is None, "numpy is required for offline PointCloud2 decoding tests")
class ExtractXYZTests(unittest.TestCase):
    def test_float32_little_and_big_endian(self):
        expected = [[1.25, -2.5, 3.75], [-4.5, 5.25, -6.75]]
        for endian, bigendian in (("<", False), (">", True)):
            with self.subTest(endian=endian):
                payload = struct.pack(endian + "6f", *expected[0], *expected[1])
                result = cloud_check.extract_xyz(cloud(
                    list(payload), width=2, is_bigendian=bigendian))
                np.testing.assert_array_equal(result, expected)
                self.assertEqual(result.shape, (2, 3))

    def test_organized_cloud_respects_point_and_row_padding(self):
        expected = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        rows = [
            struct.pack("<8f", *expected[0], 99, *expected[1], 98),
            struct.pack("<8f", *expected[2], 97, *expected[3], 96),
        ]
        payload = rows[0] + b"\xff" * 8 + rows[1] + b"\xff" * 8
        message = cloud(payload, width=2, height=2, point_step=16, row_step=40,
                        fields=[field("x", 0), field("y", 4), field("z", 8),
                                field("intensity", 12)])
        np.testing.assert_array_equal(cloud_check.extract_xyz(message), expected)

    def test_float64_and_field_order_follow_offsets(self):
        expected = [[1.000000000001, -2.000000000003, 3.000000000005]]
        for endian, bigendian in (("<", False), (">", True)):
            with self.subTest(endian=endian):
                # Physical layout is Y, unused uint64, Z, X; declarations differ.
                payload = struct.pack(endian + "dQdd", expected[0][1], 123,
                                      expected[0][2], expected[0][0])
                message = cloud(payload, point_step=32, is_bigendian=bigendian,
                                fields=[field("z", 16, 8), field("x", 24, 8),
                                        field("unused", 8, 6), field("y", 0, 8)])
                result = cloud_check.extract_xyz(message)
                np.testing.assert_array_equal(result, expected)
                self.assertEqual(result.dtype.itemsize, 8)

    def test_mixed_coordinate_precision(self):
        message = cloud(struct.pack("<fdf", 1.25, 2.000000000003, -3.5),
                        point_step=16,
                        fields=[field("y", 4, 8), field("z", 12), field("x", 0)])
        np.testing.assert_array_equal(cloud_check.extract_xyz(message),
                                      [[1.25, 2.000000000003, -3.5]])

    def test_short_and_extra_payload_bytes_are_rejected(self):
        payload = struct.pack("<3f", 1, 2, 3)
        for malformed in (b"", payload[:-1], payload + b"\0"):
            with self.subTest(length=len(malformed)), self.assertRaises(ValueError):
                cloud_check.extract_xyz(cloud(malformed))
        # Even unused padding must be present for every organized row.
        with self.assertRaises(ValueError):
            cloud_check.extract_xyz(cloud(payload, row_step=16))

    def test_too_small_row_stride_and_empty_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            cloud_check.extract_xyz(cloud(bytes(20), width=2, row_step=20))
        for setting in ({"width": 0}, {"height": 0}, {"point_step": 0}):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                cloud_check.extract_xyz(cloud(b"", **setting))

    def test_negative_and_out_of_bounds_offsets_are_rejected(self):
        for invalid in (field("x", -1), field("x", 9), field("x", 8, 8)):
            with self.subTest(offset=invalid.offset, datatype=invalid.datatype):
                with self.assertRaises(ValueError):
                    cloud_check.extract_xyz(cloud(bytes(12), fields=[
                        invalid, field("y", 4), field("z", 8)]))

    def test_missing_duplicate_nonscalar_and_nonfloat_axes_are_rejected(self):
        valid = [field("x", 0), field("y", 4), field("z", 8)]
        cases = {
            "missing_z": valid[:2],
            "duplicate_x": valid + [field("x", 0)],
            "zero_count": [field("x", 0, count=0), *valid[1:]],
            "array_count": [field("x", 0, count=2), *valid[1:]],
            "integer_axis": [field("x", 0, datatype=5), *valid[1:]],
        }
        for name, fields in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                cloud_check.extract_xyz(cloud(bytes(12), fields=fields))

    def test_nonfinite_points_are_preserved_for_summary_accounting(self):
        payload = struct.pack("<12f", 1, 2, 3, float("nan"), 4, 5,
                              6, float("inf"), 7, 8, 9, float("-inf"))
        result = cloud_check.extract_xyz(cloud(payload, width=4))
        self.assertEqual(result.shape, (4, 3))
        np.testing.assert_array_equal(result[0], [1, 2, 3])
        self.assertTrue(np.isnan(result[1, 0]))
        self.assertTrue(np.isposinf(result[2, 1]))
        self.assertTrue(np.isneginf(result[3, 2]))
        np.testing.assert_array_equal(np.isfinite(result).all(axis=1),
                                      [True, False, False, False])


if __name__ == "__main__":
    unittest.main()
