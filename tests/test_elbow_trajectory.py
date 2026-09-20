"""Physical command envelope checks; importing the plan cannot connect to hardware."""
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "elbow", Path(__file__).resolve().parents[1] / "scripts/right_elbow_check.py")
elbow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(elbow)


class TrajectoryTests(unittest.TestCase):
    def test_full_envelope_and_speed(self):
        previous = 0.0
        for step in range(18001):
            weight, offset, _ = elbow.trajectory(step / 1000)
            self.assertGreaterEqual(weight, 0)
            self.assertLessEqual(weight, 1)
            self.assertGreaterEqual(offset, -1e-12)
            self.assertLessEqual(offset, math.radians(5) + 1e-12)
            self.assertLessEqual(abs(offset - previous) / .001, math.radians(2.51))
            previous = offset

    def test_return_before_release(self):
        for t in (12, 12.5, 13, 14, 15.99, 16, 100):
            self.assertEqual(elbow.trajectory(t)[1], 0)
        for t in (16, 17, 100):
            self.assertEqual(elbow.trajectory(t), (0, 0, "released"))

    def test_control_joint_scope(self):
        self.assertEqual(set(elbow.UPPER), set(range(12, 29)))
        self.assertEqual(elbow.ELBOW, 25)

    def test_negative_time_rejected(self):
        with self.assertRaises(ValueError):
            elbow.trajectory(-1)


if __name__ == "__main__":
    unittest.main()
