"""Physical command envelope checks; importing the plan cannot connect to hardware."""
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/upper_body_check.py"
spec = importlib.util.spec_from_file_location(
    "arm", SCRIPT)
arm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arm)


class TrajectoryTests(unittest.TestCase):
    def test_full_envelope_and_speed(self):
        for degrees in (-5, -2, -0.1, 0.1, 2, 5):
            delta = math.radians(degrees)
            previous = 0.0
            with self.subTest(degrees=degrees):
                for step in range(18001):
                    weight, offset, _ = arm.trajectory(step / 1000, delta)
                    self.assertGreaterEqual(weight, 0)
                    self.assertLessEqual(weight, 1)
                    self.assertGreaterEqual(offset, min(0, delta) - 1e-12)
                    self.assertLessEqual(offset, max(0, delta) + 1e-12)
                    self.assertLessEqual(abs(offset - previous) / .001, math.radians(2.51))
                    previous = offset
                self.assertEqual(arm.trajectory(8, delta), (1, delta, "hold"))

    def test_return_before_release(self):
        for t in (12, 12.5, 13, 14, 15.99, 16, 100):
            self.assertEqual(arm.trajectory(t)[1], 0)
        for t in (16, 17, 100):
            self.assertEqual(arm.trajectory(t), (0, 0, "released"))

    def test_control_joint_scope(self):
        self.assertEqual(set(arm.UPPER), set(range(12, 29)))
        self.assertEqual({spec[0] for spec in arm.JOINTS.values()}, set(range(15, 29)))
        self.assertEqual(arm.JOINTS["right_elbow"][0], 25)

    def test_negative_time_rejected(self):
        with self.assertRaises(ValueError):
            arm.trajectory(-1)

    def test_targets_respect_joint_bounds_in_both_directions(self):
        for name, (_, lower, upper) in arm.JOINTS.items():
            with self.subTest(joint=name):
                for degrees in (-5, -0.1, 0.1, 5):
                    delta = math.radians(degrees)
                    self.assertAlmostEqual(arm.validate_target(name, 0, delta), delta)
                near_upper = upper - math.radians(6)
                near_lower = lower + math.radians(6)
                arm.validate_target(name, near_upper, math.radians(-2))
                arm.validate_target(name, near_lower, math.radians(2))
                with self.assertRaises(ValueError):
                    arm.validate_target(name, near_upper, math.radians(2))
                with self.assertRaises(ValueError):
                    arm.validate_target(name, near_lower, math.radians(-2))

    def test_invalid_offsets_and_telemetry_rejected(self):
        for delta in (0, math.radians(.09), math.radians(5.01), math.nan, math.inf):
            with self.assertRaises(ValueError):
                arm.validate_target("right_elbow", 0, delta)
        for q in (math.nan, math.inf, -math.inf, 100):
            with self.assertRaises(ValueError):
                arm.validate_target("right_elbow", q, arm.DELTA)

    def test_list_joints_needs_no_sdk(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--list-joints"],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"left_wrist_yaw": 21', result.stdout)

    def test_invalid_cli_rejected_before_sdk_import(self):
        for args in (("--execute", "--check-all"), ("--delta-deg", "nan"),
                     ("--delta-deg", "6"), ("--joint", "waist_yaw")):
            result = subprocess.run([sys.executable, str(SCRIPT), *args],
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertNotIn("unitree_sdk2py", result.stderr)


if __name__ == "__main__":
    unittest.main()
