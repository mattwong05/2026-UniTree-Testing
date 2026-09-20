"""Pure local checks; no TensorRT imports, CUDA access or device connection."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location(
    "int8_bench", Path(__file__).resolve().parents[1] / "scripts/compute_int8_bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def tensor(fmt="INT8"):
    return {"Format/Datatype": "Row major linear " + fmt + " format"}


class Int8BenchmarkTests(unittest.TestCase):
    def test_operation_count_is_dense_mac_twice(self):
        self.assertEqual(bench.NOMINAL_OPS, 4294967296)
        self.assertEqual(bench.workload_dimensions(32), ((1, 512, 32, 32), 4294967296))
        self.assertEqual(bench.workload_dimensions(64), ((1, 512, 64, 64), 17179869184))
        with self.assertRaises(ValueError):
            bench.workload_dimensions(128)

    def test_spatial_cli_is_bounded_and_gpu_only(self):
        self.assertEqual(bench.parse_args([]).spatial, 32)
        self.assertEqual(bench.parse_args(["--target", "gpu", "--spatial", "64"]).spatial, 64)
        for args in (("--spatial", "64"), ("--spatial", "128", "--target", "gpu"),
                     ("--spatial", "64", "--target", "dla0"),
                     ("--spatial", "64", "--target", "dla1")):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    bench.parse_args(args)

    def test_spatial_64_dla_rejected_before_hardware_access(self):
        with mock.patch.object(bench, "ThermalGuard") as guard:
            with self.assertRaises(ValueError):
                bench.benchmark("dla0", 5, False, 100, spatial=64)
        guard.assert_not_called()

    def test_cli_load_limits(self):
        self.assertEqual(bench.parse_args([]).seconds, 10)
        for args in (("--seconds", "0"), ("--seconds", "31"),
                     ("--window-iterations", "0"), ("--window-iterations", "101"),
                     ("--sparse",), ("--sparse", "--target", "dla0")):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    bench.parse_args(args)
        self.assertTrue(bench.parse_args(["--target", "gpu", "--sparse"]).sparse)

    def test_temperature_missing_hot_and_maximum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                bench.read_temperature(root)
            for number, value in enumerate((51000, 61000)):
                folder = root / ("thermal_zone%d" % number)
                folder.mkdir()
                (folder / "temp").write_text(str(value))
            self.assertEqual(bench.read_temperature(root), 61)
            (root / "thermal_zone1/temp").write_text("75000")
            with self.assertRaises(RuntimeError):
                bench.read_temperature(root)

    def test_gpu_requires_eight_int8_convolutions(self):
        layers = [{"LayerType": "CaskConvolution", "Inputs": [tensor()], "Outputs": [tensor()]}
                  for _ in range(8)]
        self.assertIn("eight", bench.inspect_precision({"Layers": layers}, "gpu"))
        with self.assertRaises(RuntimeError):
            bench.inspect_precision({"Layers": layers[:7]}, "gpu")
        layers[0]["Outputs"] = [tensor("FP16")]
        with self.assertRaises(RuntimeError):
            bench.inspect_precision({"Layers": layers}, "gpu")

    def test_dla_precision_evidence_retains_opaque_limitation(self):
        layer = {"LayerType": "DLA", "Inputs": [tensor()], "Outputs": [tensor()]}
        self.assertIn("opaque", bench.inspect_precision({"Layers": [layer]}, "dla0"))
        layer["Inputs"] = [tensor("FP16")]
        with self.assertRaises(RuntimeError):
            bench.inspect_precision({"Layers": [layer]}, "dla0")

    def test_watchdog_exits_when_temperature_reader_raises(self):
        with mock.patch.object(bench, "read_temperature", return_value=50):
            guard = bench.ThermalGuard("gpu")
        guard.stopped = mock.Mock()
        guard.stopped.wait.side_effect = [False, True]
        with mock.patch.object(bench, "read_temperature", side_effect=RuntimeError("sensor unavailable")), \
                mock.patch.object(bench, "emit") as emit, \
                mock.patch.object(bench.os, "_exit") as exit_process:
            guard._watch()
        exit_process.assert_called_once_with(3)
        self.assertEqual(emit.call_args.kwargs["kind"], "safety_stop")
        self.assertEqual(emit.call_args.kwargs["reason"], "sensor unavailable")

    def test_watchdog_exits_after_expired_build_deadline(self):
        with mock.patch.object(bench, "read_temperature", return_value=50):
            guard = bench.ThermalGuard("gpu")
            guard.stage, guard.deadline = "build", 0
            guard.stopped = mock.Mock()
            guard.stopped.wait.side_effect = [False, True]
            with mock.patch.object(bench, "emit") as emit, \
                    mock.patch.object(bench.os, "_exit") as exit_process:
                guard._watch()
        exit_process.assert_called_once_with(3)
        self.assertIn("build", emit.call_args.kwargs["reason"])


if __name__ == "__main__":
    unittest.main()
