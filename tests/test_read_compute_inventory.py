"""Local inventory parser/error tests; no device access or installed SDK needed."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location(
    "compute_inventory", Path(__file__).resolve().parents[1] / "scripts/read_compute_inventory.py")
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


class ComputeInventoryTests(unittest.TestCase):
    def test_os_release_parses_selected_fields_without_shell_expansion(self):
        parsed = inventory.parse_os_release('NAME="Ubuntu Linux"\nVERSION_ID="20.04"\n'
                                            'EXTRA_SECRET=omit\nBUILD_ID="$(touch unsafe)"\n')
        self.assertEqual(parsed, {"NAME": "Ubuntu Linux", "VERSION_ID": "20.04",
                                  "BUILD_ID": "$(touch unsafe)"})

    def test_cpu_and_memory_keep_units_and_selected_identity(self):
        parsed = inventory.parse_cpuinfo("processor : 0\nCPU part : 0xd41\nSerial : omit\n\n"
                                         "processor : 1\nCPU part : 0xd41\n")
        self.assertEqual(parsed["logical_processors_in_cpuinfo"], 2)
        self.assertEqual(parsed["model_fields"], {"CPU part": ["0xd41"]})
        self.assertEqual(inventory.parse_meminfo("MemTotal: 1024 kB\nMemAvailable: 512 kB\n"
                                                "SwapTotal: 0 kB\nOther: 99 kB\n"),
                         {"MemTotal_bytes": 1048576, "MemAvailable_bytes": 524288, "SwapTotal_bytes": 0})

    def test_timeout_and_missing_tool_are_evidence_gaps(self):
        with mock.patch.object(inventory.shutil, "which", return_value=None), \
                mock.patch.object(inventory.subprocess, "run") as run:
            self.assertEqual(inventory.run_command(["absent"])["availability"], "unavailable")
        run.assert_not_called()
        with mock.patch.object(inventory.shutil, "which", return_value="/usr/bin/lsusb"), \
                mock.patch.object(inventory.subprocess, "run", side_effect=subprocess.TimeoutExpired("lsusb", 2)):
            result = inventory.run_command(["lsusb"], timeout=2)
        self.assertEqual(result["error"], "command timed out")
        self.assertEqual(result["timeout_seconds"], 2)

    def test_nonzero_smart_exit_retains_json_for_interpretation(self):
        completed = subprocess.CompletedProcess(["smartctl"], 8, '{"smart_status":{"passed":false}}', "")
        with mock.patch.object(inventory.shutil, "which", return_value="/usr/sbin/smartctl"), \
                mock.patch.object(inventory.subprocess, "run", return_value=completed):
            result = inventory.run_command(["smartctl"], json_output=True)
        self.assertEqual(result["returncode"], 8)
        self.assertFalse(result["data"]["smart_status"]["passed"])
        self.assertEqual(result["availability"], "error")

    def test_malformed_json_does_not_abort_inventory(self):
        completed = subprocess.CompletedProcess(["ip"], 0, "not json", "")
        with mock.patch.object(inventory.shutil, "which", return_value="/usr/bin/ip"), \
                mock.patch.object(inventory.subprocess, "run", return_value=completed):
            result = inventory.run_command(["ip"], json_output=True)
        self.assertEqual(result["availability"], "error")
        self.assertEqual(result["stdout"], "not json")

    def test_temperature_retains_unreadable_zone(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            zone = root / "thermal_zone0"
            zone.mkdir()
            (zone / "type").write_text("GPU-therm")
            (zone / "temp").write_text("52375")
            self.assertEqual(inventory.thermal_inventory(root)["zones"][0]["temperature"]["celsius"], 52.375)
            (zone / "temp").unlink()
            self.assertEqual(inventory.thermal_inventory(root)["zones"][0]["temperature"]["availability"],
                             "unavailable")

    def test_cuda_probe_is_self_contained_and_memory_is_opt_in(self):
        compile(inventory.CUDA_PROBE, "cuda_inventory_child", "exec")
        with mock.patch.object(inventory, "run_command", return_value={}) as run:
            inventory.cuda_inventory()
        arguments = run.call_args.args[0]
        self.assertIn("-B", arguments)
        self.assertIn("probe(False)", arguments[-1])
        self.assertEqual(run.call_args.kwargs["timeout"], 6)


if __name__ == "__main__":
    unittest.main()
