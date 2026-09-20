"""Read-only Jetson inventory; run with Python 3.8+ locally or via SSH stdin.

No sudo, installation, control APIs, benchmark, or persistent configuration writes.
Missing tools and denied reads describe evidence availability, not hardware health.
Use --cuda-memory-info only when CUDA context initialization is acceptable.
"""
import sys

sys.dont_write_bytecode = True

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess


def read_text(path):
    try:
        return {"availability": "available", "value": Path(path).read_text(errors="replace").strip("\0\n ")}
    except OSError as error:
        return {"availability": "unavailable", "error": str(error)}


def run_command(arguments, timeout=4, json_output=False):
    """Run an explicit read-only command; never invoke a shell or prompt for sudo."""
    if shutil.which(arguments[0]) is None:
        return {"availability": "unavailable", "error": "command not installed: " + arguments[0]}
    try:
        result = subprocess.run(arguments, stdin=subprocess.DEVNULL, capture_output=True,
                                encoding="utf-8", errors="replace", timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"availability": "unavailable", "error": "command timed out", "timeout_seconds": timeout}
    except OSError as error:
        return {"availability": "unavailable", "error": str(error)}
    record = {"availability": "available" if result.returncode == 0 else "error",
              "returncode": result.returncode}
    if result.stderr.strip():
        record["stderr"] = result.stderr.strip()
    if result.returncode:
        record["error"] = "command returned nonzero; inspect output, not a hardware-failure verdict"
    if json_output:
        try:
            record["data"] = json.loads(result.stdout)
        except ValueError:
            record.update(availability="error", error="command did not return valid JSON",
                          stdout=result.stdout.strip())
    else:
        record["stdout"] = result.stdout.strip()
    return record


def parse_os_release(text):
    allowed = {"NAME", "PRETTY_NAME", "ID", "VERSION", "VERSION_ID", "BUILD_ID"}
    result = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in allowed:
            try:
                words = shlex.split(value, comments=False)
                result[key] = words[0] if len(words) == 1 else value
            except ValueError:
                result[key] = value
    return result


def parse_cpuinfo(text):
    allowed = {"model name", "Hardware", "CPU implementer", "CPU architecture",
               "CPU variant", "CPU part", "CPU revision"}
    models, processors = {}, set()
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if separator and key == "processor" and value.isdigit():
            processors.add(int(value))
        if separator and key in allowed:
            models.setdefault(key, set()).add(value)
    return {"logical_processors_in_cpuinfo": len(processors),
            "model_fields": {key: sorted(values) for key, values in models.items()}}


def parse_meminfo(text):
    result = {}
    for line in text.splitlines():
        match = re.fullmatch(r"(MemTotal|MemAvailable|SwapTotal|SwapFree):\s+(\d+) kB", line.strip())
        if match:
            result[match.group(1) + "_bytes"] = int(match.group(2)) * 1024
    return result


def parsed_file(path, parser):
    record = read_text(path)
    if record["availability"] == "available":
        record["data"] = parser(record.pop("value"))
    return record


def network_inventory():
    record = run_command(["ip", "-j", "address", "show"], json_output=True)
    if isinstance(record.get("data"), list):
        record["data"] = [
            {**{key: interface[key] for key in ("ifname", "ifindex", "address", "operstate", "mtu")
                if key in interface},
             "addresses": [{key: address[key] for key in ("family", "local", "prefixlen", "scope")
                            if key in address} for address in interface.get("addr_info", [])]}
            for interface in record["data"]]
    return record


def nvme_inventory(root=Path("/sys/class/nvme")):
    controllers = []
    for path in sorted(root.glob("nvme*")):
        if not re.fullmatch(r"nvme\d+", path.name):
            continue
        record = {"controller": path.name,
                  **{name: read_text(path / name) for name in ("model", "serial", "firmware_rev")}}
        device = "/dev/" + path.name
        if shutil.which("nvme"):
            record["smart"] = run_command(["nvme", "smart-log", device, "--output-format=json"],
                                          timeout=5, json_output=True)
        elif shutil.which("smartctl"):
            record["smart"] = run_command(["smartctl", "-H", "-A", "-j", device],
                                          timeout=5, json_output=True)
        else:
            record["smart"] = {"availability": "unavailable", "error": "neither nvme nor smartctl installed"}
        controllers.append(record)
    return {"availability": "available" if root.exists() else "unavailable", "controllers": controllers}


def thermal_inventory(root=Path("/sys/class/thermal")):
    zones = []
    for path in sorted(root.glob("thermal_zone*")):
        value = read_text(path / "temp")
        if value["availability"] == "available":
            try:
                value["celsius"] = int(value.pop("value")) / 1000.0
            except ValueError:
                value.update(availability="error", error="invalid millidegree temperature")
        zones.append({"zone": path.name, "type": read_text(path / "type"), "temperature": value})
    return {"availability": "available" if zones else "unavailable", "zones": zones}


def frequency_inventory():
    gpu = []
    for path in sorted(Path("/sys/class/devfreq").glob("*")):
        if any(token in path.name.lower() for token in ("gpu", "ga10b", "gv11b", "gp10b")):
            gpu.append({"device": path.name, "units": "Hz",
                        **{key: read_text(path / key) for key in
                           ("min_freq", "max_freq", "cur_freq", "available_frequencies")}})
    dla = []
    for name in ("dla0_core", "dla1_core"):
        path = Path("/sys/kernel/debug/bpmp/debug/clk") / name
        dla.append({"device": name, "units": "Hz",
                    **{key: read_text(path / key) for key in ("rate", "max_rate")}})
    return {"gpu": gpu, "dla": dla, "note": "Caps are configuration limits, not sustained measured clocks."}


# Keep this self-contained: __file__ need not exist when the main script arrives
# over SSH stdin. A subprocess timeout also bounds native CUDA initialization.
CUDA_PROBE = r'''
import ctypes as ct
import ctypes.util
import json

def probe(include_memory):
    candidates = ["/usr/local/cuda/lib64/libcudart.so", ct.util.find_library("cudart"), "libcudart.so"]
    library = None
    for name in candidates:
        if name:
            try:
                library = ct.CDLL(name)
                break
            except OSError:
                pass
    if library is None:
        return {"availability": "unavailable", "error": "CUDA runtime library not found"}
    signatures = {
        "cudaGetDeviceCount": [ct.POINTER(ct.c_int)],
        "cudaDeviceGetAttribute": [ct.POINTER(ct.c_int), ct.c_int, ct.c_int],
        "cudaRuntimeGetVersion": [ct.POINTER(ct.c_int)],
        "cudaDriverGetVersion": [ct.POINTER(ct.c_int)],
        "cudaSetDevice": [ct.c_int],
        "cudaMemGetInfo": [ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t)],
    }
    for name, arguments in signatures.items():
        getattr(library, name).argtypes = arguments
        getattr(library, name).restype = ct.c_int

    def integer_call(name, *arguments):
        value = ct.c_int()
        code = getattr(library, name)(ct.byref(value), *arguments)
        return ({"availability": "available", "value": value.value} if code == 0 else
                {"availability": "unavailable", "cuda_error_code": code})

    count = integer_call("cudaGetDeviceCount")
    result = {"device_count": count, "devices": [],
              "runtime_version": integer_call("cudaRuntimeGetVersion"),
              "driver_version": integer_call("cudaDriverGetVersion"),
              "note": "Enumeration only; no kernels or explicit GPU buffer allocations. "
                      "Optional memory query can initialize a CUDA context and is not a memory-health test."}
    if count["availability"] != "available":
        result["availability"] = "unavailable"
        return result
    result["availability"] = "available"
    for index in range(count["value"]):
        device = {"index": index}
        for name, attribute in (("multiprocessor_count", 16), ("compute_capability_major", 75),
                                ("compute_capability_minor", 76), ("clock_rate_khz", 13)):
            device[name] = integer_call("cudaDeviceGetAttribute", attribute, index)
        if include_memory:
            code = library.cudaSetDevice(index)
            free, total = ct.c_size_t(), ct.c_size_t()
            if code == 0:
                code = library.cudaMemGetInfo(ct.byref(free), ct.byref(total))
            device["memory"] = ({"availability": "available", "free_bytes": free.value,
                                 "total_bytes": total.value} if code == 0 else
                                {"availability": "unavailable", "cuda_error_code": code})
        else:
            device["memory"] = {"availability": "not_requested",
                                "reason": "avoid CUDA context initialization; use --cuda-memory-info to opt in"}
        result["devices"].append(device)
    return result
'''


def cuda_inventory(include_memory=False):
    code = CUDA_PROBE + "\nprint(json.dumps(probe(" + repr(include_memory) + ")))\n"
    return run_command([sys.executable, "-B", "-c", code], timeout=6, json_output=True)


def software_versions():
    results = {"python": sys.version.split()[0]}
    for name in ("numpy", "tensorrt"):
        code = ("import importlib, json; module = importlib.import_module(" + repr(name) + "); "
                "print(json.dumps({'version': getattr(module, '__version__', 'unknown')}))")
        results[name] = run_command([sys.executable, "-B", "-c", code], timeout=5, json_output=True)
    return results


def collect_inventory(include_cuda_memory=False):
    device_tree = {}
    for name in ("model", "compatible", "serial-number"):
        record = read_text(Path("/proc/device-tree") / name)
        if name == "compatible" and "value" in record:
            record["value"] = record["value"].split("\0")
        device_tree[name] = record
    video_root = Path("/sys/class/video4linux")
    videos = [{"device": "/dev/" + path.name, "name": read_text(path / "name")}
              for path in sorted(video_root.glob("video*"))]
    return {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_tree": device_tree,
        "os_release": parsed_file("/etc/os-release", parse_os_release),
        "l4t_release": read_text("/etc/nv_tegra_release"),
        "cpu": {"logical_count": os.cpu_count(), "online": read_text("/sys/devices/system/cpu/online"),
                "identity": parsed_file("/proc/cpuinfo", parse_cpuinfo)},
        "memory": parsed_file("/proc/meminfo", parse_meminfo),
        "disks": run_command(["lsblk", "--json", "--bytes", "--output",
                              "NAME,TYPE,SIZE,MODEL,SERIAL,TRAN,ROTA,RO"], json_output=True),
        "local_filesystems": run_command(["df", "-B1", "-P", "-l"]),
        "nvme": nvme_inventory(),
        "usb": run_command(["lsusb"]),
        "video": {"availability": "available" if video_root.exists() else "unavailable", "devices": videos},
        "network": network_inventory(),
        "temperature": thermal_inventory(),
        "power_mode": run_command(["nvpmodel", "-q"]),
        "frequency": frequency_inventory(),
        "software": software_versions(),
        "cuda": cuda_inventory(include_cuda_memory),
        "limitations": ["Read-only inventory, not performance, health, or shipment acceptance.",
                        "Unavailable tools, permission errors, and timeouts are evidence gaps, not hardware failures.",
                        "No environment, wireless SSID, process command lines, or credential files collected."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-memory-info", action="store_true",
                        help="also query CUDA free/total memory; may initialize a CUDA context")
    args = parser.parse_args()
    print(json.dumps(collect_inventory(args.cuda_memory_info), ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
