"""Short CPU/RAM and TensorRT FP16 correctness checks on Jetson (TRT 8.x).

Uses installed packages only. Does not change clocks, power modes or services.
The synthetic network's throughput is not a TOPS or application benchmark.
"""
import argparse
import ctypes as ct
import hashlib
import json
import multiprocessing
import time
from pathlib import Path


def max_temperature():
    values = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            values.append(int(path.read_text()) / 1000)
        except (OSError, ValueError):
            pass
    if not values:
        raise RuntimeError("No readable thermal sensors; refusing compute load")
    maximum = max(values)
    if maximum >= 75:
        raise RuntimeError("Conservative test stop at %.1f C" % maximum)
    return maximum


def cpu_worker(seconds):
    data = b"\x5a" * (8 * 1024 * 1024)
    expected = hashlib.sha256(data).digest()
    count, start = 0, time.monotonic()
    while time.monotonic() - start < seconds:
        if hashlib.sha256(data).digest() != expected:
            raise RuntimeError("CPU/RAM digest mismatch")
        count += 1
        max_temperature()
    return count


def inference_check(target, seconds):
    import numpy as np
    import tensorrt as trt

    max_temperature()
    cuda = ct.CDLL("/usr/local/cuda/lib64/libcudart.so")
    cuda.cudaMalloc.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t]
    cuda.cudaMemcpy.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int]
    cuda.cudaFree.argtypes = [ct.c_void_p]

    def check(code):
        if code:
            raise RuntimeError("CUDA error code %d" % code)

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 20)
    config.set_flag(trt.BuilderFlag.FP16)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if target != "gpu":
        core = int(target[-1])
        if core >= builder.num_DLA_cores:
            raise RuntimeError("Requested DLA core does not exist")
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = core
        # GPU_FALLBACK is deliberately not enabled.

    rng = np.random.RandomState(7)
    host_input = rng.uniform(-1, 1, (1, 16, 32, 32)).astype(np.float16)
    weights = rng.uniform(-0.1, 0.1, (16, 16, 1, 1)).astype(np.float32)
    tensor = network.add_input("input", trt.float16, host_input.shape)
    conv = network.add_convolution_nd(tensor, 16, (1, 1), weights, np.zeros(16, np.float32))
    conv.precision = trt.float16
    conv.set_output_type(0, trt.float16)
    output = conv.get_output(0)
    output.name, output.dtype = "output", trt.float16
    network.mark_output(output)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed for " + target)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT context creation failed")
    for binding in range(engine.num_bindings):
        if engine.get_binding_format(binding) != trt.TensorFormat.LINEAR:
            raise RuntimeError("Test supports linear boundary buffers only")
    expected = np.einsum("oc,nchw->nohw", weights[:, :, 0, 0], host_input.astype(np.float32))
    host_output = np.empty_like(host_input)
    pointers = []
    bindings = [0] * engine.num_bindings
    try:
        for name, array in [("input", host_input), ("output", host_output)]:
            pointer = ct.c_void_p()
            check(cuda.cudaMalloc(ct.byref(pointer), array.nbytes))
            pointers.append(pointer)
            bindings[engine.get_binding_index(name)] = pointer.value
        check(cuda.cudaMemcpy(pointers[0], host_input.ctypes.data, host_input.nbytes, 1))
        count, start, worst, peak = 0, time.monotonic(), 0.0, max_temperature()
        while time.monotonic() - start < seconds:
            if not context.execute_v2(bindings):
                raise RuntimeError("TensorRT execution failed")
            check(cuda.cudaDeviceSynchronize())
            check(cuda.cudaMemcpy(host_output.ctypes.data, pointers[1], host_output.nbytes, 2))
            difference = float(np.max(np.abs(host_output.astype(np.float32) - expected)))
            if not np.allclose(host_output, expected, atol=0.003, rtol=0.02):
                raise RuntimeError("Numerical verification failed: %g" % difference)
            worst = max(worst, difference)
            count += 1
            if count % 50 == 0:
                peak = max(peak, max_temperature())
        elapsed = time.monotonic() - start
        peak = max(peak, max_temperature())
        print(json.dumps(dict(kind="inference", target=target, result="PASS",
              precision="FP16", network="16x16 1x1 convolution, 1x16x32x32",
              seconds=elapsed, verified_iterations=count, max_abs_error=worst,
              peak_thermal_c=peak, gpu_fallback_enabled=False,
              tensorrt=trt.__version__, dla_cores=builder.num_DLA_cores,
              layers=engine.create_engine_inspector().get_engine_information(
                  trt.LayerInformationFormat.JSON))), flush=True)
    finally:
        for pointer in pointers:
            cuda.cudaFree(pointer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--target", choices=["all", "cpu", "gpu", "dla0", "dla1"], default="all")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 30:
        parser.error("seconds must be between 1 and 30")
    if args.target in ("all", "cpu"):
        max_temperature()
        with multiprocessing.Pool(8) as pool:
            counts = pool.map(cpu_worker, [args.seconds] * 8)
        print(json.dumps(dict(kind="cpu_ram", result="PASS", workers=8,
              buffer_mib_per_worker=8, seconds_per_worker=args.seconds,
              verified_hashes=counts, temperature_after_c=max_temperature())), flush=True)
    for target in ("gpu", "dla0", "dla1"):
        if args.target in ("all", target):
            inference_check(target, args.seconds)


if __name__ == "__main__":
    main()
