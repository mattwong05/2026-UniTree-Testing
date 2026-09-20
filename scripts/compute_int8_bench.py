"""Bounded synthetic INT8 throughput on Jetson / TensorRT 8.x.

This is not a measurement of advertised whole-module peak TOPS. The operation
count is dense-equivalent (multiply and add count separately), also with 2:4
weights. No clocks, power modes, packages, files or services are changed.
Run via SSH stdin; JSON lines, including the full engine inspector, go to stdout.
"""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import sys
import threading
import time


CHANNELS = 512
LAYERS = 8
STOP_TEMPERATURE_C = 75


def workload_dimensions(spatial):
    if spatial not in (32, 64):
        raise ValueError("spatial must be 32 or 64")
    shape = (1, CHANNELS, spatial, spatial)
    nominal_ops = LAYERS * 2 * CHANNELS * CHANNELS * spatial * spatial
    return shape, nominal_ops


SHAPE, NOMINAL_OPS = workload_dimensions(32)


def emit(**record):
    print(json.dumps(record, allow_nan=False), flush=True)


def read_temperature(root=Path("/sys/class/thermal")):
    temperatures = []
    for path in root.glob("thermal_zone*/temp"):
        try:
            value = int(path.read_text()) / 1000.0
            if -20 <= value <= 150:
                temperatures.append(value)
        except (OSError, ValueError):
            pass
    if not temperatures:
        raise RuntimeError("No readable plausible thermal sensor; refusing load")
    maximum = max(temperatures)
    if maximum >= STOP_TEMPERATURE_C:
        raise RuntimeError("Conservative test stop at %.3f C (threshold 75 C)" % maximum)
    return maximum


class ThermalGuard:
    """Monitor temperature and phase deadlines when the Python thread can run."""

    def __init__(self, target):
        self.target = target
        self.current = self.peak = read_temperature()
        self.stage = "initialization"
        self.deadline = time.monotonic() + 30
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def set_stage(self, stage, maximum_seconds):
        self.stage = stage
        self.deadline = time.monotonic() + maximum_seconds

    def _watch(self):
        while not self.stopped.wait(0.5):
            try:
                self.current = read_temperature()
                self.peak = max(self.peak, self.current)
                if time.monotonic() >= self.deadline:
                    raise RuntimeError("Hard time limit exceeded during " + self.stage)
            except Exception as error:
                emit(kind="safety_stop", target=self.target, result="FAIL",
                     stage=self.stage, reason=str(error))
                # Native build/execute may not return promptly. Process exit
                # releases its CUDA context; ordinary paths clean up explicitly.
                os._exit(3)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stopped.set()
        self.thread.join(timeout=1)


def make_weights(np, sparse):
    row, col = np.indices((CHANNELS, CHANNELS))
    if sparse:
        # Exactly two nonzeros per four consecutive input channels, for
        # every output channel. All rows sum to one. Never prune after build.
        values = np.where((col - row) % 4 < 2, 1.0 / 256, 0.0)
    else:
        # 384 positive and 128 negative coefficients: row sum is exactly one;
        # no zero weights and no constant network input available to the builder.
        values = np.where((col - row) % CHANNELS < 384, 1.0 / 256, -1.0 / 256)
    return values.astype(np.float32).reshape(CHANNELS, CHANNELS, 1, 1)


def inspect_precision(info, target):
    layers = info.get("Layers", [])
    if not layers:
        raise RuntimeError("Empty engine inspector; cannot verify INT8 execution")
    if target == "gpu":
        convolutions = [layer for layer in layers
                        if "convolution" in str(layer.get("LayerType", "")).lower()
                        or "caskconvolution" in str(layer.get("ParameterType", "")).lower()]
        if len(convolutions) != LAYERS:
            raise RuntimeError("Expected eight inspectable GPU convolution layers, got %d" % len(convolutions))
        for layer in convolutions:
            formats = [str(item.get("Format/Datatype", "")).lower()
                       for item in layer.get("Inputs", []) + layer.get("Outputs", [])]
            if not formats or not all("int8" in value for value in formats):
                raise RuntimeError("GPU convolution lacks INT8 input/output evidence: " + str(layer.get("Name")))
        return "eight GPU convolutions with INT8 inputs and outputs"
    dla_layers = [layer for layer in layers if str(layer.get("LayerType", "")).upper() == "DLA"]
    if not dla_layers:
        raise RuntimeError("Requested DLA engine has no inspectable DLA node")
    for layer in dla_layers:
        formats = [str(item.get("Format/Datatype", "")).lower()
                   for item in layer.get("Inputs", []) + layer.get("Outputs", [])]
        if not formats or not all("int8" in value for value in formats):
            raise RuntimeError("DLA node boundaries do not demonstrate INT8")
    # TensorRT merges DLA layers into opaque ForeignNode records. Preserve this
    # limitation rather than claiming the inspector enumerates every operation.
    return "INT8 DLA node boundaries; eight requested INT8 convolutions under OBEY_PRECISION_CONSTRAINTS; internal fused layers opaque"


class Cuda:
    def __init__(self):
        self.lib = ct.CDLL("/usr/local/cuda/lib64/libcudart.so")
        signatures = {
            "cudaMalloc": [ct.POINTER(ct.c_void_p), ct.c_size_t],
            "cudaFree": [ct.c_void_p],
            "cudaMemcpy": [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int],
            "cudaStreamCreate": [ct.POINTER(ct.c_void_p)],
            "cudaStreamSynchronize": [ct.c_void_p],
            "cudaStreamDestroy": [ct.c_void_p],
            "cudaEventCreate": [ct.POINTER(ct.c_void_p)],
            "cudaEventRecord": [ct.c_void_p, ct.c_void_p],
            "cudaEventSynchronize": [ct.c_void_p],
            "cudaEventElapsedTime": [ct.POINTER(ct.c_float), ct.c_void_p, ct.c_void_p],
            "cudaEventDestroy": [ct.c_void_p],
            "cudaRuntimeGetVersion": [ct.POINTER(ct.c_int)],
            "cudaDriverGetVersion": [ct.POINTER(ct.c_int)],
        }
        for name, arguments in signatures.items():
            getattr(self.lib, name).argtypes = arguments
            getattr(self.lib, name).restype = ct.c_int

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            raise RuntimeError("%s failed with CUDA code %d" % (name, code))

    def version(self, name):
        value = ct.c_int()
        self.call(name, ct.byref(value))
        return value.value


def benchmark(target, seconds, sparse, window_iterations, spatial=32):
    shape, nominal_ops = workload_dimensions(spatial)
    if spatial == 64 and target != "gpu":
        raise ValueError("--spatial 64 requires --target gpu")
    with ThermalGuard(target) as guard:
        import numpy as np
        import tensorrt as trt

        cuda = Cuda()
        class StderrLogger(trt.ILogger):
            def __init__(self):
                trt.ILogger.__init__(self)

            def log(self, severity, message):
                if int(severity) <= int(trt.Logger.WARNING):
                    print("TensorRT [%s] %s" % (severity, message), file=sys.stderr, flush=True)

        logger = StderrLogger()
        builder = trt.Builder(logger)
        if not builder.platform_has_fast_int8:
            raise RuntimeError("Platform does not advertise fast INT8")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 128 << 20)
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        if sparse:
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        if target != "gpu":
            if int(target[-1]) >= builder.num_DLA_cores:
                raise RuntimeError("Requested DLA core does not exist")
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = int(target[-1])
            # GPU_FALLBACK remains disabled. Boundary reformat kernels may still
            # run on the GPU; their overhead remains in inference timings.

        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        boundary_dtype = trt.float16 if target == "gpu" else trt.int8
        tensor = network.add_input("input", boundary_dtype, shape)
        tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        tensor.dynamic_range = (-1.0, 1.0)
        weights = make_weights(np, sparse)
        biases = np.zeros(CHANNELS, np.float32)
        for index in range(LAYERS):
            conv = network.add_convolution_nd(tensor, CHANNELS, (1, 1), weights, biases)
            conv.name = "int8_conv_%d" % index
            conv.precision = trt.int8
            conv.set_output_type(0, trt.int8)
            tensor = conv.get_output(0)
            tensor.dynamic_range = (-1.0, 1.0)
        if target == "gpu":
            cast = network.add_identity(tensor)
            cast.name = "output_to_fp16"
            cast.set_output_type(0, trt.float16)
            tensor = cast.get_output(0)
        network.mark_output(tensor)
        tensor.name, tensor.dtype = "output", boundary_dtype
        tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        tensor.dynamic_range = (-1.0, 1.0)

        guard.set_stage("build", 180)
        build_start = time.monotonic()
        serialized = builder.build_serialized_network(network, config)
        build_seconds = time.monotonic() - build_start
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")
        guard.set_stage("engine_setup", 30)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise RuntimeError("TensorRT engine deserialization failed")
        inspector_json = engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
        inspector_info = json.loads(inspector_json)
        emit(kind="engine", target=target, tensorrt=trt.__version__,
             cuda_runtime=cuda.version("cudaRuntimeGetVersion"),
             cuda_driver=cuda.version("cudaDriverGetVersion"), dla_cores=builder.num_DLA_cores,
             shape=shape, convolution_layers=LAYERS, kernel=[1, 1],
             nominal_ops=nominal_ops, ops_convention="dense-equivalent; multiply=1, add=1",
             sparse_weights=sparse, sparse_tactic_selected="not independently verified" if sparse else "not requested",
             gpu_fallback_enabled=False, boundary_dtype=str(boundary_dtype),
             dynamic_range=[-1.0, 1.0], build_seconds=build_seconds,
             inspector=inspector_info)
        precision_evidence = inspect_precision(inspector_info, target)
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("TensorRT context creation failed")
        host_dtype = np.float16 if target == "gpu" else np.int8
        input_value = 64.0 / 127 if target == "gpu" else 64
        host_input = np.full(shape, input_value, dtype=host_dtype)
        host_output = np.empty_like(host_input)
        pointers, events = [], []
        stream = ct.c_void_p()
        bindings = [0] * engine.num_bindings
        try:
            for name, array in (("input", host_input), ("output", host_output)):
                binding = engine.get_binding_index(name)
                if (binding < 0 or engine.get_binding_format(binding) != trt.TensorFormat.LINEAR
                        or tuple(engine.get_binding_shape(binding)) != shape
                        or engine.get_binding_dtype(binding) != boundary_dtype):
                    raise RuntimeError("Unexpected boundary shape, dtype or non-LINEAR format")
                pointer = ct.c_void_p()
                cuda.call("cudaMalloc", ct.byref(pointer), array.nbytes)
                pointers.append(pointer)
                bindings[binding] = pointer.value
            cuda.call("cudaStreamCreate", ct.byref(stream))
            for _ in range(2):
                event = ct.c_void_p()
                cuda.call("cudaEventCreate", ct.byref(event))
                events.append(event)
            cuda.call("cudaMemcpy", pointers[0], host_input.ctypes.data, host_input.nbytes, 1)

            def enqueue(count):
                for _ in range(count):
                    if not context.execute_async_v2(bindings=bindings, stream_handle=stream.value):
                        raise RuntimeError("TensorRT execution failed")

            def verify(previous=None):
                cuda.call("cudaMemcpy", host_output.ctypes.data, pointers[1], host_output.nbytes, 2)
                tolerance = 2.0 / 127 if target == "gpu" else 2.0
                difference = float(np.max(np.abs(host_output.astype(np.float32) - input_value)))
                if not np.isfinite(host_output).all() or difference > tolerance:
                    raise RuntimeError("Analytic constant-input check failed: max error %g" % difference)
                if previous is not None and not np.array_equal(previous, host_output):
                    raise RuntimeError("Repeated output is not bitwise stable")
                return difference

            guard.set_stage("warmup_and_verification", 10)
            enqueue(3)
            cuda.call("cudaStreamSynchronize", stream)
            max_error = verify()
            reference_output = host_output.copy()
            enqueue(2)
            cuda.call("cudaStreamSynchronize", stream)
            max_error = max(max_error, verify(reference_output))
            guard.set_stage("benchmark", min(30, seconds + 1))
            started = time.monotonic()
            count, wall_total, event_total, windows = 0, 0.0, 0.0, 0
            per_iteration = None
            while True:
                remaining = seconds - (time.monotonic() - started)
                if remaining <= max(0.02, (per_iteration or 0.0) * 2):
                    break
                iterations = window_iterations
                if per_iteration is not None:
                    iterations = min(iterations, max(1, int(remaining * 0.7 / per_iteration)))
                wall_start = time.monotonic()
                cuda.call("cudaEventRecord", events[0], stream)
                enqueue(iterations)
                cuda.call("cudaEventRecord", events[1], stream)
                cuda.call("cudaEventSynchronize", events[1])
                wall_seconds = time.monotonic() - wall_start
                event_ms = ct.c_float()
                cuda.call("cudaEventElapsedTime", ct.byref(event_ms), events[0], events[1])
                event_seconds = event_ms.value / 1000.0
                if wall_seconds <= 0 or event_seconds <= 0:
                    raise RuntimeError("Invalid non-positive timing result")
                windows += 1
                count += iterations
                wall_total += wall_seconds
                event_total += event_seconds
                per_iteration = wall_seconds / iterations
                emit(kind="window", target=target, index=windows, iterations=iterations,
                     wall_seconds=wall_seconds, cuda_event_seconds=event_seconds,
                     wall_dense_equivalent_tops=iterations * nominal_ops / wall_seconds / 1e12,
                     cuda_event_dense_equivalent_tops=iterations * nominal_ops / event_seconds / 1e12,
                     temperature_c=guard.current)
            elapsed = time.monotonic() - started
            guard.set_stage("final_verification_and_cleanup", 10)
            if not count:
                raise RuntimeError("No measured iterations")
            max_error = max(max_error, verify(reference_output))
            final_temperature = read_temperature()
            emit(kind="summary", target=target, result="PASS", meaning="synthetic execution and limited numerical checks passed; not 100 TOPS certification",
                 precision_evidence=precision_evidence, sparse_weights=sparse,
                 requested_seconds=seconds, benchmark_elapsed_seconds=elapsed,
                 measured_iterations=count, windows=windows, shape=shape, nominal_ops=nominal_ops,
                 wall_seconds=wall_total, cuda_event_seconds=event_total,
                 primary_timing="host wall clock around enqueue and completion",
                 wall_dense_equivalent_tops=count * nominal_ops / wall_total / 1e12,
                 cuda_event_dense_equivalent_tops=count * nominal_ops / event_total / 1e12,
                 end_to_end_dense_equivalent_tops=count * nominal_ops / elapsed / 1e12,
                 cuda_event_note="DLA event timing is auxiliary; use wall timing for DLA",
                 excluded="build, warmup, H2D, D2H, numerical checks and JSON logging excluded from summed timing windows",
                 reference="row-normalized weights preserve constant input 64/127 (raw INT8 value 64)",
                 max_abs_error=max_error, error_units="real value" if target == "gpu" else "raw INT8 units",
                 reference_checks=3, bitwise_stable=True, finite_outputs=True,
                 peak_thermal_c=max(guard.peak, final_temperature), build_seconds=build_seconds)
        finally:
            # Best-effort cleanup must also run when verification fails.
            if stream.value:
                cuda.lib.cudaStreamSynchronize(stream)
            for event in events:
                cuda.lib.cudaEventDestroy(event)
            if stream.value:
                cuda.lib.cudaStreamDestroy(stream)
            for pointer in pointers:
                cuda.lib.cudaFree(pointer)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("all", "gpu", "dla0", "dla1"), default="all")
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--spatial", type=int, choices=(32, 64), default=32,
                        help="square spatial extent; 64 requires --target gpu")
    parser.add_argument("--window-iterations", type=int, default=100)
    parser.add_argument("--sparse", action="store_true", help="request 2:4 weight tactics; requires --target gpu")
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 30:
        parser.error("--seconds must be between 1 and 30")
    if not 1 <= args.window_iterations <= 100:
        parser.error("--window-iterations must be between 1 and 100")
    if args.sparse and args.target != "gpu":
        parser.error("--sparse requires --target gpu")
    if args.spatial == 64 and args.target != "gpu":
        parser.error("--spatial 64 requires --target gpu")
    return args


def main():
    args = parse_args()
    try:
        for target in ("gpu", "dla0", "dla1"):
            if args.target in ("all", target):
                benchmark(target, args.seconds, args.sparse, args.window_iterations, args.spatial)
    except Exception as error:
        emit(kind="error", result="FAIL", reason=str(error))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
