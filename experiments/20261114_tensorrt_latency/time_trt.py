"""TensorRT engines for time_fused.py: built from build/*.onnx and timed on one GPU, as in Wiesmayr et al. (2025).

As in their code release (scripts/export_onnx.py, trtexec --fp16): FP16 engines with fixed shapes and batch size 1.
Inputs stay on the GPU; the time is the GPU compute time of one inference, measured with CUDA events around
execute_async_v3, without host-device copies. Needs the TensorRT venv (TensorRT is not in the neural_rx env).
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE/"build"


def check(error):
    if int(error) != 0:
        raise RuntimeError(f"CUDA error {error}")


def call(result):
    check(result[0])
    return result[1] if len(result) == 2 else result[1:]


FORCED = {}
CGNN_SCOPES = ("state_init", "cgnn_it", "readout_ll_rs")


def filter_layers_to_fp32(trt, network):
    """Keep in FP32, in every FP16 engine alike, the layers that do not depend on the received slot and lie outside the
    CGNN layers (StateInit, iterations, readout): what the network does to its other inputs before the CGNN. The CGNN,
    including the input normalization, stays FP16. Returns the number of layers constrained."""
    received = {network.get_input(i).name for i in range(network.num_inputs)} & {"rx_slot"}
    dependent, forced = set(received), 0
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        inputs = [layer.get_input(i) for i in range(layer.num_inputs)]
        outputs = [layer.get_output(i) for i in range(layer.num_outputs)]
        if any(t is not None and t.name in dependent for t in inputs):
            dependent.update(t.name for t in outputs)
            continue
        if layer.name.startswith(CGNN_SCOPES):
            continue
        if outputs and all(t.dtype in (trt.float32, trt.float16) for t in outputs):
            layer.precision = trt.float32
            for i in range(layer.num_outputs):
                layer.set_output_type(i, trt.float32)
            forced += 1
    return forced


def build(trt, logger, network_name, precision):
    plan = BUILD/f"{network_name}_{precision}.plan"
    if plan.is_file():
        return plan.read_bytes()
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(BUILD/f"{network_name}.onnx")):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    # One timing cache for all engines of a build folder: layers that the networks have in common get the same
    # implementation in each engine, so the engines differ only where the networks do.
    timings = BUILD/f"timing_{precision}.cache"
    config.set_timing_cache(config.create_timing_cache(timings.read_bytes() if timings.is_file() else b""), False)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        forced = filter_layers_to_fp32(trt, network)
        if forced:
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            FORCED[network_name] = forced
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"engine build failed for {network_name} ({precision})")
    timings.write_bytes(bytes(config.get_timing_cache().serialize()))
    plan.write_bytes(bytes(serialized))
    return bytes(serialized)


class Engine:
    def __init__(self, trt, cudart, np, runtime, serialized, feeds):
        self.cudart, self.np = cudart, np
        self.engine = runtime.deserialize_cuda_engine(serialized)
        self.context = self.engine.create_execution_context()
        self.buffers, self.outputs, self.addresses = [], {}, {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            pointer = call(cudart.cudaMalloc(max(1, int(np.prod(shape))*dtype.itemsize)))
            self.buffers.append(pointer)
            self.addresses[name.split(":")[0]] = pointer
            self.context.set_tensor_address(name, pointer)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                value = np.ascontiguousarray(feeds[name.split(":")[0]], dtype)
                if value.shape != shape:
                    raise ValueError(f"{name}: input {value.shape} vs engine {shape}")
                call(cudart.cudaMemcpy(pointer, value.ctypes.data, value.nbytes,
                                       cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
            else:
                self.outputs[name] = (pointer, shape, dtype)
        self.stream = call(cudart.cudaStreamCreate())
        self.start, self.end = call(cudart.cudaEventCreate()), call(cudart.cudaEventCreate())

    def output(self, index=0):
        """The engine's output: the LLRs."""
        pointer, shape, dtype = list(self.outputs.values())[index]
        host = self.np.empty(shape, dtype)
        call(self.cudart.cudaMemcpy(host.ctypes.data, pointer, host.nbytes,
                                    self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        return host

    def enqueue(self):
        """Everything one inference puts on the stream (time_fused.py adds the filter kernels before the engine)."""
        return self.context.execute_async_v3(self.stream)

    def run(self):
        if not self.enqueue():
            raise RuntimeError("inference failed")
        call(self.cudart.cudaStreamSynchronize(self.stream))

    def capture(self):
        """Capture one inference as a CUDA graph, which removes the per-kernel launch overhead of the host."""
        cudart = self.cudart
        call(cudart.cudaStreamBeginCapture(self.stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal))
        self.enqueue()
        graph = call(cudart.cudaStreamEndCapture(self.stream))
        self.graph = call(cudart.cudaGraphInstantiate(graph, 0))

    def time_graph(self, count):
        cudart = self.cudart
        times = []
        for _ in range(count):
            call(cudart.cudaEventRecord(self.start, self.stream))
            call(cudart.cudaGraphLaunch(self.graph, self.stream))
            call(cudart.cudaEventRecord(self.end, self.stream))
            call(cudart.cudaEventSynchronize(self.end))
            times.append(call(cudart.cudaEventElapsedTime(self.start, self.end)))
        return times

    def time(self, count):
        cudart = self.cudart
        times = []
        for _ in range(count):
            call(cudart.cudaEventRecord(self.start, self.stream))
            self.enqueue()
            call(cudart.cudaEventRecord(self.end, self.stream))
            call(cudart.cudaEventSynchronize(self.end))
            times.append(call(cudart.cudaEventElapsedTime(self.start, self.end)))
        return times
