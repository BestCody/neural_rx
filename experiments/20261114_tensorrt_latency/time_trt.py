"""Build TensorRT engines from build/*.onnx and time them on one GPU, as in Wiesmayr et al. (2025).

As in their code release (scripts/export_onnx.py, trtexec --fp16): FP16 engines with fixed shapes and batch size 1.
Inputs stay on the GPU; the time is the GPU compute time of one inference, measured with CUDA events around
execute_async_v3, without host-device copies. The engines are timed in interleaved rounds so that clock drift affects
all of them alike. Each engine's output is checked against the TensorFlow reference of export.py first.

Runs in the TensorRT venv (TensorRT is not in the neural_rx env):

    ~/venvs/tensorrt/bin/python time_trt.py [--gpu 0] [--precision fp16 fp32]
"""
import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE/"build"
NETWORKS = ("large_k2", "large_k3", "large_k8", "proposed")


def check(error):
    if int(error) != 0:
        raise RuntimeError(f"CUDA error {error}")


def call(result):
    check(result[0])
    return result[1] if len(result) == 2 else result[1:]


FORCED = {}
CGNN_SCOPES = ("state_init", "cgnn_it", "readout_ll_rs")


def filter_layers_to_fp32(trt, network):
    """Keep the channel filter in FP32 when the engine is FP16. Its floors (about 1e-7 of the channel power) flush to
    zero in FP16 and the log of the power of masked neighbors becomes -inf. Filter layers are those that do not depend
    on the received slot and lie outside the CGNN layers (StateInit, iterations, readout); the CGNN, including the input
    normalization, stays FP16 as for the Large NRX. Returns the number of layers constrained."""
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
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        forced = filter_layers_to_fp32(trt, network)
        if forced:
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            FORCED[network_name] = forced
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"engine build failed for {network_name} ({precision})")
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

    def output(self):
        (pointer, shape, dtype), = self.outputs.values()
        host = self.np.empty(shape, dtype)
        call(self.cudart.cudaMemcpy(host.ctypes.data, pointer, host.nbytes,
                                    self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        return host

    def enqueue(self):
        """Everything one inference puts on the stream (time_fused.py adds the filter kernel before the engine)."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--precision", nargs="+", default=["fp16", "fp32"], choices=("fp16", "fp32"))
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--per-round", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=200)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import numpy as np
    import tensorrt as trt
    from cuda import cudart

    reference = np.load(BUILD/"reference.npz")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    gpu = call(cudart.cudaGetDeviceProperties(0)).name.decode().rstrip("\x00")
    report = {"gpu": gpu, "tensorrt": trt.__version__, "batch_size": 1,
              "timing": {"rounds": args.rounds, "per_round": args.per_round, "warmup": args.warmup,
                         "measure": "CUDA events around execute_async_v3, inputs on the GPU"}}
    for precision in args.precision:
        engines = {}
        for name in NETWORKS:
            feeds = {key.split("/")[-1]: reference[key] for key in reference.files
                     if key.startswith(f"{name}/input/")}
            engine = Engine(trt, cudart, np, runtime, build(trt, logger, name, precision), feeds)
            engine.run()
            output, expected = engine.output().astype(np.float32), reference[f"{name}/output"]
            confident = np.abs(expected) > 1.
            engines[name] = engine
            report.setdefault(precision, {})[name] = {"check": {
                "llr_max_abs_diff": float(np.max(np.abs(output-expected))),
                "llr_max_abs": float(np.max(np.abs(expected))),
                "sign_agreement": float(np.mean(np.sign(output) == np.sign(expected))),
                "sign_agreement_abs_llr_above_1": float(np.mean(np.sign(output[confident]) == np.sign(expected[confident])))}}
            if not np.all(np.isfinite(output)):
                raise RuntimeError(f"{name} ({precision}): non-finite output")
        for engine in engines.values():
            engine.time(args.warmup)
        samples = {name: [] for name in engines}
        for _ in range(args.rounds):
            for name, engine in engines.items():
                samples[name] += engine.time(args.per_round)
        graphs = {name: [] for name in engines}
        for engine in engines.values():
            engine.capture()
            engine.time_graph(args.warmup)
        for _ in range(args.rounds):
            for name, engine in engines.items():
                graphs[name] += engine.time_graph(args.per_round)
        for name, values in graphs.items():
            report[precision][name]["median_ms_cuda_graph"] = float(np.median(values))
        for name, values in samples.items():
            report[precision][name]["median_ms"] = float(np.median(values))
            report[precision][name]["mean_ms"] = float(np.mean(values))
            report[precision][name]["samples"] = len(values)
        np.savez(HERE/"results"/f"timing_{precision}.npz", **{n: np.asarray(v) for n, v in samples.items()})
        k2, k8 = (report[precision][n]["median_ms"] for n in ("large_k2", "large_k8"))
        per_iteration = (k8-k2)/6
        report[precision]["derived"] = {
            "per_cgnn_iteration_ms": per_iteration, "fixed_ms": k2-2*per_iteration, "k8_over_k2": k8/k2,
            "proposed_minus_k2_ms": report[precision]["proposed"]["median_ms"]-k2,
            "proposed_minus_k2_in_iterations": (report[precision]["proposed"]["median_ms"]-k2)/per_iteration,
            "proposed_over_k2": report[precision]["proposed"]["median_ms"]/k2}
        g2, g8, gp = (report[precision][n]["median_ms_cuda_graph"] for n in ("large_k2", "large_k8", "proposed"))
        report[precision]["derived_cuda_graph"] = {
            "per_cgnn_iteration_ms": (g8-g2)/6, "fixed_ms": g2-2*(g8-g2)/6, "k8_over_k2": g8/g2,
            "proposed_minus_k2_ms": gp-g2, "proposed_minus_k2_in_iterations": (gp-g2)/((g8-g2)/6),
            "proposed_over_k2": gp/g2}
    report["fp16_layers_forced_to_fp32"] = FORCED
    (HERE/"results").mkdir(exist_ok=True)
    (HERE/"results/timing.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
