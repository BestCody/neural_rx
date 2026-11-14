"""Time the proposed NRX without IDD with its channel filter as one hand-written CUDA kernel (pilot_filter.cu).

The kernel runs in the CUDA stream of the TensorRT engine of the two CGNN iterations (build/proposed_cgnn.onnx, FP16
like the other engines), right before it, and writes the refined channel into the engine's input buffer. A TensorRT
plugin would run the same kernel inside the engine; the pip TensorRT has no C++ headers to build one, and for the
CUDA-graph timing both are captured into one graph.

Checks: the kernel's refined channel against the TensorFlow receiver (export.py; the buffer is filled with NaN first, so
every grid position must be written), and the LLRs of kernel + engine against the TensorFlow LLRs. Timing as in
time_trt.py: CUDA events, interleaved rounds, with and without CUDA graphs, together with the Large NRX (N_it = 2, 3, 8)
and the proposed NRX with the TensorRT-compiled filter.

    ~/venvs/tensorrt/bin/python time_fused.py [--gpu 0]     # results/fused.json, results/fused_samples.npz
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from time_trt import FORCED, Engine, build, call

HERE = Path(__file__).resolve().parent
BUILD = HERE/"build"
NVCC = "/usr/local/cuda/bin/nvcc"
THREADS = 64


class FilterKernel:
    def __init__(self, np, cuda, cudart, cubin, tables, feeds, output, stream):
        self.cuda = cuda
        self.memory = []
        call(cuda.cuInit(0))
        self.module = call(cuda.cuModuleLoadData(cubin))
        self.function = call(cuda.cuModuleGetFunction(self.module, b"pilot_filter"))
        users, cells = tables["pack"].shape
        positions = tables["phi"].shape[1]
        if (feeds["memory_observations"].shape != (1, users, 7, cells, 9)
                or tables["neighbor_index"].shape != (users, users, cells, 6) or tables["weights"].size != 343
                or feeds["h_hat_ls"].shape[1] != users or np.prod(feeds["h_hat_ls"].shape[2:4]) != positions):
            raise ValueError("tables and inputs do not match the kernel's dimensions")

        def upload(array, dtype):
            array = np.ascontiguousarray(array, dtype)
            pointer = call(cudart.cudaMalloc(array.nbytes))
            call(cudart.cudaMemcpy(pointer, array.ctypes.data, array.nbytes,
                                   cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
            self.memory.append(pointer)
            return pointer

        f32, i32 = np.float32, np.int32
        pointers = [upload(feeds["h_hat_ls"], f32), upload(feeds["noise_variance"], f32),
                    upload(feeds["memory_observations"], f32), upload(feeds["memory_age"], i32),
                    upload(feeds["memory_ports"], i32), upload(feeds["memory_valid"], f32), upload(feeds["active"], f32),
                    upload(tables["pack"], i32), upload(tables["noise_factors"], f32),
                    upload(tables["neighbor_index"], i32), upload(tables["neighbor_valid"], np.uint8),
                    upload(tables["neighbor_delta"], f32), upload(tables["geometry"], f32),
                    upload(tables["weights"], f32), upload(tables["phi"], f32), upload(tables["cell_offsets"], i32),
                    upload(tables["cell_positions"], i32), output]
        self.args = ([np.array([int(p)], np.uint64) for p in pointers]
                     + [np.array([users], i32), np.array([cells], i32), np.array([positions], i32),
                        np.array([tables["tb_ms"]], f32)])
        self.params = np.array([a.ctypes.data for a in self.args], np.uint64)
        self.blocks = users*cells
        self.stream = cuda.CUstream(int(stream))

    def launch(self):
        call(self.cuda.cuLaunchKernel(self.function, self.blocks, 1, 1, THREADS, 1, 1, 0, self.stream,
                                      self.params.ctypes.data, 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--per-round", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=200)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import numpy as np
    import tensorrt as trt
    from cuda import cuda, cudart

    properties = call(cudart.cudaGetDeviceProperties(0))
    architecture = f"sm_{properties.major}{properties.minor}"
    subprocess.run([NVCC, "-cubin", f"-arch={architecture}", "-O3", "-o", str(BUILD/"pilot_filter.cubin"),
                    str(HERE/"pilot_filter.cu")], check=True)
    cubin = (BUILD/"pilot_filter.cubin").read_bytes()
    reference = np.load(BUILD/"reference.npz")
    tables = dict(np.load(BUILD/"filter_tables.npz"))
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    def feeds(name):
        return {key.split("/")[-1]: reference[key] for key in reference.files if key.startswith(f"{name}/input/")}

    class Pipeline(Engine):
        """Filter kernel, then the CGNN engine, on one stream."""
        def __init__(self, kernel_only=False):
            super().__init__(trt, cudart, np, runtime, build(trt, logger, "proposed_cgnn", "fp16"),
                             feeds("proposed_cgnn"))
            self.kernel_only = kernel_only
            self.kernel = FilterKernel(np, cuda, cudart, cubin, tables, feeds("proposed"), self.addresses["channel"],
                                       self.stream)

        def enqueue(self):
            self.kernel.launch()
            return True if self.kernel_only else self.context.execute_async_v3(self.stream)

    engines = {name: Engine(trt, cudart, np, runtime, build(trt, logger, name, "fp16"), feeds(name))
               for name in ("large_k2", "large_k3", "large_k8", "proposed", "proposed_cgnn")}
    engines["fused"] = Pipeline()
    engines["fused_kernel_only"] = Pipeline(kernel_only=True)
    report = {"gpu": properties.name.decode().rstrip("\x00"), "tensorrt": trt.__version__, "precision": "fp16",
              "kernel": {"architecture": architecture, "blocks": int(engines["fused"].kernel.blocks),
                         "threads": THREADS}}

    # Check: refined channel and LLRs. The engine's channel buffer starts as NaN, so the kernel must fill all of it.
    fused = engines["fused"]
    channel = fused.addresses["channel"]
    size = tables["reference_channel"].nbytes
    call(cudart.cudaMemset(channel, 0xFF, size))
    fused.run()
    refined = np.empty_like(tables["reference_channel"])
    call(cudart.cudaMemcpy(refined.ctypes.data, channel, size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
    expected = tables["reference_channel"]
    llr, llr_expected = fused.output().astype(np.float32), reference["proposed/output"]
    confident = np.abs(llr_expected) > 1.
    report["check"] = {
        "channel_all_finite": bool(np.all(np.isfinite(refined))),
        "channel_max_abs_diff": float(np.max(np.abs(refined-expected))), "channel_max_abs": float(np.max(np.abs(expected))),
        "llr_max_abs_diff": float(np.max(np.abs(llr-llr_expected))), "llr_max_abs": float(np.max(np.abs(llr_expected))),
        "sign_agreement": float(np.mean(np.sign(llr) == np.sign(llr_expected))),
        "sign_agreement_abs_llr_above_1": float(np.mean(np.sign(llr[confident]) == np.sign(llr_expected[confident])))}
    for name in ("large_k2", "large_k3", "large_k8", "proposed", "proposed_cgnn"):
        engine = engines[name]
        engine.run()
        output, wanted = engine.output().astype(np.float32), reference[f"{name}/output"]
        report.setdefault("engine_checks", {})[name] = float(np.mean(
            np.sign(output[np.abs(wanted) > 1.]) == np.sign(wanted[np.abs(wanted) > 1.])))
    if not (report["check"]["channel_all_finite"]
            and report["check"]["channel_max_abs_diff"] <= 1e-4*report["check"]["channel_max_abs"]):
        raise RuntimeError(f"the kernel's refined channel differs from the receiver: {report['check']}")

    for engine in engines.values():
        engine.time(args.warmup)
    events = {name: [] for name in engines}
    for _ in range(args.rounds):
        for name, engine in engines.items():
            events[name] += engine.time(args.per_round)
    for engine in engines.values():
        engine.capture()
        engine.time_graph(args.warmup)
    graphs = {name: [] for name in engines}
    for _ in range(args.rounds):
        for name, engine in engines.items():
            graphs[name] += engine.time_graph(args.per_round)
    np.savez(HERE/"results/fused_samples.npz", **{f"events_{n}": np.asarray(v) for n, v in events.items()},
             **{f"graph_{n}": np.asarray(v) for n, v in graphs.items()})
    report["median_ms"] = {name: float(np.median(v)) for name, v in events.items()}
    report["median_ms_cuda_graph"] = {name: float(np.median(v)) for name, v in graphs.items()}
    for key in ("median_ms", "median_ms_cuda_graph"):
        values = report[key]
        per_iteration = (values["large_k8"]-values["large_k2"])/6
        report[f"derived_{key}"] = {
            "per_cgnn_iteration_ms": per_iteration,
            "fused_minus_k2_ms": values["fused"]-values["large_k2"],
            "fused_minus_k2_in_iterations": (values["fused"]-values["large_k2"])/per_iteration,
            "fused_over_k2": values["fused"]/values["large_k2"],
            "trt_filter_minus_k2_ms": values["proposed"]-values["large_k2"],
            "kernel_in_iterations": values["fused_kernel_only"]/per_iteration}
    report["fp16_layers_forced_to_fp32"] = FORCED
    report["samples_per_receiver"] = args.rounds*args.per_round
    (HERE/"results/fused.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
