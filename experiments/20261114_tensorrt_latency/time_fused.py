"""Time the NRXs with TensorRT, and the proposed NRX with its channel filter as CUDA kernels (fitted_filter.cu).

The proposed NRX is the kernels that apply the filter (filter_predict, filter_expand), which write the
refined channel into the input buffer of the TensorRT engine of the two CGNN iterations (build/proposed_cgnn.onnx,
FP16 like the other engines), followed by that engine, all on one CUDA stream. The filter's weights and the part of
its predictions that comes from the memory only depend on earlier slots; they are computed before the slot
(filter_gram, filter_solve, filter_history, "fit_only"). Two further measurements include that fit:
  proposed_and_fit     the slot's filter and network, then the fit for the next slot, one after the other on one
                       stream: all GPU work per slot
  proposed_overlapped  the slot's filter and network while the fit for the next slot runs on a second stream, with
                       its own buffers (as with double buffering): the slot's latency under that contention

Checks: the kernels' refined channel against the TensorFlow receiver (export.py; the buffer is filled with NaN first,
so every grid position must be written), and the LLRs of kernels + engine against the TensorFlow LLRs. Timing as in
time_trt.py: CUDA events, interleaved rounds, median, together with the Large NRX (N_it = 2, 3, 8). The same is then
timed with each receiver's inference captured as a CUDA graph ("median_ms_cuda_graph"), which takes the host's time to
launch the kernels out of the measurement; it shows how much of a time is the host's processor and not the GPU.

    ~/venvs/tensorrt/bin/python time_fused.py [--gpu 0]     # results/fused.json, results/fused_samples.npz

--channels-first times the engines of export.py --channels-first instead and writes results/fused_cf.json.
--tag NAME adds a suffix to the result files (e.g. the GPU), --build DIR reads the exported files from another folder.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import time_trt
from time_trt import FORCED, Engine, build, call
from filter_kernels import FilterKernels

HERE = Path(__file__).resolve().parent
A100_K2_MS = .97     # Large NRX with two iterations on the A100 in Wiesmayr et al. (2025)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--per-round", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--channels-first", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--build", type=Path, default=HERE/"build")
    args = parser.parse_args()
    suffix = "_cf" if args.channels_first else ""
    tag = f"_{args.tag}" if args.tag else ""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    time_trt.BUILD = args.build                     # where the ONNX files are and the engine plans are kept
    import numpy as np
    import tensorrt as trt
    from cuda import cuda, cudart

    properties = call(cudart.cudaGetDeviceProperties(0))
    architecture = f"sm_{properties.major}{properties.minor}"
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    cubin = args.build/f"fitted_filter_{architecture}.cubin"
    if Path(nvcc).is_file():
        subprocess.run([nvcc, "-cubin", f"-arch={architecture}", "-O3", "-o", str(cubin), str(HERE/"fitted_filter.cu")],
                       check=True)
    elif not cubin.is_file():                       # a machine without the compiler needs the compiled kernels
        raise FileNotFoundError(f"no nvcc and no {cubin}")
    reference = np.load(args.build/f"reference{suffix}.npz")
    tables = dict(np.load(args.build/f"filter_tables{suffix}.npz"))
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    def feeds(name):
        return {key.split("/")[-1]: reference[key] for key in reference.files
                if key.startswith(f"{name}{suffix}/input/")}

    class Proposed(Engine):
        """Filter kernels, then the engine of the two CGNN iterations, on one stream."""
        def __init__(self, part="all"):
            super().__init__(trt, cudart, np, runtime, build(trt, logger, f"proposed_cgnn{suffix}", "fp16"),
                             feeds("proposed_cgnn"))
            self.part = part
            self.kernels = FilterKernels(np, cuda, cudart, cubin.read_bytes(), tables, feeds("filter"),
                                         self.addresses["channel"], self.stream)
            self.kernels.fit()                      # what the memory gives the slot, as it exists when it arrives
            call(cudart.cudaStreamSynchronize(self.stream))
            if part == "overlap":                   # the next slot's fit: own stream, own buffers
                self.side = call(cudart.cudaStreamCreate())
                self.next = FilterKernels(np, cuda, cudart, cubin.read_bytes(), tables, feeds("filter"),
                                          self.addresses["channel"], self.side)
                flags = cudart.cudaEventDisableTiming
                self.fork, self.join = (call(cudart.cudaEventCreateWithFlags(flags)) for _ in range(2))

        def enqueue(self):
            if self.part == "fit":
                self.kernels.fit()
                return True
            if self.part == "overlap":              # fork the next slot's fit onto the side stream, join at the end
                call(cudart.cudaEventRecord(self.fork, self.stream))
                call(cudart.cudaStreamWaitEvent(self.side, self.fork, 0))
                self.next.fit()
                call(cudart.cudaEventRecord(self.join, self.side))
            self.kernels.apply()
            if self.part == "filter":
                return True
            done = self.context.execute_async_v3(self.stream)
            if self.part == "total":
                self.kernels.fit()                  # recomputes the same weights from the same memory
            if self.part == "overlap":
                call(cudart.cudaStreamWaitEvent(self.stream, self.join, 0))
            return done

    engines = {name: Engine(trt, cudart, np, runtime, build(trt, logger, name+suffix, "fp16"), feeds(name))
               for name in ("large_k2", "large_k3", "large_k8")}
    engines["proposed"] = Proposed()
    engines["filter_only"] = Proposed("filter")
    engines["fit_only"] = Proposed("fit")
    engines["proposed_and_fit"] = Proposed("total")
    engines["proposed_overlapped"] = Proposed("overlap")
    report = {"gpu": properties.name.decode().rstrip("\x00"), "tensorrt": trt.__version__, "precision": "fp16",
              "export": "channels first" if args.channels_first else "as Wiesmayr et al.",
              "kernels": {"architecture": architecture,
                          "launches": {k: [list(v[0]), list(v[1])] for k, v in engines["proposed"].kernels.launches.items()}}}

    # Check: refined channel and LLRs. The engine's channel buffer starts as NaN, so the kernels must fill all of it.
    proposed = engines["proposed"]
    channel = proposed.addresses["channel"]
    expected = tables["reference_channel"]
    call(cudart.cudaMemset(channel, 0xFF, expected.nbytes))
    proposed.run()
    refined = np.empty_like(expected)
    call(cudart.cudaMemcpy(refined.ctypes.data, channel, expected.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
    llr, llr_expected = proposed.output().astype(np.float32), reference[f"proposed_cgnn{suffix}/output"]
    confident = np.abs(llr_expected) > 1.
    report["check"] = {
        "channel_all_finite": bool(np.all(np.isfinite(refined))),
        "channel_max_abs_diff": float(np.max(np.abs(refined-expected))), "channel_max_abs": float(np.max(np.abs(expected))),
        "channel_relative_error_db": float(10*np.log10(np.sum((refined-expected)**2)/np.sum(expected**2))),
        "entries_chosen": proposed.kernels.read("choice").tolist(), "gain": proposed.kernels.read("gain").round(4).tolist(),
        "llr_max_abs_diff": float(np.max(np.abs(llr-llr_expected))), "llr_max_abs": float(np.max(np.abs(llr_expected))),
        "sign_agreement": float(np.mean(np.sign(llr) == np.sign(llr_expected))),
        "sign_agreement_abs_llr_above_1": float(np.mean(np.sign(llr[confident]) == np.sign(llr_expected[confident])))}
    for name in ("large_k2", "large_k3", "large_k8"):
        engine = engines[name]
        engine.run()
        output, wanted = engine.output().astype(np.float32), reference[f"{name}{suffix}/output"]
        report.setdefault("engine_checks", {})[name] = float(np.mean(
            np.sign(output[np.abs(wanted) > 1.]) == np.sign(wanted[np.abs(wanted) > 1.])))
    print(json.dumps(report["check"], indent=1), flush=True)
    if not (report["check"]["channel_all_finite"]
            and report["check"]["channel_max_abs_diff"] <= 5e-3*report["check"]["channel_max_abs"]):
        raise RuntimeError(f"the kernels' refined channel differs from the receiver: {report['check']}")

    for engine in engines.values():
        engine.time(args.warmup)
    events = {name: [] for name in engines}
    for _ in range(args.rounds):
        for name, engine in engines.items():
            events[name] += engine.time(args.per_round)
    graphs = {name: [] for name in engines}
    for engine in engines.values():
        engine.capture()
        engine.time_graph(args.warmup)
    for _ in range(args.rounds):
        for name, engine in engines.items():
            graphs[name] += engine.time_graph(args.per_round)
    (HERE/"results").mkdir(exist_ok=True)
    np.savez(HERE/f"results/fused{suffix}{tag}_samples.npz", **{f"events_{n}": np.asarray(v) for n, v in events.items()},
             **{f"graph_{n}": np.asarray(v) for n, v in graphs.items()})
    values = report["median_ms"] = {name: float(np.median(v)) for name, v in events.items()}
    report["median_ms_cuda_graph"] = {name: float(np.median(v)) for name, v in graphs.items()}
    per_iteration = (values["large_k8"]-values["large_k2"])/6
    report["derived_median_ms"] = {
        "per_cgnn_iteration_ms": per_iteration, "proposed_minus_k2_ms": values["proposed"]-values["large_k2"],
        "proposed_and_fit_minus_k2_ms": values["proposed_and_fit"]-values["large_k2"],
        "overlap_slowdown_ms": values["proposed_overlapped"]-values["proposed"],
        "proposed_over_k2": values["proposed"]/values["large_k2"], "k3_minus_proposed_ms": values["large_k3"]-values["proposed"],
        "k8_over_k2": values["large_k8"]/values["large_k2"]}
    if "A100" not in report["gpu"]:
        # this GPU's times scaled by the ratio of the two GPUs for the Large NRX with N_it = 2 (not a measurement)
        ratio = A100_K2_MS/values["large_k2"]
        report["scaled_to_a100_ms"] = {name: ratio*values[name] for name in ("large_k2", "large_k3", "large_k8", "proposed",
                                                                            "proposed_and_fit", "proposed_overlapped")}
    report["fp16_layers_forced_to_fp32"] = FORCED
    report["samples_per_receiver"] = args.rounds*args.per_round
    (HERE/f"results/fused{suffix}{tag}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("gpu", "median_ms", "median_ms_cuda_graph", "derived_median_ms") if k in report}
                     | ({"scaled_to_a100_ms": report["scaled_to_a100_ms"]} if "scaled_to_a100_ms" in report else {}), indent=1))


if __name__ == "__main__":
    main()
