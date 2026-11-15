"""Can IDD's decoder-independent part run next to the network without delaying it?

Only IDD's correction kernel needs the decoder's output (time_idd.py: decoder_dependent_us_per_slot). The pair preparation
and the symbol-pair scores need only the received signal and the channel filter's output, so they can start right after
the filter and run on a second CUDA stream while the two CGNN iterations run on the first. This script measures that on
one GPU: the network alone (filter kernel + TensorRT engine of the CGNN, as in time_fused.py), the two IDD kernels
alone, and both together, with the IDD stream waiting for the filter kernel.

The IDD kernels are the receiver's own: their device code is cut from src/neural_rx/receiver/native_ops/*.cu.cc and
compiled without the TensorFlow wrappers. Their outputs are checked against TensorFlow's (build/idd_kernels.npz, written
by time_idd.py). The error-covariance kernels (about 0.05 ms) and TensorFlow's glue operations are not included.

    ~/venvs/tensorrt/bin/python time_overlap.py [--gpu 0]     # results/overlap.json, results/overlap_samples.npz
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from time_fused import NVCC, FilterKernel
from time_trt import Engine, build, call

HERE = Path(__file__).resolve().parent
BUILD = HERE/"build"
NATIVE = HERE.parents[1]/"src/neural_rx/receiver/native_ops"


def kernel_source():
    """Device code of FusedPrepare and ScoresWarp, as in the receiver's native operations."""
    correction = (NATIVE/"pair_correction.cu.cc").read_text()
    scores = (correction[correction.index("__device__ C add("):correction.index("size_t ScalarWorkspace(")]
              + correction[correction.index("#define WARP_RES 4"):correction.index("__global__ void CorrectionWarp(")])
    prepare = (NATIVE/"pair_prepare.cu.cc").read_text()
    helpers = prepare[prepare.index("__device__ float squared("):prepare.index("// The whole preparation in one kernel")]
    fused = prepare[prepare.index("#define FUSED_ANTENNAS 8"):prepare.index("Status FusedPreparation(")]
    return ("#include <cuda_runtime.h>\n#include <cmath>\n#include <cstdint>\n#include <cstddef>\nusing C=float2;\n"
            + scores + helpers + fused)


class IddKernels:
    """Pair preparation, then symbol-pair scores, on one stream."""
    def __init__(self, np, cuda, cudart, cubin, names, data, stream):
        self.np, self.cuda, self.cudart = np, cuda, cudart
        module = call(cuda.cuModuleLoadData(cubin))
        self.prepare = call(cuda.cuModuleGetFunction(module, names["FusedPrepare"].encode()))
        self.scores = call(cuda.cuModuleGetFunction(module, names["ScoresWarp"].encode()))
        batch, cells, users, r, _ = data["theta"].shape
        levels, n = len(data["energy_levels"]), data["y"].shape[1]
        m, q = data["labels"].shape
        groups, pairs, kbits = levels*levels, m*m, 2*q
        if batch != 1 or users != 2:
            raise ValueError("expected one slot with two UEs")
        self.shapes = {"factor": (1, cells, groups, r, r), "white": (1, cells, groups, r, 6), "logdet": (1, cells, groups),
                       "scores": (1, n, pairs+kbits), "fit": (1, n)}
        self.types = {"factor": np.complex64, "white": np.complex64, "logdet": np.float32, "scores": np.float32,
                      "fit": np.float32}

        def upload(array, dtype):
            array = np.ascontiguousarray(array, dtype)
            pointer = call(cudart.cudaMalloc(array.nbytes))
            call(cudart.cudaMemcpy(pointer, array.ctypes.data, array.nbytes,
                                   cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
            return pointer

        self.out = {k: call(cudart.cudaMalloc(int(np.prod(s))*np.dtype(self.types[k]).itemsize))
                    for k, s in self.shapes.items()}
        status = call(cudart.cudaMalloc(4))
        call(cudart.cudaMemset(status, 0, 4))
        self.status = status
        c64, f32, i64, i32 = np.complex64, np.float32, np.int64, np.int32

        def pack(values):
            arrays = [np.array([v], t) for v, t in values]
            return arrays, np.array([a.ctypes.data for a in arrays], np.uint64)

        count = cells*groups
        self.prepare_args, self.prepare_params = pack([
            (count, i32), (cells, i32), (r, i32), (levels, i32), (data["symbol_energy"], f32),
            (upload(data["theta"], c64), np.uint64), (upload(data["covariance"], c64), np.uint64),
            (upload(data["noise"], f32), np.uint64), (0, np.uint64), (upload(data["energy_levels"], f32), np.uint64),
            (self.out["factor"], np.uint64), (self.out["white"], np.uint64), (self.out["logdet"], np.uint64),
            (status, np.uint64)])
        self.prepare_blocks = (count+127)//128
        self.scores_args, self.scores_params = pack([
            (False, np.bool_), (n, i32), (n, i32), (r, i32), (cells, i32), (groups, i32), (m, i32), (q, i32),
            (False, np.bool_),
            (upload(data["y"], c64), np.uint64), (self.out["factor"], np.uint64), (self.out["white"], np.uint64),
            (self.out["logdet"], np.uint64), (upload(data["phi"], f32), np.uint64), (upload(data["cells"], i64), np.uint64),
            (upload(data["symbols"], c64), np.uint64), (upload(data["energy_indices"], i64), np.uint64),
            (upload(data["labels"], i64), np.uint64), (upload(data["enabled"], np.bool_), np.uint64),
            (self.out["scores"], np.uint64), (self.out["fit"], np.uint64)])
        self.scores_blocks = (n+3)//4  # four resource elements per block (WARP_RES)
        self.stream = cuda.CUstream(int(stream))

    def launch(self):
        cuda = self.cuda
        call(cuda.cuLaunchKernel(self.prepare, self.prepare_blocks, 1, 1, 128, 1, 1, 0, self.stream,
                                 self.prepare_params.ctypes.data, 0))
        call(cuda.cuLaunchKernel(self.scores, self.scores_blocks, 1, 1, 128, 1, 1, 0, self.stream,
                                 self.scores_params.ctypes.data, 0))

    def read(self, name):
        host = self.np.empty(self.shapes[name], self.types[name])
        call(self.cudart.cudaMemcpy(host.ctypes.data, self.out[name], host.nbytes,
                                    self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        return host


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
    (BUILD/"idd_kernels.cu").write_text(kernel_source())
    for source in ("pilot_filter", "idd_kernels"):
        subprocess.run([NVCC, "-cubin", f"-arch={architecture}", "-O3", "-o", str(BUILD/f"{source}.cubin"),
                        str((HERE if source == "pilot_filter" else BUILD)/f"{source}.cu")], check=True)
    symbols = subprocess.run(["nm", str(BUILD/"idd_kernels.cubin")], check=True, capture_output=True, text=True).stdout
    names = {key: next(line.split()[-1] for line in symbols.splitlines() if key in line and " T " in line)
             for key in ("FusedPrepare", "ScoresWarp")}
    reference = np.load(BUILD/"reference.npz")
    tables = dict(np.load(BUILD/"filter_tables.npz"))
    data = dict(np.load(BUILD/"idd_kernels.npz"))
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    def feeds(name):
        return {key.split("/")[-1]: reference[key] for key in reference.files if key.startswith(f"{name}/input/")}

    engine = Engine(trt, cudart, np, runtime, build(trt, logger, "proposed_cgnn", "fp16"), feeds("proposed_cgnn"))
    call(cuda.cuInit(0))
    network = FilterKernel(np, cuda, cudart, (BUILD/"pilot_filter.cubin").read_bytes(), tables, feeds("proposed"),
                           engine.addresses["channel"], engine.stream)
    second = call(cudart.cudaStreamCreate())
    idd = IddKernels(np, cuda, cudart, (BUILD/"idd_kernels.cubin").read_bytes(), names, data, second)
    events = {k: call(cudart.cudaEventCreate()) for k in ("start", "filter", "network", "idd")}

    def run(mode):
        """Times in ms from the start until the network's LLRs and until the IDD scores are ready."""
        first = engine.stream if mode != "idd" else second
        call(cudart.cudaEventRecord(events["start"], first))
        if mode != "idd":
            network.launch()
            call(cudart.cudaEventRecord(events["filter"], engine.stream))
        if mode != "network":
            if mode == "both":
                call(cudart.cudaStreamWaitEvent(second, events["filter"], 0))
            idd.launch()
            call(cudart.cudaEventRecord(events["idd"], second))
        if mode != "idd":
            engine.context.execute_async_v3(engine.stream)
            call(cudart.cudaEventRecord(events["network"], engine.stream))
        result = {}
        for key in ("network", "idd"):
            if mode in (key, "both"):
                call(cudart.cudaEventSynchronize(events[key]))
                result[key] = call(cudart.cudaEventElapsedTime(events["start"], events[key]))
        return result

    # Check the standalone kernels against TensorFlow's outputs for the same slot.
    run("both")
    check = {}
    for name in ("factor", "white", "logdet", "scores", "fit"):
        value, wanted = idd.read(name), data[name]
        check[name] = float(np.max(np.abs(value-wanted))/np.max(np.abs(wanted)))
    status = np.empty(1, np.int32)
    call(cudart.cudaMemcpy(status.ctypes.data, idd.status, 4, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
    llr, wanted = engine.output().astype(np.float32), reference["proposed/output"]
    confident = np.abs(wanted) > 1.
    check["network_sign_agreement_abs_llr_above_1"] = float(np.mean(np.sign(llr[confident]) == np.sign(wanted[confident])))
    if int(status[0]) or max(check[k] for k in ("factor", "white", "logdet", "scores", "fit")) > 1e-4:
        raise RuntimeError(f"standalone IDD kernels differ from TensorFlow: {check}, status {int(status[0])}")

    modes = ("network", "idd", "both")
    for mode in modes:
        for _ in range(args.warmup):
            run(mode)
    samples = {"network_alone": [], "idd_alone": [], "network_with_idd": [], "idd_with_network": []}
    for _ in range(args.rounds):
        for mode in modes:
            for _ in range(args.per_round):
                result = run(mode)
                if mode == "both":
                    samples["network_with_idd"].append(result["network"])
                    samples["idd_with_network"].append(result["idd"])
                else:
                    samples[f"{mode}_alone"].append(result[mode])
    np.savez(HERE/"results/overlap_samples.npz", **{k: np.asarray(v) for k, v in samples.items()})
    median = {k: float(np.median(v)) for k, v in samples.items()}
    report = {"gpu": properties.name.decode().rstrip("\x00"), "tensorrt": trt.__version__, "check": check,
              "median_ms": median,
              "network_delay_ms": median["network_with_idd"]-median["network_alone"],
              "network_delay_percent": 100*(median["network_with_idd"]/median["network_alone"]-1),
              "idd_ready_before_network_ms": median["network_with_idd"]-median["idd_with_network"],
              "samples_per_mode": args.rounds*args.per_round,
              "measure": "CUDA events; the IDD stream waits for the filter kernel, then runs FusedPrepare and "
                         "ScoresWarp while the CGNN engine runs on the network's stream"}
    (HERE/"results/overlap.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
