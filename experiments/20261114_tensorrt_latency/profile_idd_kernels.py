"""Where does the time of IDD's detector kernel go?

DetectorWarp is cut from src/neural_rx/receiver/native_ops/pair_correction.cu.cc, compiled without the TensorFlow
wrappers, and run on the inputs of one slot (build/idd_kernels.npz and build/idd_correction.npz, written by
time_idd.py), timed with CUDA events (median). Its output is checked against TensorFlow's.

- Stage costs: each stage is run twice by editing the source text before compiling; the added time is the stage's
  cost. (Early exits after a stage mislead, because the compiler drops the work whose results are then unused.)
- Tuning: resource elements per block and the compiler's register limit, which keep the results.

    ~/venvs/tensorrt/bin/python profile_idd_kernels.py [--gpu 0]     # results/idd_kernel_profile.json

History (commit c23faeb to this one, same slot, A40, GPU time of IDD per slot): 0.49 ms with separate score and
correction kernels; 0.39 ms with the covariance as one kernel and single gathers; 0.33 ms with the bit sums spread over
the warp; 0.32 ms with the preparation as one warp per cell; 0.27 ms with the merged, padded detector kernel. Tried
without gain: per-symbol products of the residual, the blend weights or the hidden units in shared memory, and the
settings under TUNING.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from time_fused import NVCC
from time_trt import call

HERE = Path(__file__).resolve().parent
BUILD = HERE/"build"
NATIVE = HERE.parents[1]/"src/neural_rx/receiver/native_ops"
KERNEL = "DetectorWarp"
SYNC = " __syncwarp();\n"
# stage: (start text, end text); the stage runs from the start through the end
STAGES = {
    "setup_whitening_and_channel": (" if(lane<groups){\n   int g=lane;", SYNC),
    "pair_scores": (" for(int pair=lane;pair<pairs;pair+=32){", " minimum=warp_min(minimum);\n"+SYNC),
    "symbol_sums_without_feedback": (" rows[lane]=symbol_lane?padded_symbol_sum(maxlog,user,symbol,score,nullptr)", SYNC),
    "bit_sums_without_feedback": (" for(int item=lane;item<2*WARP_BITS*WARP_SYMBOLS;item+=32){\n   int bit=item/WARP_SYMBOLS;\n",
                                  " physical=padded_bit_llrs(maxlog,lane,kbits,q,codes,metric,maxima);\n"),
    "calibration": (" if(lane<kbits){\n   table=(bit_user*n+p)*q+target;", SYNC),
    "priors": (" {\n   float value=0.f;\n   for(int k=0;k<WARP_BITS;++k){", SYNC),
    "symbol_sums_with_feedback": (" rows[lane]=symbol_lane?padded_symbol_sum(maxlog,user,symbol,score,single?nullptr:priors)", SYNC),
    "bit_sums_with_feedback": (" for(int item=lane;item<2*WARP_BITS*WARP_SYMBOLS;item+=32){\n   int bit=item/WARP_SYMBOLS,other=",
                               " llr=padded_bit_llrs(maxlog,lane,kbits,q,codes,metric,maxima);\n"),
    "blend_and_output": (" if(lane<kbits){\n   float delta=llr-physical;", "   }\n }\n"),
}
# name: (source edits, nvcc flags)
RES = "#define WARP_RES 4"
TUNING = {
    "res2": ([(RES, "#define WARP_RES 2")], []),
    "res8": ([(RES, "#define WARP_RES 8")], []),
    "registers40": ([], ["-maxrregcount=40"]),
    "registers48": ([], ["-maxrregcount=48"]),
}


def kernel_source():
    correction = (NATIVE/"pair_correction.cu.cc").read_text()
    return ("#include <cuda_runtime.h>\n#include <cmath>\n#include <cstdint>\n#include <cstddef>\nusing C=float2;\n"
            + correction[correction.index("__device__ C add("):correction.index("size_t ScalarWorkspace(")]
            + correction[correction.index("#define WARP_RES 4"):correction.index("bool MaxLog()")])


def repeated(source, start, end):
    """The source with one stage of the kernel in a loop that runs twice."""
    first = source.index(f"__global__ void {KERNEL}(")
    last = source.index("\n}\n", first)+1
    body = source[first:last]
    # values that a stage defines for later stages are declared before it
    for declaration, name in ((" const float physical=", "physical"), (" float llr=", "llr")):
        if body.count(declaration) != 1:
            raise ValueError(f"expected one occurrence of {declaration!r}")
        body = body.replace(declaration, f" {name}=")
    body = body.replace(" int warp=threadIdx.x/32,", " float physical=0.f,llr=0.f;\n int warp=threadIdx.x/32,", 1)
    if body.count(start) != 1:
        raise ValueError(f"expected one occurrence of {start!r}")
    a = body.index(start)
    b = body.index(end, a)+len(end)
    return source[:first]+body[:a]+" for(int rep=0;rep<2;++rep){\n"+body[a:b]+" }\n"+body[b:]+source[last:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import numpy as np
    from cuda import cuda, cudart

    call(cuda.cuInit(0))
    properties = call(cudart.cudaGetDeviceProperties(0))
    architecture = f"sm_{properties.major}{properties.minor}"
    data = {**np.load(BUILD/"idd_kernels.npz"), **np.load(BUILD/"idd_correction.npz")}
    cells, groups, r = data["factor"].shape[1:4]
    n = data["y"].shape[1]
    m, q = data["labels"].shape
    code_size = int(np.prod(data["evidence"].shape[2:]))
    hidden = data["blend_kernel"].shape[1]
    c64, f32, i64, i32, u64 = np.complex64, np.float32, np.int64, np.int32, np.uint64
    stream = call(cudart.cudaStreamCreate())
    custream = cuda.CUstream(int(stream))

    def upload(name, dtype):
        array = np.ascontiguousarray(data[name], dtype)
        pointer = call(cudart.cudaMalloc(array.nbytes))
        call(cudart.cudaMemcpy(pointer, array.ctypes.data, array.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
        return pointer, u64

    effective = upload("effective", f32)
    corrected = call(cudart.cudaMalloc(data["effective"].nbytes))
    values = [(False, np.bool_), (True, np.bool_), (data["llr_limit"], f32), (n, i32), (n, i32), (r, i32), (cells, i32),
              (groups, i32), (code_size, i32), (hidden, i32), (m, i32), (q, i32), (False, np.bool_),
              upload("y", c64), upload("factor", c64), upload("white", c64), upload("logdet", f32), upload("phi", f32),
              upload("cells", i64), upload("symbols", c64), upload("energy_indices", i64), upload("labels", i64),
              upload("enabled", np.bool_), upload("evidence", f32), upload("bit_indices", i64), upload("bit_signs", f32),
              upload("wanted", np.bool_), effective, upload("calibration", f32), upload("strength", f32),
              upload("active", f32), upload("blend_kernel", f32), upload("blend_bias", f32), upload("blend_output", f32),
              upload("blend_output_bias", f32), (corrected, u64)]
    keep = [np.array([value], dtype) for value, dtype in values]
    params = np.array([a.ctypes.data for a in keep], u64)
    events = [call(cudart.cudaEventCreate()) for _ in range(2)]
    base = kernel_source()

    def time(name, source, flags=()):
        resolution = int(re.search(r"#define WARP_RES (\d+)", source).group(1))  # resource elements per block
        (BUILD/f"profile_{name}.cu").write_text(source)
        subprocess.run([NVCC, "-cubin", f"-arch={architecture}", "-O3", *flags, "-o", str(BUILD/f"profile_{name}.cubin"),
                        str(BUILD/f"profile_{name}.cu")], check=True, capture_output=True)
        symbols = subprocess.run(["nm", str(BUILD/f"profile_{name}.cubin")], check=True, capture_output=True,
                                 text=True).stdout
        mangled = next(line.split()[-1] for line in symbols.splitlines() if KERNEL in line and " T " in line)
        module = call(cuda.cuModuleLoadData((BUILD/f"profile_{name}.cubin").read_bytes()))
        function = call(cuda.cuModuleGetFunction(module, mangled.encode()))

        def run():
            # as InitializeCorrected does before the kernel (not timed)
            call(cudart.cudaMemcpyAsync(corrected, effective[0], data["effective"].nbytes,
                                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream))
            call(cudart.cudaEventRecord(events[0], stream))
            call(cuda.cuLaunchKernel(function, (n+resolution-1)//resolution, 1, 1, 32*resolution, 1, 1, 0, custream,
                                     params.ctypes.data, 0))
            call(cudart.cudaEventRecord(events[1], stream))
            call(cudart.cudaEventSynchronize(events[1]))
            return 1000*call(cudart.cudaEventElapsedTime(events[0], events[1]))

        for _ in range(args.warmup):
            run()
        median = float(np.median([run() for _ in range(args.runs)]))
        output = np.empty(data["corrected"].shape, f32)
        call(cudart.cudaMemcpy(output.ctypes.data, corrected, output.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        print(name, round(median, 1), flush=True)
        return median, float(np.max(np.abs(output-data["corrected"])))

    full, difference = time("full", base)
    if difference > 0:
        raise RuntimeError(f"the standalone kernel differs from TensorFlow by {difference}")
    start = " __shared__ C yw_all["
    if base.count(start) != 1:
        raise ValueError("expected one kernel with shared tables")
    empty = time("empty", base.replace(start, " return;\n"+start))[0]
    stages = {name: time(f"repeat_{name}", repeated(base, *anchors))[0]-full for name, anchors in STAGES.items()}
    tuning = {}
    for name, (edits, flags) in TUNING.items():
        source = base
        for old, new in edits:
            source = source.replace(old, new)
        tuning[name], difference = time(name, source, flags)
        if difference > 0:
            raise RuntimeError(f"{name}: the kernel differs from TensorFlow by {difference}")
    report = {"gpu": properties.name.decode().rstrip("\x00"), "runs": args.runs, "kernel": KERNEL, "full_us": full,
              "launch_and_early_exit_us": empty, "stage_cost_us": stages, "tuning_us": tuning,
              "measure": "CUDA events around one launch, median; stage cost = time with the stage run twice - full"}
    (HERE/"results/idd_kernel_profile.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
