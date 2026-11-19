#!/usr/bin/env bash
# Pack what time_fused.py needs on another machine (a rented A100) into build/a100_bundle.tar.gz: the scripts, the
# kernels' source and their compiled form for the A100 (sm_80), and the exported files of build/ (export.py).
#
#     ./make_a100_bundle.sh
#
# On the other machine: tar xzf a100_bundle.tar.gz && nrx_latency/run_a100.sh, then copy nrx_latency/results back
# into this folder's results/.
set -euo pipefail
cd "$(dirname "$0")"
stage=$(mktemp -d)
trap 'rm -rf "${stage:?}"' EXIT
mkdir -p "$stage/nrx_latency/build"
cp time_trt.py time_fused.py filter_kernels.py fitted_filter.cu run_a100.sh "$stage/nrx_latency/"
cp build/large_k2.onnx build/large_k3.onnx build/large_k8.onnx build/proposed_cgnn.onnx build/reference.npz \
   build/filter_tables.npz "$stage/nrx_latency/build/"
"$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)" -cubin -arch=sm_80 -O3 \
    -o "$stage/nrx_latency/build/fitted_filter_sm_80.cubin" fitted_filter.cu
tar czf build/a100_bundle.tar.gz -C "$stage" nrx_latency
ls -l build/a100_bundle.tar.gz
