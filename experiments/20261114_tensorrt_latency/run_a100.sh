#!/usr/bin/env bash
# Time the receivers on this machine's GPU. Meant for a rented A100, from the bundle of make_a100_bundle.sh:
#
#     tar xzf a100_bundle.tar.gz && nrx_latency/run_a100.sh          # arguments go to time_fused.py (e.g. --gpu 1)
#
# Needs an NVIDIA driver for CUDA 12 and Python 3.10 to 3.12 with venv (PYTHON=... chooses the interpreter). TensorRT
# 10.3, as on the project's server, is installed into a virtual environment in this folder. The kernels are compiled
# here if nvcc is found, otherwise the bundle's compiled kernels for the A100 are used.
# Results: results/fused_a100.json, results/fused_a100_samples.npz, results/time_fused_a100.log
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x venv/bin/python ]; then
    "${PYTHON:-python3}" -m venv venv
    venv/bin/pip install --quiet --upgrade pip
    venv/bin/pip install --quiet tensorrt-cu12==10.3.0 cuda-python==12.4.0 numpy==1.26.4
fi
# the CUDA runtime that came with TensorRT, for machines without the CUDA toolkit
runtime=$(venv/bin/python -c 'import nvidia.cuda_runtime as m; print(list(m.__path__)[0]+"/lib")')
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$runtime"
mkdir -p results
{
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    grep -m1 'model name' /proc/cpuinfo
    venv/bin/python -u time_fused.py --tag a100 "$@"
} 2>&1 | tee results/time_fused_a100.log
echo "done: results/fused_a100.json, results/fused_a100_samples.npz, results/time_fused_a100.log"
