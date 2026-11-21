#!/usr/bin/env bash
# Log the stateless retrained two-iteration NRX's evidence on the power-controlled test sequences
# (stateless_evidence.py), one process per GPU listed in GPUS, repeating what could not finish (GPU memory).
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniforge3/envs/neural_rx/bin/python}
ROOT=$(cd ../.. && pwd)
export CUDA_MPS_PIPE_DIRECTORY=$ROOT/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$ROOT/.mps/log
GPU=${GPU:-1}
for attempt in 1 2 3; do
    for c in natural_umi controlled_topology_change schedule_gap_reentry; do
        "$PY" stateless_evidence.py --gpu "$GPU" --condition "$c" || echo "attempt $attempt of $c did not finish"
    done
done
echo "stateless evidence done"
