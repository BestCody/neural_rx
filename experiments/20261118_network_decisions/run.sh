#!/usr/bin/env bash
# Log the proposed NRX's per-TB evidence on the power-controlled test sequences, then fit the controller and replay
# the fallback. Four logging processes, two per GPU, each in its own tmux session.
#   ./run.sh            start the logging
#   ./run.sh status     count the logged sequences (32 per condition when done)
#   ./run.sh analyze    controller.py and fallback.py on the logged evidence
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniforge3/envs/neural_rx/bin/python}
CONDITIONS="natural_umi controlled_topology_change schedule_gap_reentry"
if [ "${1:-}" = status ]; then
    for c in $CONDITIONS; do
        echo "$c $(ls results/evidence_pc/$c 2>/dev/null | grep -c npz || true)/32"
    done
    tmux ls 2>/dev/null | grep evidence_ || echo "no logging session is running"
    exit 0
fi
if [ "${1:-}" = analyze ]; then
    "$PY" controller.py && "$PY" fallback.py
    exit 0
fi
mkdir -p logs
# With NVIDIA's multi-process service running, the processes on one GPU run at the same time instead of taking turns.
ROOT=$(cd ../.. && pwd)
MPS=""
pid=$(cat "$ROOT/.mps/pipe/nvidia-cuda-mps-control.pid" 2>/dev/null) && kill -0 "$pid" 2>/dev/null \
    && MPS="CUDA_MPS_PIPE_DIRECTORY=$ROOT/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$ROOT/.mps/log"
chain() {  # gpu first last
    for c in $CONDITIONS; do
        echo "$MPS $PY log_evidence.py --gpu $1 --condition $c --first-episode $2 --last-episode $3"
    done | paste -sd'&' | sed 's/&/ \&\& /g'
}
start=0
for session in 0a 0b 1a 1b; do                  # two processes per GPU (three ran out of GPU memory on scheduling gaps)
    tmux new-session -d -s evidence_$session "( $(chain ${session:0:1} $start $((start+8))) ) >> logs/gpu$session.log 2>&1"
    start=$((start+8))
done
tmux ls
