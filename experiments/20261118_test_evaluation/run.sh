#!/usr/bin/env bash
# Decode every test sequence with the proposed receiver (with and without its memory) and the trained two-iteration
# neural_rx. Six processes, three per GPU, each in its own tmux session.
#   ./run.sh          start
#   ./run.sh status   count the decoded sequences per setting and condition (32 each when done)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniforge3/envs/neural_rx/bin/python}
SETTINGS="sweep sweep_high power_control"
CONDITIONS="natural_umi controlled_topology_change schedule_gap_reentry"
if [ "${1:-}" = status ]; then
    for s in $SETTINGS; do
        for c in $CONDITIONS; do
            echo "$s $c $(ls results/$s/$c/pilot_memory 2>/dev/null | grep -c npz || true) $(ls results/$s/$c/neural_rx_k2_trained 2>/dev/null | grep -c npz || true) /32"
        done
    done
    tmux ls 2>/dev/null | grep test_eval || echo "no evaluation session is running"
    exit 0
fi
mkdir -p logs
# RECEIVERS="pilot_memory pilot_memory_emptied" ./run.sh limits the run to these receivers.
ONLY=${RECEIVERS:+--receivers $RECEIVERS}
# With NVIDIA's multi-process service running (as the training launchers start it), the processes on one GPU run at
# the same time instead of taking turns.
ROOT=$(cd ../.. && pwd)
MPS=""
pid=$(cat "$ROOT/.mps/pipe/nvidia-cuda-mps-control.pid" 2>/dev/null) && kill -0 "$pid" 2>/dev/null \
    && MPS="CUDA_MPS_PIPE_DIRECTORY=$ROOT/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$ROOT/.mps/log"
chain() {  # gpu first last
    for s in $SETTINGS; do
        for c in $CONDITIONS; do
            echo "$MPS $PY evaluate.py --gpu $1 --setting $s --condition $c --first-episode $2 --last-episode $3 $ONLY"
        done
    done | paste -sd'&' | sed 's/&/ \&\& /g'
}
start=0
for session in 0a 0b 0c 1a 1b 1c; do            # three processes per GPU (four ran out of GPU memory on scheduling
    case $session in *a) count=6 ;; *) count=5 ;; esac     # gaps); episodes 0-15 on GPU 0, 16-31 on GPU 1
    tmux new-session -d -s test_eval_$session "( $(chain ${session:0:1} $start $((start+count))) ) >> logs/gpu$session.log 2>&1"
    start=$((start+count))
done
tmux ls
