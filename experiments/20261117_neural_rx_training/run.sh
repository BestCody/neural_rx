#!/usr/bin/env bash
# Stage 1, neural_rx with two iterations: ./run.sh start | status | stop
# The training runs until it is stopped; start after a stop continues from the last saved update.
# It uses both GPUs (gradients on both, data generation on GPU 1, development checks on GPU 0), so it does not start while
# stage 2 is running.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PY=${PY:-$HOME/miniforge3/envs/neural_rx/bin/python}
# NVIDIA's multi-process service lets the processes that share a GPU (data generation and the second gradient process
# on GPU 1; the learner and a development check on GPU 0) run at the same time instead of taking turns. Its socket path
# must be short, hence a directory at the top of the repository.
export CUDA_MPS_PIPE_DIRECTORY=$ROOT/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$ROOT/.mps/log
mps_running() {
  local pid
  pid=$(cat "$CUDA_MPS_PIPE_DIRECTORY/nvidia-cuda-mps-control.pid" 2>/dev/null) && kill -0 "$pid" 2>/dev/null
}
MODELS=$ROOT/models/neural_rx_k2_20261117
case "${1:-status}" in
  start)
    tmux has-session -t pilot_memory_training 2>/dev/null && { echo "stage 2 is running and uses both GPUs; stop it first"; exit 1; }
    mkdir -p "$HERE/logs" "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    if command -v nvidia-cuda-mps-control > /dev/null && ! mps_running; then
      for stale in control control_lock control_privileged nvidia-cuda-mps-control.pid; do   # left by a reboot
        rm -f "${CUDA_MPS_PIPE_DIRECTORY:?}/${stale:?}"
      done
      nvidia-cuda-mps-control -d
    fi
    mps_running && echo "multi-process service running" || echo "multi-process service not available; training runs without it"
    tmux new-session -d -s neural_rx_training "cd '$ROOT' && CUDA_MPS_PIPE_DIRECTORY='$CUDA_MPS_PIPE_DIRECTORY' CUDA_MPS_LOG_DIRECTORY='$CUDA_MPS_LOG_DIRECTORY' '$PY' -u '$HERE/run.py' >> '$HERE/logs/train.log' 2>&1"
    echo "started tmux session neural_rx_training" ;;
  stop)
    tmux has-session -t neural_rx_training 2>/dev/null || { echo "not running"; exit 0; }
    tmux send-keys -t neural_rx_training C-c
    for _ in $(seq 90); do tmux has-session -t neural_rx_training 2>/dev/null || break; sleep 1; done
    tmux kill-session -t neural_rx_training 2>/dev/null || true
    mps_running && echo quit | nvidia-cuda-mps-control || true
    echo "stopped"
    [ -f "$HERE/results/best.json" ] && cat "$HERE/results/best.json" || echo "no development check finished yet" ;;
  status)
    tmux has-session -t neural_rx_training 2>/dev/null && echo "running" || echo "not running"
    [ -f "$MODELS/progress.json" ] && "$PY" -c "import json; r=json.load(open('$MODELS/progress.json')); print('update', r['update'], 'loss', round(r['loss'], 4), 'lr', r['learning_rate'])"
    for f in "$HERE"/results/development_*/results.json; do
      [ -f "$f" ] && "$PY" -c "import json; r=json.load(open('$f')); s=r['summary']['candidate_persistent']; print('development', r['update'], round(s['core_tbler'], 4))"
    done
    [ -f "$HERE/results/best.json" ] && cat "$HERE/results/best.json" || true ;;
esac
