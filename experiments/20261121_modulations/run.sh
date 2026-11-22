#!/usr/bin/env bash
# The 16-QAM receivers at other modulations (MCS), in order: ./run.sh start | status | stop
#   1. calibration anchors of each configuration (calibrate.sh; started separately, this waits for them)
#   2. the development sets of all configurations, in /dev/shm, spread over both GPUs
#   3. per configuration, stage 1 then stage 2 (train.py), each in its own tmux session on both GPUs; a stage ends by
#      itself (train.py) and is resumed from its last checkpoint if it stopped for another reason (up to 3 times);
#      then the test evaluation of its NRXs (evaluate.py, 2 processes per GPU, 8 sequences each, repeated up to 3
#      times for what a process could not finish, e.g. for lack of GPU memory) and a summary
#   4. the classical receivers of all configurations (K-Best decodes a sequence about 50 times slower than an NRX, so
#      it runs after all training), and the final summaries
# CONFIGS (default: 64-QAM, then QPSK) selects the configurations.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PY=${PY:-$HOME/miniforge3/envs/neural_rx/bin/python}
CONFIGS=${CONFIGS:-"nrx_large_64qam.cfg nrx_large_qpsk.cfg"}
export CUDA_MPS_PIPE_DIRECTORY=$ROOT/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$ROOT/.mps/log
mps_running() {
  local pid
  pid=$(cat "$CUDA_MPS_PIPE_DIRECTORY/nvidia-cuda-mps-control.pid" 2>/dev/null) && kill -0 "$pid" 2>/dev/null
}
finished() { [ -f "$HERE/results/${1%.cfg}/stage$2/finished.json" ]; }
SETTINGS="sweep power_control"
CONDITIONS="natural_umi controlled_topology_change schedule_gap_reentry"
NEURAL="pilot_memory pilot_memory_emptied neural_rx_k2_trained frozen_cold_k2 frozen_cold_k3 frozen_cold_k8"
CLASSICAL="baseline_lslin_lmmse baseline_lmmse_kbest baseline_perf_csi_kbest"
evaluate_config() {   # config receivers...: these receivers on all test sequences
  local c=$1 pids=() worker
  shift
  local receivers=$*
  for worker in 0 1 2 3; do
    (cd "$HERE" && for attempt in 1 2 3; do
       for s in $SETTINGS; do for k in $CONDITIONS; do
         "$PY" -u evaluate.py --config "$c" --setting "$s" --condition "$k" --gpu $((worker/2)) --receivers $receivers \
           --first-episode $((8*worker)) --last-episode $((8*worker+8)) || echo "attempt $attempt of $s $k did not finish"
       done; done
     done >> "$HERE/logs/${c%.cfg}/evaluate_$worker.log" 2>&1) &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  (cd "$HERE" && "$PY" summarize.py --config "$c")
}
case "${1:-status}" in
  start)
    for other in neural_rx_training no_teacher_training modulations; do
      tmux has-session -t "$other" 2>/dev/null && { echo "$other is running and uses both GPUs"; exit 1; }
    done
    mkdir -p "$HERE/logs"
    tmux new-session -d -s modulations "'$HERE/run.sh' pipeline >> '$HERE/logs/pipeline.log' 2>&1"
    echo "started tmux session modulations" ;;
  pipeline)
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    if command -v nvidia-cuda-mps-control > /dev/null && ! mps_running; then
      for stale in control control_lock control_privileged nvidia-cuda-mps-control.pid; do
        rm -f "${CUDA_MPS_PIPE_DIRECTORY:?}/${stale:?}"
      done
      nvidia-cuda-mps-control -d
    fi
    echo "$(date -u +%FT%TZ) waiting for the calibration anchors"
    for c in $CONFIGS; do
      result=$HERE/results/${c%.cfg}/calibration/results.json
      until [ -f "$result" ]; do
        tmux has-session -t "calibrate_${c#nrx_large_}" 2>/dev/null || [ -f "$result" ] \
          || { echo "no calibration of $c is running or finished: ./calibrate.sh $c <gpu>"; exit 1; }
        sleep 60
      done
      "$PY" -c "import json,sys; r=json.load(open('$result')); sys.exit(1 if r['unbracketed_conditions'] else 0)" \
        || { echo "calibration of $c did not bracket 10% TBLER everywhere"; exit 1; }
    done
    echo "$(date -u +%FT%TZ) generating the development sets"
    set -- $CONFIGS
    pids=()
    for c in "$@"; do
      mkdir -p "$HERE/logs/${c%.cfg}"
      for part in 0 1; do
        (cd "$HERE" && "$PY" -u pregenerate_dev.py --config "$c" --gpu "$part" --part "$part" --parts 2 \
          >> "$HERE/logs/${c%.cfg}/pregenerate_$part.log" 2>&1) &
        pids+=($!)
      done
    done
    for pid in "${pids[@]}"; do wait "$pid" || { echo "generating a development set failed"; exit 1; }; done
    for c in $CONFIGS; do
      label=${c%.cfg}
      for stage in 1 2; do
        for attempt in 1 2 3; do
          finished "$c" "$stage" && break
          echo "$(date -u +%FT%TZ) $label stage $stage, attempt $attempt"
          session=train_${label#nrx_large_}_$stage
          tmux new-session -d -s "$session" \
            "cd '$HERE' && '$PY' -u train.py --config '$c' --stage $stage >> '$HERE/logs/$label/stage$stage.log' 2>&1"
          while tmux has-session -t "$session" 2>/dev/null; do sleep 30; done
          sleep 60                                   # let the stage's worker processes release the GPUs
        done
        finished "$c" "$stage" || { echo "$label stage $stage did not finish"; exit 1; }
        echo "$(date -u +%FT%TZ) $label stage $stage finished: $(cat "$HERE/results/$label/stage$stage/finished.json" | tr -d '\n ' | cut -c1-200)"
      done
      echo "$(date -u +%FT%TZ) $label test evaluation of the NRXs"
      evaluate_config "$c" $NEURAL
    done
    for c in $CONFIGS; do
      echo "$(date -u +%FT%TZ) ${c%.cfg} test evaluation of the classical receivers"
      evaluate_config "$c" $CLASSICAL
    done
    echo "$(date -u +%FT%TZ) all configurations trained and evaluated" ;;
  stop)
    tmux kill-session -t modulations 2>/dev/null || true
    for session in $(tmux ls -F '#{session_name}' 2>/dev/null | grep '^train_' || true); do
      tmux send-keys -t "$session" C-c
      for _ in $(seq 90); do tmux has-session -t "$session" 2>/dev/null || break; sleep 1; done
      tmux kill-session -t "$session" 2>/dev/null || true
    done
    echo "stopped" ;;
  status)
    tmux ls 2>/dev/null | grep -E '^(modulations|train_|calibrate_)' || echo "nothing running"
    for c in $CONFIGS; do
      for stage in 1 2; do
        best=$HERE/results/${c%.cfg}/stage$stage/best.json
        [ -f "$best" ] && "$PY" -c "import json; r=json.load(open('$best')); print('${c%.cfg} stage $stage: best', r['update'], round(r['core_tbler'], 4), 'checks', len(r['development']), '(finished)' if $(finished "$c" "$stage" && echo True || echo False) else '')"
      done
    done ;;
esac
