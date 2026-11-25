#!/usr/bin/env bash
# The classical receivers' sequences that run.sh could not finish, e.g. because two LMMSE + K-Best workers do not fit
# into one GPU's memory at 64-QAM (ResourceExhaustedError): after the tmux session modulations has ended, every
# configuration's classical receivers on all test sequences with one worker per GPU (evaluate.py skips sequences whose
# results exist), then the summaries.     ./run_missing.sh start | status
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$HOME/miniforge3/envs/neural_rx/bin/python}
CONFIGS=${CONFIGS:-"nrx_large_64qam.cfg nrx_large_qpsk.cfg"}
CLASSICAL="baseline_lslin_lmmse baseline_lmmse_kbest baseline_perf_csi_kbest"
case "${1:-status}" in
  start)
    tmux has-session -t =modulations_missing 2>/dev/null && { echo "already running"; exit 1; }
    tmux new-session -d -s modulations_missing "'$HERE/run_missing.sh' run >> '$HERE/logs/missing.log' 2>&1"
    echo "started tmux session modulations_missing" ;;
  run)
    echo "$(date -u +%FT%TZ) waiting for the tmux session modulations to end"
    while tmux has-session -t =modulations 2>/dev/null; do sleep 120; done   # =: exact name, not a prefix
    for c in $CONFIGS; do
      echo "$(date -u +%FT%TZ) ${c%.cfg}: classical receivers, one worker per GPU"
      pids=()
      for gpu in 0 1; do
        (cd "$HERE" && for first in $((16*gpu)) $((16*gpu+8)); do
           for s in sweep power_control; do for k in natural_umi controlled_topology_change schedule_gap_reentry; do
             "$PY" -u evaluate.py --config "$c" --setting "$s" --condition "$k" --gpu $gpu --receivers $CLASSICAL \
               --first-episode $first --last-episode $((first+8)) || echo "$s $k $first did not finish"
           done; done
         done >> "$HERE/logs/${c%.cfg}/missing_$gpu.log" 2>&1) &
        pids+=($!)
      done
      for pid in "${pids[@]}"; do wait "$pid"; done
      (cd "$HERE" && "$PY" summarize.py --config "$c")
    done
    echo "$(date -u +%FT%TZ) done" ;;
  status)
    tmux ls 2>/dev/null | grep -E '^(modulations|modulations_missing)' || echo "nothing running"
    tail -3 "$HERE/logs/missing.log" 2>/dev/null || true ;;
esac
