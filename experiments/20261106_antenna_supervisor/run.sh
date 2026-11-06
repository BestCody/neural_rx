#!/usr/bin/env bash
# Antenna-setting runs on both A40s: three conditions, 32 sequences each, in two halves per condition.
# Resumable: finished (condition, run, sequence) files are skipped. ./run.sh summarize writes summary.json.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniforge3/envs/neural_rx/bin/python}
if [ "${1:-}" = summarize ]; then
    CUDA_VISIBLE_DEVICES= exec "$PY" summarize.py
fi
mkdir -p logs
gpu=0
for condition in natural_umi controlled_topology_change schedule_gap_reentry; do
    for range in "0 16" "16 32"; do
        set -- $range
        tmux new-window -t nrx_antenna -n "${condition}_$1" \
            "$PY run.py $condition $1 $2 $gpu 2>&1 | grep --line-buffered -v 'cuda_\|WARNING' > logs/${condition}_$1.log; sleep 86400"
        gpu=$((1-gpu))
    done
done
