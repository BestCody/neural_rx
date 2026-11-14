#!/usr/bin/env bash
# Log the proposed NRX's per-TB evidence on the TBLER sweep's episodes, in two detached tmux sessions.
# Each condition uses the SNR grid of the stored sweep directory that holds its operating point
# (UMi and channel change: 7 dB in results/sweep; scheduling gaps: 8 dB in results/sweep_high),
# so every decision can be compared with the stored sweep.
#   ./run.sh            start both sessions
#   ./run.sh status     count the logged episodes
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniforge3/envs/neural_rx/bin/python}
REF=../20261104_tbler_sweep/results
mkdir -p logs
if [ "${1:-}" = status ]; then
    for c in natural_umi controlled_topology_change schedule_gap_reentry; do
        echo "$c $(ls results/evidence/$c 2>/dev/null | grep -c npz || true)/32"
    done
    exit 0
fi
GAPS="--condition schedule_gap_reentry --snr-offsets 3 4 5 6 7 8 --reference-dir $REF/sweep_high"
tmux new-session -d -s evidence_gpu0 \
    "$PY log_evidence.py --gpu 0 --condition natural_umi --reference-dir $REF/sweep > logs/gpu0.log 2>&1 && \
     $PY log_evidence.py --gpu 0 $GAPS --first-episode 0 --last-episode 16 >> logs/gpu0.log 2>&1"
tmux new-session -d -s evidence_gpu1 \
    "$PY log_evidence.py --gpu 1 --condition controlled_topology_change --reference-dir $REF/sweep > logs/gpu1.log 2>&1 && \
     $PY log_evidence.py --gpu 1 $GAPS --first-episode 16 --last-episode 32 >> logs/gpu1.log 2>&1"
tmux ls
