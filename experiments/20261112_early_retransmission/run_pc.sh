#!/usr/bin/env bash
# Log the proposed NRX's per-TB evidence on the TBLER sweep's episodes with closed-loop power control
# (power_control.py), in two detached tmux sessions. With power control the TBLER drops steeply, so the grid
# uses 0.5 dB steps from -2 to 6 dB plus 7, 8, and 10 dB.
#   ./run_pc.sh            start both sessions
#   ./run_pc.sh status     count the logged episodes
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniforge3/envs/neural_rx/bin/python}
OUT=results/evidence_pc
GRID="-2 -1.5 -1 -0.5 0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 7 8 10"
mkdir -p logs
if [ "${1:-}" = status ]; then
    for c in natural_umi controlled_topology_change schedule_gap_reentry; do
        echo "$c $(ls $OUT/$c 2>/dev/null | grep -c npz || true)/32"
    done
    exit 0
fi
RUN="$PY log_evidence.py --power-control --snr-db $GRID --output-dir $OUT"
tmux new-session -d -s pc_gpu0 \
    "$RUN --gpu 0 --condition natural_umi > logs/pc_gpu0.log 2>&1 && \
     $RUN --gpu 0 --condition schedule_gap_reentry --first-episode 0 --last-episode 16 >> logs/pc_gpu0.log 2>&1"
tmux new-session -d -s pc_gpu1 \
    "$RUN --gpu 1 --condition controlled_topology_change > logs/pc_gpu1.log 2>&1 && \
     $RUN --gpu 1 --condition schedule_gap_reentry --first-episode 16 --last-episode 32 >> logs/pc_gpu1.log 2>&1"
tmux ls
