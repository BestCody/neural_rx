#!/usr/bin/env bash
# Decode all test sequences without IDD on both GPUs.
#   ./run.sh          start two tmux sessions (GPU 0 and GPU 1)
#   ./run.sh status   count the decoded episodes per grid and condition
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniforge3/envs/neural_rx/bin/python}
if [ "${1:-}" = status ]; then
    for s in sweep sweep_high; do
        for c in natural_umi controlled_topology_change schedule_gap_reentry; do
            echo "$s $c $(ls results/test/$s/$c/candidate_persistent_no_idd 2>/dev/null | grep -c npz || true)/32"
        done
    done
    exit 0
fi
mkdir -p logs
run() {  # gpu condition first last
    for s in sweep sweep_high; do
        echo "$PY evaluate.py --gpu $1 --setting $s --condition $2 --first-episode $3 --last-episode $4"
    done | paste -sd'&' | sed 's/&/ \&\& /g'
}
tmux new-session -d -s noidd_gpu0 "( $(run 0 natural_umi 0 32) && $(run 0 schedule_gap_reentry 0 16) ) >> logs/gpu0.log 2>&1"
tmux new-session -d -s noidd_gpu1 "( $(run 1 controlled_topology_change 0 32) && $(run 1 schedule_gap_reentry 16 32) ) >> logs/gpu1.log 2>&1"
tmux ls
