#!/usr/bin/env bash
# Decode every test sequence with the selected retrained checkpoint, once training has written results/best.json.
#   ./run_test.sh          queue two tmux sessions (GPU 0 and GPU 1)
#   ./run_test.sh status   count the decoded episodes per setting and condition
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniforge3/envs/neural_rx/bin/python}
if [ "${1:-}" = status ]; then
    for s in sweep sweep_high power_control; do
        for c in natural_umi controlled_topology_change schedule_gap_reentry; do
            echo "$s $c $(ls results/test/$s/$c/retrained_k2 2>/dev/null | grep -c npz || true)/32"
        done
    done
    exit 0
fi
mkdir -p logs
WAIT='while [ ! -f results/best.json ]; do sleep 120; done; CK=$('"$PY"' -c "import json; print(json.load(open(\"results/best.json\"))[\"checkpoint\"])")'
run() {  # gpu condition first last
    for s in sweep sweep_high power_control; do
        echo "$PY evaluate_test.py --gpu $1 --setting $s --condition $2 --first-episode $3 --last-episode $4 --checkpoint \$CK"
    done | paste -sd'&' | sed 's/&/ \&\& /g'
}
tmux new-session -d -s test_gpu0 "$WAIT; ( $(run 0 natural_umi 0 32) && $(run 0 schedule_gap_reentry 0 16) ) >> logs/test_gpu0.log 2>&1"
tmux new-session -d -s test_gpu1 "$WAIT; ( $(run 1 controlled_topology_change 0 32) && $(run 1 schedule_gap_reentry 16 32) ) >> logs/test_gpu1.log 2>&1"
tmux ls
