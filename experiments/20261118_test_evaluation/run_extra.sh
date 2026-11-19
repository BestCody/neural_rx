#!/usr/bin/env bash
# Decode the test sequences with receivers that run.sh does not decode: the ablation pilot_memory_current_tb and the
# classical comparisons filter_kbest, wiener_kbest (see evaluate.py). PER_GPU processes per GPU (default 2), each in
# its own tmux session; a process repeats what it could not finish (GPU memory), up to three times.
#   RECEIVERS="filter_kbest wiener_kbest" ./run_extra.sh          start
#   RECEIVERS="filter_kbest wiener_kbest" ./run_extra.sh status   decoded sequences per setting and condition
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniforge3/envs/neural_rx/bin/python}
SETTINGS=${SETTINGS:-"sweep sweep_high power_control"}
CONDITIONS=${CONDITIONS:-"natural_umi controlled_topology_change schedule_gap_reentry"}
: "${RECEIVERS:?name the receivers}"
missing() {  # setting condition first last
    for r in $RECEIVERS; do
        for e in $(seq "$3" $(($4-1))); do
            [ -f "results/$1/$2/$r/$(printf 'episode_%05d.npz' "$e")" ] || return 0
        done
    done
    return 1
}
if [ "${1:-}" = status ]; then
    for s in $SETTINGS; do
        for c in $CONDITIONS; do
            echo "$s $c $(for r in $RECEIVERS; do printf '%s %s/32  ' "$r" "$(ls results/$s/$c/$r 2>/dev/null | grep -c npz || true)"; done)"
        done
    done
    tmux ls 2>/dev/null | grep extra_ || echo "no session is running"
    exit 0
elif [ "${1:-}" = worker ]; then  # gpu first last
    for attempt in 1 2 3; do
        for s in $SETTINGS; do
            for c in $CONDITIONS; do
                if missing "$s" "$c" "$3" "$4"; then
                    # shellcheck disable=SC2086
                    "$PY" evaluate.py --gpu "$2" --setting "$s" --condition "$c" --first-episode "$3" \
                        --last-episode "$4" --receivers $RECEIVERS || echo "attempt $attempt of $s $c did not finish"
                fi
            done
        done
    done
    exit 0
fi
mkdir -p logs
# With NVIDIA's multi-process service running, the processes on one GPU run at the same time instead of taking turns.
ROOT=$(cd ../.. && pwd)
MPS=""
pid=$(cat "$ROOT/.mps/pipe/nvidia-cuda-mps-control.pid" 2>/dev/null) && kill -0 "$pid" 2>/dev/null \
    && MPS="CUDA_MPS_PIPE_DIRECTORY=$ROOT/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$ROOT/.mps/log"
PER_GPU=${PER_GPU:-2}                           # processes per GPU; episodes 0-15 on GPU 0, 16-31 on GPU 1
for gpu in 0 1; do
    for part in $(seq 0 $((PER_GPU-1))); do
        first=$((16*gpu+16*part/PER_GPU)) last=$((16*gpu+16*(part+1)/PER_GPU))
        tmux new-session -d -s "extra_$gpu$part" "$MPS RECEIVERS='$RECEIVERS' SETTINGS='$SETTINGS' CONDITIONS='$CONDITIONS' ./run_extra.sh worker $gpu $first $last >> logs/extra_$gpu$part.log 2>&1"
    done
done
tmux ls
