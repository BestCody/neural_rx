#!/usr/bin/env bash
# Decode the power-controlled episodes with every comparison receiver, in two detached tmux sessions that start
# once the proposed NRX's logging (run_pc.sh) has finished on the same GPU.
#   ./run_receivers_pc.sh          queue both sessions
#   ./run_receivers_pc.sh status   count the decoded episodes per method
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniforge3/envs/neural_rx/bin/python}
METHODS="candidate_reset frozen_cold_k2 frozen_cold_k3 frozen_cold_k8 baseline_lslin_lmmse \
baseline_lmmse_kbest baseline_perf_csi_kbest"
if [ "${1:-}" = status ]; then
    for c in natural_umi controlled_topology_change schedule_gap_reentry; do
        for m in $METHODS; do
            echo "$c $m $(ls results/receivers_pc/$c/$m 2>/dev/null | grep -c npz || true)/32"
        done
    done
    exit 0
fi
RUN="$PY run_receivers_pc.py --methods $METHODS"
tmux new-session -d -s rx_gpu0 \
    "while tmux has-session -t pc_gpu0 2>/dev/null; do sleep 60; done; \
     $RUN --gpu 0 --condition natural_umi > logs/rx_gpu0.log 2>&1 && \
     $RUN --gpu 0 --condition schedule_gap_reentry --first-episode 0 --last-episode 16 >> logs/rx_gpu0.log 2>&1"
tmux new-session -d -s rx_gpu1 \
    "while tmux has-session -t pc_gpu1 2>/dev/null; do sleep 60; done; \
     $RUN --gpu 1 --condition controlled_topology_change > logs/rx_gpu1.log 2>&1 && \
     $RUN --gpu 1 --condition schedule_gap_reentry --first-episode 16 --last-episode 32 >> logs/rx_gpu1.log 2>&1"
tmux ls
