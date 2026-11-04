#!/usr/bin/env bash
# TBLER-vs-SNR sweep of the proposed receiver, the Cold NRX, and classical baselines.
# Episodes are generated in memory; results are per (condition, method, episode) and resumable.
#   ./run.sh <condition> <gpu>        decode SNR offsets -6..+3 dB around the operating point
#   ./run.sh high <condition> <gpu>   decode +3..+8 dB on the same channels; +3 dB must agree
#   ./run.sh summarize                write results/summary.json
# redecode_single_ue.py then corrected the K-Best baselines on TBs with one scheduled UE.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniforge3/envs/neural_rx/bin/python}
METHODS="candidate_persistent candidate_reset frozen_cold_k8 frozen_cold_k3 frozen_cold_k2 \
baseline_lslin_lmmse baseline_perf_csi_kbest baseline_lmmse_kbest"
if [ "$1" = summarize ]; then
    exec "$PY" -m neural_rx.cli.sweep summarize --results-dir results/sweep results/sweep_high \
        --output results/summary.json
elif [ "$1" = high ]; then
    # shellcheck disable=SC2086
    exec "$PY" -m neural_rx.cli.sweep --gpu "$3" run --condition "$2" --methods $METHODS \
        --output-dir results/sweep_high --covariance-dir covariance --snr-offsets 3 4 5 6 7 8
fi
# shellcheck disable=SC2086
"$PY" -m neural_rx.cli.sweep --gpu "$2" run --condition "$1" --methods $METHODS \
    --output-dir results/sweep --covariance-dir covariance
