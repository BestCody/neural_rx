#!/usr/bin/env bash
# Calibration anchors of one radio configuration (MCS): per condition, the SNR closest to 10% TBLER of the released
# eight-iteration Large NRX of that MCS, as for 16-QAM (neural_rx.benchmark.frozen.run_experiment, calibration phase,
# protocol v1: the eight statistical conditions). Episodes are cached in /dev/shm (approved by the user, 2026-11-21).
#   ./calibrate.sh nrx_large_64qam.cfg 0 [episodes]      config, GPU, episodes per condition (default 8)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PY=${PY:-$HOME/miniforge3/envs/neural_rx/bin/python}
CONFIG=$1 GPU=$2 EPISODES=${3:-}
LABEL=${CONFIG%.cfg}
OUT=$HERE/results/$LABEL/calibration${EPISODES:+_smoke}
export CUDA_MPS_PIPE_DIRECTORY=$ROOT/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$ROOT/.mps/log
"$PY" -m neural_rx.cli.benchmark --gpu "$GPU" --experiment-phase calibration --experiment-prbs 132 \
    --benchmark-protocol "$ROOT/config/temporal_benchmark_v1.json" --radio-config "$CONFIG" \
    ${EPISODES:+--experiment-episodes "$EPISODES"} \
    --episode-cache "/dev/shm/nrx_modulations/$LABEL/calibration${EPISODES:+_smoke}" \
    --reference-cache "$HERE/results/$LABEL/reference_cache${EPISODES:+_smoke}" --output-dir "$OUT"
