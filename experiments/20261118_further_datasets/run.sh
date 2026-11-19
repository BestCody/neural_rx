#!/usr/bin/env bash
# The proposed NRX (with and without its memory) and the Large NRX with eight iterations on two further datasets at a
# fixed SNR per condition, independent of the test sequences and of the validation dataset:
#   v2      the benchmark's confirmation dataset: TDL, UMi, UMa, channel change, scheduling gaps, closely spaced LOS
#           UEs, near-far UEs, and ray-traced Munich channels
#   seed86  the same conditions without the ray-traced one, drawn with another seed
#   ./run.sh          start both in tmux (one per GPU)
#   ./run.sh status
set -euo pipefail
cd "$(dirname "$0")/../.."
X=experiments/20261118_further_datasets
PY=${PY:-$HOME/miniforge3/envs/neural_rx/bin/python}
CK=models/pilot_memory_no_teacher_20261118/checkpoint_006000
if [ "${1:-}" = status ]; then
    for d in v2 seed86; do
        [ -f $X/results/$d/results.json ] && echo "$d done" || echo "$d running or not started ($(ls $X/episodes/$d 2>/dev/null | grep -c npz || true) sequences generated)"
    done
    tmux ls 2>/dev/null | grep further_ || echo "no session is running"
    exit 0
fi
mkdir -p $X/logs
tmux new-session -d -s further_v2 "$PY -m neural_rx.cli.benchmark --experiment-phase confirmation \
    --experiment-checkpoint $CK --experiment-prbs 132 --gpu 0 --episode-cache $X/episodes/v2 \
    --reference-cache $X/reference_cache \
    --experiment-anchors experiments/20261021_package_layout_validation/results/v2_calibration/anchors.json \
    --output-dir $X/results/v2 > $X/logs/v2.log 2>&1"
tmux new-session -d -s further_seed86 "$PY -m neural_rx.cli.benchmark --experiment-phase confirmation \
    --benchmark-protocol $X/inputs/temporal_benchmark_v1_seed86000000.json \
    --experiment-checkpoint $CK --experiment-prbs 132 --gpu 1 --episode-cache $X/episodes/seed86 \
    --reference-cache $X/reference_cache \
    --experiment-anchors experiments/20261019_learned_blend_joint_training_6000/inputs/anchors.json \
    --output-dir $X/results/seed86 > $X/logs/seed86.log 2>&1"
tmux ls
