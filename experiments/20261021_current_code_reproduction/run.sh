#!/usr/bin/env bash
# Re-evaluate the selected checkpoint with the current code on the original confirmation seed (88000000).
# An earlier run on seed 86000000 is kept in results/*_seed86000000 and run_seed86000000.log.
set -euo pipefail
cd "$(dirname "$0")/../.."
X=experiments/20261021_current_code_reproduction
C=experiments/20261019_learned_blend_joint_training_6000
python scripts/run_temporal_benchmark.py --experiment-phase confirmation \
    --experiment-checkpoint models/learned_blend_joint_20261019/checkpoint_005000 --experiment-prbs 132 --gpu 0 \
    --episode-cache /dev/shm/neural_rx_current_code_reproduction/confirmation \
    --reference-cache "$X/reference_cache" \
    --experiment-anchors "$C/inputs/anchors.json" --output-dir "$X/results/confirmation_005000"
