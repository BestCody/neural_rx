#!/usr/bin/env bash
# Package-layout check: re-evaluate the selected checkpoint under the unchanged v1 protocol.
# Must reproduce experiments/20261019_learned_blend_joint_training_6000/results/confirmation_005000 exactly.
set -euo pipefail
cd "$(dirname "$0")/../.."
X=experiments/20261021_package_layout_validation
C=experiments/20261019_learned_blend_joint_training_6000
nrx-benchmark --benchmark-protocol config/temporal_benchmark_v1.json --experiment-phase confirmation \
    --experiment-checkpoint models/learned_blend_joint_20261019/checkpoint_005000 --experiment-prbs 132 --gpu 0 \
    --episode-cache /dev/shm/neural_rx_package_validation/confirmation \
    --reference-cache "$X/reference_cache_v1" \
    --experiment-anchors "$C/inputs/anchors.json" --output-dir "$X/results/v1_confirmation_005000"
