#!/usr/bin/env bash
# Default v2 protocol: calibrate every condition (including ray_traced_munich), then confirm.
set -euo pipefail
cd "$(dirname "$0")/../.."
X=experiments/20261021_package_layout_validation
common=(--experiment-checkpoint models/learned_blend_joint_20261019/checkpoint_005000 --experiment-prbs 132 --gpu 1
        --reference-cache "$X/reference_cache_v2")
nrx-benchmark --experiment-phase calibration "${common[@]}" \
    --episode-cache /dev/shm/neural_rx_package_validation/calibration \
    --output-dir "$X/results/v2_calibration"
nrx-benchmark --experiment-phase confirmation "${common[@]}" \
    --episode-cache /dev/shm/neural_rx_package_validation/confirmation \
    --experiment-anchors "$X/results/v2_calibration/anchors.json" \
    --output-dir "$X/results/v2_confirmation_005000"
