#!/usr/bin/env bash
# Validation behind the sweep, in the order it ran. Results are in validation/ and logs/.
#   doubletdl   restored baselines vs the curves stored in NVlabs/neural_rx (results/nrx_rt_results)
#   perfect     Perf.-CSI + K-Best at 3 dB with more blocks
#   lmmse       LMMSE + K-Best with the final covariance matrices
#   reproduce   episode fingerprints and block decisions vs the sealed confirmation run
#   covariance  condition-specific covariance matrices for LMMSE estimation
#   latency     median latency on the benchmark timer; run while nothing else uses the machine
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniforge3/envs/neural_rx/bin/python}
RUN="$PY -m neural_rx.cli.sweep --gpu ${GPU:-0}"
case "$1" in
  doubletdl) $RUN doubletdl --systems baseline_lslin_lmmse baseline_perf_csi_kbest baseline_lmmse_kbest --snr-db 2 3 4 --batch-size 7 --batches 60 --covariance-dir covariance_quick --output validation/doubletdl.json ;;
  perfect) $RUN doubletdl --systems baseline_perf_csi_kbest --snr-db 3 --batch-size 7 --batches 480 --covariance-dir covariance_quick --output validation/doubletdl_perfect_recheck.json ;;
  lmmse) $RUN doubletdl --systems baseline_lmmse_kbest --snr-db 2 3 --batch-size 7 --batches 240 --covariance-dir covariance --output validation/doubletdl_lmmse_final_covariance.json ;;
  reproduce) $RUN reproduce --sealed ../20261021_current_code_reproduction/results/confirmation_005000/errors.npz --conditions natural_umi natural_uma controlled_topology_change schedule_gap_reentry --methods candidate_persistent frozen_cold_k8 --episodes 4 ;;
  covariance) $RUN covariance --conditions natural_umi natural_uma controlled_topology_change schedule_gap_reentry --covariance-dir covariance --draws 64 --batch-size 32 ;;
  latency) $RUN latency --methods candidate_persistent candidate_reset frozen_cold_k2 frozen_cold_k3 frozen_cold_k8 --output results/latency.json ;;
esac
