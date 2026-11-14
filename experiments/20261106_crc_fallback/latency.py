"""Median per-TB latency of the fallback receiver LMMSE + K-Best, on the inputs and timer of the paper's latency figure.

Same 128 UMi TBs at the anchor SNR, one TB at a time, after 16 warm-up TBs, host wall clock until the decoded bits
and CRC are back on the host. LMMSE channel estimation is part of this receiver, so it is timed. The proposed NRX is
timed again in the same process as a check against experiments/20261104_tbler_sweep/results/latency.json.
Usage: python latency.py <gpu>
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from neural_rx.backbone import E2E_Model, Parameters  # noqa: E402
from neural_rx.benchmark import tbler_sweep as sweep  # noqa: E402
from neural_rx.benchmark.frozen import protocol_and_plan  # noqa: E402
from neural_rx.benchmark.runtime import _latency_inputs, _measure_latency  # noqa: E402
from neural_rx.benchmark.scenarios import TemporalBenchmarkScenarioBank  # noqa: E402
from neural_rx.benchmark.scoring import payload_error_flags  # noqa: E402

HERE = Path(__file__).parent
SWEEP = HERE.parent/"20261104_tbler_sweep"
BACKUP, PRIMARY = "baseline_lmmse_kbest", "candidate_persistent"


def time_backup(protocol, plan, bank, receiver):
    timing = protocol["execution"]["latency"]
    warmup, count = int(timing["warmup_samples"]), int(timing["sample_count"])
    episodes, snr_index = _latency_inputs(protocol, plan, bank)
    values, errors, seen = [], [], 0
    for episode in episodes:
        for tb in range(episode.bits.shape[0]):
            inputs = episode.receiver_inputs[snr_index][tb]
            started = time.perf_counter()
            hard, crc = receiver.decode(np.asarray(inputs["received_grid"])[None],
                                        np.asarray([inputs["noise_variance"]], np.float32),
                                        np.asarray(episode.active[tb])[None], slot_number=int(inputs["slot_number"]))
            elapsed = 1e3*(time.perf_counter()-started)
            flags, _ = payload_error_flags(hard[0], crc[0], episode.bits[tb], episode.active[tb])
            if seen >= warmup:
                values.append(elapsed)
                errors.append(flags)
            seen += 1
            if len(values) == count:
                return np.asarray(values), np.asarray(errors)
    raise ValueError("too few TBs for the latency measurement")


def main():
    sweep.configure_runtime()
    protocol, plan = protocol_and_plan("latency", sweep.PRBS, protocol_path=sweep.PROTOCOL)
    parameters = Parameters(sweep.CONFIG, training=False, num_tx_eval=2, n_size_bwp_override=sweep.PRBS)
    bank = TemporalBenchmarkScenarioBank(parameters, E2E_Model(parameters, training=False, mcs_arr_eval_idx=0), tf)
    condition = protocol["execution"]["latency"]["condition"]
    backup = sweep.make_receiver(BACKUP, protocol, sweep.load_covariance(SWEEP/"covariance", condition))
    values, errors = time_backup(protocol, plan, bank, backup)
    primary = _measure_latency(protocol, plan, bank, {PRIMARY: sweep.make_receiver(PRIMARY, protocol)}, (PRIMARY,))
    result = {BACKUP: {"median_ms": float(np.median(values)), "samples": len(values),
                       "tbler": float(errors.mean())},
              PRIMARY: {"median_ms": float(np.median(primary[f"latency_ms_{PRIMARY}"])),
                        "samples": len(primary[f"latency_ms_{PRIMARY}"]),
                        "reference_median_ms": json.loads((SWEEP/"results/latency.json").read_text())[PRIMARY]
                        ["median_ms"]}}
    (HERE/"results").mkdir(exist_ok=True)
    np.savez_compressed(HERE/"results/latency.npz", **{f"latency_ms_{BACKUP}": values, f"errors_{BACKUP}": errors,
                                                       **primary})
    (HERE/"results/latency.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
