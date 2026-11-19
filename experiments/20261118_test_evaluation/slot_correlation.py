"""Correlation of the channel over time, across slots, per condition: E[h(t+lag) h(t)^*]/E|h|^2 for lags of 0, 1, ...
OFDM symbols. Known statistics for the classical multi-slot estimator (wiener.py), from channel draws seeded apart
from the test sequences, as the sweep's covariance matrices (neural_rx.cli.sweep covariance), but over as many
consecutive slots as a memory entry can be old.

    python slot_correlation.py --gpu 0          # slot_correlation/<condition>.npz: time_correlation [lags]
"""
import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONDITIONS = ("natural_umi", "controlled_topology_change", "schedule_gap_reentry")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--subcarrier-step", type=int, default=8)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import numpy as np
    import tensorflow as tf
    from neural_rx.benchmark import tbler_sweep as sweep

    sweep.configure_runtime()
    protocol, plan, parameters, bank, anchors = sweep.sweep_context()
    w = tf.cast(parameters.transmitters[0]._precoder._w, tf.complex64)
    memory = json.loads((ROOT/"config/memory_receiver.json").read_text())["benchmark_memory"]
    slots = int(memory["expiry_tbs"])+1                  # the current slot and the oldest entry that is still used
    (HERE/"slot_correlation").mkdir(exist_ok=True)
    for condition in args.conditions:
        job = sweep.sweep_job(plan, anchors, condition)
        job = dict(job, seed_streams={k: v-sweep.SWEEP_SEED+sweep.COVARIANCE_SEED for k, v in job["seed_streams"].items()})
        strata = len(job["provider_settings"].get("doppler_strata_hz", [0]))
        products, power = None, 0.
        for draw in range(args.draws):
            generator = bank._generator(job, draw % strata)
            tf.random.set_seed(sweep.COVARIANCE_SEED+draw)
            h = generator.sample_pool_channel(args.batch_size, slots)[:, :, 0]      # [batch, slot, a, UE, t, s, f]
            h = tf.transpose(h, perm=[0, 1, 3, 5, 6, 2, 4])[:, :, :, :, ::args.subcarrier_step]
            effective = tf.matmul(h, tf.reshape(w[0], [1, 1, 1, 1, 1, 2, 1]))[..., 0]   # [batch, slot, UE, s, f, a]
            symbols = effective.shape[3]
            x = tf.reshape(tf.transpose(effective, [0, 2, 5, 4, 1, 3]), [-1, slots*symbols])   # one row per path in time
            lags = slots*symbols
            sums = np.stack([tf.reduce_mean(x[:, lag:]*tf.math.conj(x[:, :lags-lag])).numpy() for lag in range(lags)])
            products = sums if products is None else products+sums
            power += float(tf.reduce_mean(tf.abs(x)**2))
        correlation = (products/power).astype(np.complex64)
        np.savez(HERE/"slot_correlation"/f"{condition}.npz", time_correlation=correlation, slots=slots,
                 draws=args.draws, batch_size=args.batch_size)
        shown = [0, 1, symbols-1, symbols, 2*symbols, 4*symbols, lags-1]
        print(condition, {lag: round(float(abs(correlation[lag])), 3) for lag in shown}, flush=True)


if __name__ == "__main__":
    main()
