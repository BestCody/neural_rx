"""One-stream K-Best on one active UE (TDL-B100) vs the single-UE curves stored in NVlabs/neural_rx."""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sionna.utils import BinarySource, ebnodb2no  # noqa: E402

from neural_rx.backbone import Parameters  # noqa: E402
from neural_rx.benchmark import tbler_sweep as sweep  # noqa: E402
from neural_rx.benchmark.baselines import ClassicalBaselineReceiver  # noqa: E402

STORED = {"baseline_perf_csi_kbest": {0: .30254, 1: .05294}, "baseline_lmmse_kbest": {0: .4065, 1: .11829}}
sweep.configure_runtime()
parameters = Parameters(sweep.CONFIG, training=False, num_tx_eval=2, n_size_bwp_override=sweep.PRBS)
tx, channel, source = parameters.transmitters[0], parameters.channel, BinarySource()
covariance = sweep.load_covariance("covariance", "natural_umi")
covariance["space"] = covariance["space"]/covariance["power"]
batch = 5
for system, stored in STORED.items():
    receiver = ClassicalBaselineReceiver(parameters, system, covariance)
    for snr, expected in stored.items():
        no = ebnodb2no(float(snr), tx._num_bits_per_symbol, tx._target_coderate, tx._resource_grid)
        failures = {0: 0, 1: 0}
        for position in failures:
            active = np.zeros((batch, 2), bool)
            active[:, position] = True
            for _ in range(60):
                bits = source([batch, 2, tx._tb_size])
                x = tx(bits)*tf.cast(active[:, :, None, None, None], tf.complex64)
                h1, h2 = channel._gen_channel_1(batch), channel._gen_channel_2(batch)
                h = tf.concat([h1, h2] if position == 0 else [h2, h1], axis=3)
                y = channel._apply_channel([x, h, no])
                hard, _ = receiver.decode(y, tf.fill([batch], no), active, h=h)
                failures[position] += int(np.any(hard != bits.numpy(), axis=-1)[:, position].sum())
        print(system, snr, "dB:", failures[0]/300, failures[1]/300, "stored", expected, flush=True)
