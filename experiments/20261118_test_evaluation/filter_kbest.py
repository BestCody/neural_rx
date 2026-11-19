"""The proposed NRX's pilot memory in front of the classical K-Best detector, with the NRX's channel filter or with a
classical estimator.

    receiver = FilterKBest(sweep, protocol, tf)                                   # interface of the NRX's adapter
    receiver = FilterKBest(sweep, protocol, tf, "wiener", condition)              # classical multi-slot estimator

The memory and the filter are those of the proposed NRX's benchmark adapter: the same memory policy, the same fitted
filter, and no trained weights. Only what follows the filter differs: instead of the two CGNN iterations, the K-Best
detector of the classical baselines (list size 64, src/neural_rx/benchmark/baselines.py) detects on the refined
channel estimate, followed by the same TB decoder. A TB with one scheduled UE uses the single-stream detector, as the
baselines do.

The detector also needs the error variance of the channel estimate. The filter combines its prediction and the LS
estimate with the weight g = sigma_LS^2 / sigma_K^2 of the prediction, which leaves the error variance
sigma_LS^2 (1 - g). g is not an output of the filter, but refined - LS = g (prediction - LS) has the mean power
g^2 sigma_K^2 = g sigma_LS^2 at the pilots, so g is that power over sigma_LS^2, per UE.

With estimator "wiener", the channel filter is replaced by a classical LMMSE (Wiener) estimator over frequency,
antennas and the DMRS symbols of the current and the stored TBs (wiener.py), which takes its filters from the known
statistics of the condition: the sweep's covariance matrices and the correlation over several slots
(slot_correlation.py). "wiener_slot" is the same estimator on the current TB only.
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COVARIANCE = HERE.parent/"20261104_tbler_sweep/covariance"      # per condition, as for LMMSE + K-Best
CHUNK = 5        # SNR points per K-Best call, as BASELINE_CHUNK in the sweep


class FilterKBest:
    def __init__(self, sweep, protocol, tf, estimator="filter", condition=None, first_tb=0):
        from neural_rx.backbone import Parameters
        from neural_rx.benchmark.baselines import ClassicalBaselineReceiver, lmmse_reduction
        if estimator not in ("filter", "wiener", "wiener_slot"):
            raise ValueError(f"Unknown estimator {estimator}")
        self.tf, self.estimator = tf, estimator
        self.first_tb = int(first_tb)                   # earlier TBs only fill the memory and are not detected
        self.adapter = sweep.make_receiver("candidate_persistent", protocol)
        parameters = Parameters(sweep.CONFIG, training=False, num_tx_eval=2, n_size_bwp_override=sweep.PRBS)
        self.classical = ClassicalBaselineReceiver(parameters, "baseline_perf_csi_kbest")   # detector and decoder
        self.wiener = None
        if estimator != "filter":
            from wiener import SlotWiener
            with np.load(HERE/"slot_correlation"/f"{condition}.npz") as data:
                correlation = data["time_correlation"]
            self.wiener = SlotWiener(tf, self.adapter.layout, sweep.load_covariance(COVARIANCE, condition), correlation,
                                     lmmse_reduction(parameters.n_size_bwp), history=estimator == "wiener")
        self._buffer = None
        self.last = None                                # channel estimate and error variance of the last TB

    def start_episode_batch(self, points):
        self.adapter.start_episode_batch(points)
        if self.adapter._buffer is not self._buffer:    # a new memory buffer: build the graph around it
            self._buffer = self.adapter._buffer
            self._execute = self._build()
        self.adapter._execute = self._execute

    def decode_tb_batch(self, inputs):
        self.classical.set_slot(int(inputs[0]["slot_number"]))
        return self.adapter.decode_tb_batch(inputs)

    def checksums_unchanged(self):
        return self.adapter.checksums_unchanged()

    def _build(self):
        tf, adapter, classical = self.tf, self.adapter, self.classical
        buffer, model, layout, receiver = adapter._buffer, adapter.model, adapter.layout, classical.receiver

        @tf.function(reduce_retracing=True, jit_compile=True)
        def refine(channel, noise, ids, active, step, head, count, tables):
            """The estimate [batch, users, antennas, symbols, subcarriers], its error variance, and the observation."""
            history = buffer.gather(tables, head, count, ids, active, step)
            observation = model.observe(channel, noise)
            if self.wiener is not None:
                return (observation, *self.wiener(observation, history, active))
            _, refined = model.prepare([None, None, channel, tf.cast(active, tf.float32)], observation, history, layout)
            width = refined.shape[-1]                                            # real and imaginary parts
            moved = model.observe(refined, noise)[..., :width]-observation[..., :width]
            moved = tf.reduce_mean(tf.reduce_sum(tf.square(moved), -1), -1)/(width//2)      # [batch, users]
            ls_variance = tf.reduce_mean(tf.maximum(observation[..., -1], 0.), -1)
            weight = tf.clip_by_value(tf.math.divide_no_nan(moved, ls_variance), 0., 1.)
            refined = tf.complex(refined[..., :width//2], refined[..., width//2:])          # [batch, users, f, s, a]
            return observation, tf.transpose(refined, [0, 1, 4, 3, 2]), (ls_variance*(1.-weight))[:, :, None, None, None]

        @tf.function(reduce_retracing=True, jit_compile=False)
        def filter_and_store(channel, noise, control, active, head, count):
            ids, step = tf.cast(control[:, :-1], tf.int32), control[0, -1]
            tables = tuple(v.read_value() for v in (buffer.observations, buffer.timestamps, buffer.ports))
            observation, estimate, error_variance = refine(channel, noise, ids, active, step, head, count, tables)
            pending, finite = buffer.prepare_stage(observation, head, count, ids, active)
            with tf.control_dependencies([estimate]):
                next_head, next_count, complete = buffer.commit(pending, finite, step)
            return estimate, error_variance, next_head, next_count, complete

        def per_stream(value):
            """[batch, users, antennas, symbols, subcarriers] -> [batch, 1, antennas, users, 1, symbols, subcarriers],
            also for values that are constant over some of these dimensions."""
            return tf.transpose(value, [0, 2, 1, 3, 4])[:, None, :, :, None]

        @tf.function(jit_compile=False)
        def detect(y, estimate, error_variance, noise, active):
            scheduled = tf.cast(active, tf.float32)[:, None, None, :, None, None, None]
            h_hat = per_stream(estimate)*tf.cast(scheduled, tf.complex64)
            err_var = per_stream(error_variance)*scheduled
            llr = receiver._layer_demapper(receiver._mimo_detector([y, h_hat, err_var, noise]))
            return receiver._tb_decoder(llr)

        @tf.function(jit_compile=False)
        def detect_one(y, estimate, error_variance, noise, position):
            h_hat = per_stream(estimate)[:, :, :, position:position+1]
            err_var = per_stream(error_variance)[:, :, :, position:position+1]
            llr = classical.one_stream[position]([y, h_hat, err_var, noise])
            paused = tf.zeros_like(llr)
            llr = tf.concat([llr, paused] if position == 0 else [paused, llr], axis=1)
            return receiver._tb_decoder(receiver._layer_demapper(llr))

        def execute(y, channel, noise, control, active, head, count):
            refined, error_variance, next_head, next_count, complete = filter_and_store(
                channel, noise, control, active, head, count)
            self.last = (refined, error_variance)
            if int(control[0, -1]) < self.first_tb:
                hard = tf.zeros([int(y.shape[0]), int(active.shape[1]), int(classical.tx._tb_size)], tf.float32)
                return (hard, tf.cast(hard, tf.bool), tf.zeros(hard.shape[:2], tf.bool), next_head, next_count,
                        complete)
            y = tf.cast(y, tf.complex64)
            scheduled = active.numpy()[0]               # the SNR points of a batch share the TB's scheduling
            hard, crc = [], []
            for start in range(0, int(y.shape[0]), CHUNK):
                part = slice(start, start+CHUNK)
                if scheduled.all():
                    bits, status = detect(y[part], refined[part], error_variance[part], noise[part], active[part])
                else:
                    bits, status = detect_one(y[part], refined[part], error_variance[part], noise[part],
                                              int(np.argmax(scheduled)))
                hard.append(bits)
                crc.append(status)
            hard, crc = tf.concat(hard, 0), tf.concat(crc, 0)
            return tf.cast(hard, tf.float32), tf.cast(hard, tf.bool), crc, next_head, next_count, complete

        return execute
