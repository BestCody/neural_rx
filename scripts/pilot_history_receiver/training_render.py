from __future__ import annotations

from threading import Lock, RLock
from types import SimpleNamespace

import tensorflow as tf

from .pilot_layout import PilotLayout
from .training_options import DEFAULT_RENDER_BATCH_TBS

_registry_lock = Lock()


class TrainingRenderer:
    def __init__(self, generator, batch_tbs=DEFAULT_RENDER_BATCH_TBS):
        if type(batch_tbs) is not int or batch_tbs < 1:
            raise ValueError("The renderer batch capacity must be positive")
        self.generator = generator
        self.batch_tbs = batch_tbs
        self.layout = PilotLayout(generator.rx._ls_est, generator.tx.resource_grid, 0)
        if generator.tx._output_domain != "freq":
            raise ValueError("Training channel application requires a frequency-domain transmitter")
        grid_type = generator.tx.resource_grid.build_type_grid()
        self.grid_shape = tf.constant(grid_type.shape, tf.int32)
        self.pilot_indices = tf.cast(tf.where(grid_type == 1), tf.int32)
        self.data_indices = tf.cast(tf.where(grid_type == 0), tf.int32)
        with _registry_lock:
            if not hasattr(generator.rx, "_memory_training_render_lock"):
                generator.rx._memory_training_render_lock = RLock()
            self.lock = generator.rx._memory_training_render_lock
        self.sequence = tf.function(self._sequence, reduce_retracing=True)

    def _estimate(self, y, pilots):
        estimator = self.generator.rx._ls_est
        pilots = self._pilots(estimator._pilot_pattern, pilots)
        context = SimpleNamespace(_pilot_pattern=SimpleNamespace(pilots=pilots[:, None, None]),
            _dmrs_length=estimator._dmrs_length, _num_dmrs_syms=estimator._num_dmrs_syms,
            _num_pilots_per_dmrs_sym=estimator._num_pilots_per_dmrs_sym,
            _num_cdm_groups_without_data=estimator._num_cdm_groups_without_data)
        effective = estimator._removed_nulled_scs(y)
        flat = tf.reshape(effective, tf.concat([tf.shape(effective)[:3], [-1]], 0))
        samples = tf.gather(flat, estimator._pilot_ind, axis=-1)
        h, variance = type(estimator).estimate_at_pilot_locations(context, samples, tf.constant(.1))
        h, _ = estimator._interpol(h, variance)
        h = tf.transpose(h[:, 0, :, :self.generator.num_scheduled_tx, 0], [0, 2, 4, 3, 1])
        return tf.concat([tf.math.real(h), tf.math.imag(h)], -1)

    @staticmethod
    def _pilots(pattern, pilots):
        return tf.cond(pattern.normalize, lambda: pilots*tf.cast(
            1/tf.sqrt(tf.reduce_mean(tf.abs(pilots)**2, -1, keepdims=True)), pilots.dtype), lambda: pilots)

    def _batch(self, channel, mask, slots, no, payload_seed, noise_seed, times):
        gen = self.generator
        count, lanes = tf.shape(channel)[0], tf.shape(channel)[1]
        payload = tf.map_fn(lambda t: gen._sample_bits(
            [lanes, gen.num_scheduled_tx, gen.tx._tb_size], payload_seed, t), times,
            fn_output_signature=tf.TensorSpec([None, gen.num_scheduled_tx, gen.tx._tb_size], gen.source.dtype))
        payload = tf.reshape(payload, [-1, gen.num_scheduled_tx, gen.tx._tb_size])
        coded = gen.tx._tb_encoder(payload)
        symbols = gen.tx._layer_mapper(gen.tx._mapper(coded))
        pilots = tf.repeat(tf.gather(gen.p.pilots, slots), lanes, axis=0)
        tx_pilots = self._pilots(gen.tx.pilot_pattern, pilots)
        template = tf.scatter_nd(self.pilot_indices, tf.transpose(tf.reshape(tx_pilots, [count*lanes, -1])),
                                 tf.concat([self.grid_shape, [count*lanes]], 0))
        grid = tf.tensor_scatter_nd_update(template, self.data_indices,
                                           tf.transpose(tf.reshape(symbols, [count*lanes, -1])))
        transmitted = tf.transpose(grid, [4, 0, 1, 2, 3])
        if gen.tx._precoding == "codebook":
            transmitted = gen.tx._precoder(transmitted)
        mask = tf.reshape(mask, [-1, gen.num_scheduled_tx])
        transmitted *= tf.cast(mask[:, :, None, None, None], transmitted.dtype)
        h = tf.reshape(channel, [-1, *channel.shape[2:]])
        clean = gen.apply_channel_noiseless([transmitted, h])
        clean = tf.reshape(clean, [count, lanes, *clean.shape[1:]])
        y = tf.map_fn(lambda values: gen._add_seeded_noise(values[0], values[1], noise_seed, values[2]),
            (clean, no, times), fn_output_signature=tf.TensorSpec(clean.shape[1:], clean.dtype))
        y = tf.reshape(y, [-1, *y.shape[2:]])
        targets = self.layout.channel_targets(gen.rx.preprocess_channel_ground_truth(h))
        values = (payload, y, self._estimate(y, pilots), targets, coded)
        return tuple(tf.reshape(value, [count, lanes, *value.shape[1:]]) for value in values)

    def __call__(self, channel, active, ue_ids, slot_numbers, ebno_db, payload_seed, noise_seed):
        no = self.generator.ebno_to_no(tf.convert_to_tensor(ebno_db, tf.float32))
        with self.lock:
            bits, y, ls, targets, coded = self.sequence(channel, tf.convert_to_tensor(active, tf.float32),
                tf.convert_to_tensor(slot_numbers, tf.int32), no,
                tf.constant(payload_seed, tf.int32), tf.constant(noise_seed, tf.int32))
            ls[0, -1, 0, 0, 0, 0].numpy()
        return dict(bits=bits, y=y, ls=ls, no=no, channel_targets=targets, coded=coded,
                    active=tf.convert_to_tensor(active, tf.float32),
                    ue_ids=tf.convert_to_tensor(ue_ids, tf.int32),
                    slot_number=tf.convert_to_tensor(slot_numbers, tf.int32))

    def _sequence(self, channel, active, slots, no, payload_seed, noise_seed):
        length = tf.shape(channel)[1]
        channel, active, no = (tf.transpose(v, [1, 0, *range(2, v.shape.rank)]) for v in (channel, active, no))
        arrays = tuple(tf.TensorArray(dtype, size=length) for dtype in
            (self.generator.source.dtype, channel.dtype, channel.dtype.real_dtype,
             channel.dtype.real_dtype, self.generator.source.dtype))

        def step(t, arrays):
            end = tf.minimum(t+self.batch_tbs, length)
            times = tf.range(t, end)
            result = self._batch(channel[t:end], active[t:end], slots[t:end], no[t:end],
                                 payload_seed, noise_seed, times)
            return end, tuple(array.scatter(times, value) for array, value in zip(arrays, result))

        _, arrays = tf.while_loop(lambda t, _: t < length, step, (0, arrays), parallel_iterations=1)
        return tuple(tf.transpose(a.stack(), [1, 0, *range(2, a.element_shape.rank+1)]) for a in arrays)
