from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from neural_rx.receiver.pilot_history import MemoryPolicy, PilotHistory
from neural_rx.receiver.training_execution import BatchedTrainingExecution, routing_plan
from neural_rx.verification.pilot_history import ReferenceMemoryManager


class TrainingRoutingTests(unittest.TestCase):
    def test_batched_renderer_preserves_per_tb_seeds_pilots_and_schedule(self):
        from neural_rx.backbone.core import build_backbone
        from neural_rx.data.temporal import TemporalTrainingDataGenerator
        from neural_rx.receiver.training_render import TrainingRenderer
        p, e2e = build_backbone('nrx_large.cfg', 2, True, n_size_bwp_override=4)
        generator = TemporalTrainingDataGenerator(p, e2e)
        length, lanes, users = 7, 2, generator.num_scheduled_tx
        shape = [lanes, length, 1, p.num_rx_antennas, users, p.num_antenna_ports, 14, 48]
        channel = tf.complex(tf.random.stateless_normal(shape, [17, 3]),
                             tf.random.stateless_normal(shape, [21, 4]))
        active = np.ones([lanes, length, users], np.float32)
        active[:, 2:4, 0] = 0
        ids = np.tile(np.arange(users, dtype=np.int32), [lanes, length, 1])
        ids[:, 1::2] = ids[:, 1::2, ::-1]
        slots = (np.arange(length, dtype=np.int32)+int(p.pilots.shape[0])-2) % int(p.pilots.shape[0])
        ebno = tf.reshape(tf.linspace(-1., 6., lanes*length), [lanes, length])
        no = generator.ebno_to_no(ebno)
        renderer = TrainingRenderer(generator, batch_tbs=3)
        expected = []
        for t in range(length):
            generator._set_slot_pilots(slots[t])
            bits = generator._sample_bits([lanes, users, generator.tx._tb_size], 72, t)
            x = generator.tx(bits)*tf.cast(active[:, t, :, None, None, None], tf.complex64)
            y = generator._add_seeded_noise(generator.apply_channel_noiseless([x, channel[:, t]]), no[:, t], 91, t)
            expected.append(dict(bits=bits, y=y, ls=generator.rx.estimate_channel(y, users),
                coded=generator.tx._tb_encoder(bits)))
        pilots_before = np.asarray(generator.tx.pilot_pattern.pilots).copy()
        for batch in (1, 3, 16):
            actual = TrainingRenderer(generator, batch)(channel, active, ids, slots, ebno, 72, 91)
            for key in expected[0]:
                reference = tf.stack([value[key] for value in expected], 1)
                if key in ('bits', 'coded'):
                    np.testing.assert_array_equal(actual[key], reference)
                else:
                    np.testing.assert_allclose(actual[key], reference, rtol=2e-6, atol=2e-6)
            np.testing.assert_array_equal(actual['active'], active)
            np.testing.assert_array_equal(actual['ue_ids'], ids)
            np.testing.assert_array_equal(generator.tx.pilot_pattern.pilots, pilots_before)

    def test_channel_only_context_matches_receiver_context_for_all_training_families(self):
        from neural_rx.backbone import Parameters, E2E_Model
        from neural_rx.receiver.training_data import TrainingEpisodes
        recipe = json.loads((Path(__file__).resolve().parents[1]/"config/memory_training.json").read_text())
        parameters = Parameters(recipe["radio_config"], training=True, n_size_bwp_override=4)
        model = E2E_Model(parameters, training=True)
        full = TrainingEpisodes({4: (parameters, model)}, recipe, recipe["seed"])
        channels = TrainingEpisodes({4: (parameters, None)}, recipe, recipe["seed"])
        streams = {name: 1000+i for i, name in enumerate(("channel", "payload", "noise", "schedule"))}
        for family in ("tdl", "umi", "uma"):
            for changed in (False, True):
                regime = np.zeros((1, 3, 4), bool)
                regime[:, 1:, 0] = changed
                outputs = []
                for context in (full, channels):
                    generator = context._generator(4, family, 0)
                    generator._activate_seed_batch(streams)
                    outputs.append(generator.sample_pool_channel(1, 3, tf.constant(regime)).numpy())
                np.testing.assert_array_equal(*outputs)

    def test_prepare_accepts_one_sequence_per_example_only(self):
        trainer = SimpleNamespace(manager=MemoryPolicy(4, 7, 8), model=SimpleNamespace(history_tbs=7), recipe={})
        execution = BatchedTrainingExecution(trainer)
        def packet(lanes):
            return dict(y=np.zeros([lanes, 4, 1]), ls=np.zeros([lanes, 4, 1]), bits=np.zeros([lanes, 4, 1]),
                        ue_ids=np.tile([[[0, 1]]], [lanes, 4, 1]), active=np.ones([lanes, 4, 2]),
                        coded=np.zeros([lanes, 4, 1]), no=np.ones([lanes, 4]), h=np.zeros([lanes, 4, 1]),
                        metadata={})
        with mock.patch.object(execution, "describe", return_value=tf.zeros([1, 4, 2, 6, 9])):
            _, _, den = execution.prepare([packet(1)])
        self.assertEqual(den, 8.)
        with self.assertRaisesRegex(ValueError, "one sequence"):
            execution.prepare([packet(2)])

    def test_plan_matches_chronological_reference_with_gaps_reordering_and_real_times(self):
        rng = np.random.RandomState(133)
        ids = np.array([[rng.permutation(4)[:2] for _ in range(20)] for _ in range(3)], np.int32)
        active = rng.randint(2, size=ids.shape).astype(np.float32)
        times = np.cumsum(rng.randint(1, 4, size=(3, 20)), axis=1)
        written = tf.reshape(tf.range(3*20*2*3*9, dtype=tf.float32), [3, 20, 2, 3, 9])
        bank = tf.reshape(written, [-1, 3, 9])
        for expiry in (None, 8):
            plan = routing_plan(ids, active, 4, expiry, 7, times)
            for lane in range(3):
                manager = ReferenceMemoryManager(MemoryPolicy(4, 7, expiry), 3, 9)
                state = manager.zero_state(1)
                for t in range(20):
                    expected = manager.gather(state, ids[lane:lane+1, t], times[lane, t], active[lane:lane+1, t])
                    valid = tf.constant(plan["valid"][lane:lane+1, t])
                    observed = tf.gather(bank, plan["indices"][lane:lane+1, t])
                    actual = PilotHistory(tf.where(valid[..., None, None], observed, 0.),
                        plan["age"][lane:lane+1, t], plan["ports"][lane:lane+1, t], valid)
                    for a, b in zip(actual, expected, strict=True):
                        np.testing.assert_array_equal(a, b)
                    state = manager.scatter(state, ids[lane:lane+1, t], written[lane:lane+1, t], active[lane:lane+1, t],
                                            times[lane, t])
                    for user in range(2):
                        used = plan["indices"][lane, t, user][plan["valid"][lane, t, user]]
                        self.assertTrue(np.all(used//2 < lane*20+t))

    def test_invalid_metadata_is_rejected(self):
        for ids, active in [([[[0, 0]]], [[[1, 1]]]), ([[[-1, 1]]], [[[1, 1]]]),
                            ([[[0, 4]]], [[[1, 1]]]), ([[[0, 1]]], [[[1, np.nan]]]),
                            ([[[0., 1.]]], [[[1, 1]]])]:
            with self.assertRaises(ValueError):
                routing_plan(np.asarray(ids), np.asarray(active), 4, 8, 7)
        with self.assertRaisesRegex(ValueError, "timestamps"):
            routing_plan(np.array([[[0], [0]]]), np.ones([1, 2, 1]), 4, 8, 7, [[1, 1]])


class FullHistoryTrainingTests(unittest.TestCase):
    def test_chunked_gradients_match_sequential_reference_with_all_loss_terms(self):
        from test_memory_receiver import build_model, inputs
        from neural_rx.receiver.training import per_user_bce
        tf.keras.utils.set_random_seed(205)
        tf32 = tf.config.experimental.tensor_float_32_execution_enabled()
        self.addCleanup(tf.config.experimental.enable_tensor_float_32_execution, tf32)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        model = build_model()
        model(inputs(), tf.constant([.1]))
        teacher = build_model().base
        variables = model.optimization_variables()
        receiver = SimpleNamespace(_neural_rx=None)
        trainer = SimpleNamespace(model=model, variant="pilot_memory", optimization_variables=variables,
            contexts={1:(None, SimpleNamespace(_receiver=receiver))}, manager=MemoryPolicy(4, 7, 8),
            recipe={"channel_loss_weight": .3, "distillation_weight": .25}, teacher=teacher)
        layout = model.layouts[(2,12,4)]
        positions = layout.code_positions
        def gather(grid):
            return tf.gather(tf.reshape(grid, [tf.shape(grid)[0], 2, -1, grid.shape[-1]]), positions, axis=2)
        def mapping(grid):
            return tf.reshape(gather(grid), [tf.shape(grid)[0], 2, -1])
        coded_length = len(positions)*4
        code_map = SimpleNamespace(grid_to_coded=mapping)
        def prepare(receiver, y, ls, active):
            return [y, tf.zeros([2, 12, 4, 2]), ls, active, tf.ones([tf.shape(y)[0], 2, 1])]
        rng = np.random.RandomState(27)
        packets = []
        for length, lanes in ((6, 1), (3, 1)):
            ids = np.tile([[[0, 1]]], [lanes, length, 1]).astype(np.int32)
            ids[:, 1::2] = ids[:, 1::2, ::-1]
            active = np.ones_like(ids, np.float32)
            active[:, 1:3, 0] = 0
            y = rng.normal(size=[lanes, length, 12, 4, 8]).astype(np.float32)
            ls = rng.normal(size=[lanes, length, 2, 12, 4, 8]).astype(np.float32)
            coded = rng.randint(2, size=[lanes, length, 2, coded_length]).astype(np.float32)
            bits = coded[..., :coded_length//2].copy()
            noise = np.full([lanes, length], .1, np.float32)
            truth = rng.normal(size=[lanes, length, 2, 12, 4, 8]).astype(np.float32)
            packets.append(dict(y=y, ls=ls, bits=bits, no=noise, active=active, ue_ids=ids, h=truth,
                                coded=coded, metadata={}))
        den = sum(p["active"][0].sum() for p in packets)
        with tf.GradientTape() as tape:
            total = tf.constant(0.)
            for packet in packets:
                lanes, length = packet["active"].shape[:2]
                manager = ReferenceMemoryManager(trainer.manager, model.layouts[(2,12,4)].count, 9)
                state = manager.zero_state(lanes)
                for t in range(length):
                    active = packet["active"][:, t]
                    sample = prepare(None, packet["y"][:, t], packet["ls"][:, t], active)
                    history = manager.gather(state, packet["ue_ids"][:, t], t, active)
                    observation = model.observe(sample[2], packet["no"][:, t])
                    grid, _, final = model.detect_state(sample, packet["no"][:, t], history, observation)
                    logits = code_map.grid_to_coded(grid)
                    taught = teacher.initialize_state(sample)
                    for iteration in teacher._iterations:
                        taught = iteration([taught, sample[1], sample[3]])
                    soft = tf.sigmoid(code_map.grid_to_coded(teacher._readout_llrs[0](taught)))
                    truth = packet["h"][:, t]*model.base.input_scale(sample[0])[:, None]
                    loss = (.75*per_user_bce(packet["coded"][:, t], logits)+.25*per_user_bce(soft, logits)
                            +.3*tf.reduce_mean(tf.square(model.channel_estimate(final)-truth), [2, 3, 4]))
                    total += tf.reduce_sum(active*loss)/(den*lanes)
                    # the memory keeps the LS estimate of the TB
                    state = manager.scatter(state, packet["ue_ids"][:, t], observation, active, t)
        expected = tape.gradient(total, variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        with mock.patch('neural_rx.receiver.training_execution.prepare_cgnn_inputs', prepare), \
             mock.patch('neural_rx.receiver.training_execution.OneLayerGridCodeMap',
                        return_value=SimpleNamespace(grid_to_coded=mapping, gather_stream_grid=gather,
                                                     num_data_symbols=len(positions), bits_per_symbol=4)):
            for batch in (1, 4, 16):
                execution = BatchedTrainingExecution(trainer, batch)
                actual, metrics, fingerprints, used_den = execution.run(1, packets)
                self.assertEqual(used_den, den)
                np.testing.assert_allclose(metrics[0], total, rtol=1e-5, atol=2e-6)
                # the filter's fit is poorly conditioned on this one-PRB random data, so float32 rounding shows up
                for a, b in zip(actual, expected, strict=True):
                    np.testing.assert_allclose(a, b, rtol=3e-3, atol=1e-5)
                self.assertGreater(metrics[1], 0.)


if __name__ == '__main__':
    unittest.main()
