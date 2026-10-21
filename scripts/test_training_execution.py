from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from pilot_history_receiver.pilot_history import MemoryPolicy, PilotHistory
from pilot_history_receiver.training_execution import BatchedTrainingExecution, routing_plan, channel_nll
from verification.pilot_history import ReferenceMemoryManager


class TrainingRoutingTests(unittest.TestCase):
    def test_batched_renderer_preserves_per_tb_seeds_pilots_and_schedule(self):
        from temporal_receiver_core import build_backbone
        from temporal_training_data import TemporalTrainingDataGenerator
        from pilot_history_receiver.training_render import TrainingRenderer
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
                coded=generator.tx._tb_encoder(bits), channel_targets=renderer.layout.channel_targets(
                    generator.rx.preprocess_channel_ground_truth(channel[:, t]))))
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
        from utils import Parameters, E2E_Model
        from pilot_history_receiver.training_data import TrainingEpisodes
        recipe = json.loads((Path(__file__).resolve().parents[1]/"config/memory_training.json").read_text())
        parameters = Parameters(recipe["radio_config"], training=True, n_size_bwp_override=4)
        model = E2E_Model(parameters, training=True)
        full = TrainingEpisodes({4: (parameters, model)}, recipe, recipe["seed"])
        channels = TrainingEpisodes({4: (parameters, None)}, recipe, recipe["seed"])
        streams = {name: 1000+i for i, name in enumerate(("channel", "payload", "noise", "schedule"))}
        for family in ("tdl", "umi", "uma"):
            for changed in (False, True):
                regime = np.zeros((1, 3, 4), bool); regime[:, 1:, 0] = changed
                outputs = []
                for context in (full, channels):
                    generator = context._generator(4, family, 0)
                    generator._activate_seed_batch(streams)
                    outputs.append(generator.sample_pool_channel(1, 3, tf.constant(regime)).numpy())
                np.testing.assert_array_equal(*outputs)

    def test_sharing_rejects_different_paired_futures(self):
        trainer = SimpleNamespace(manager=MemoryPolicy(4, 7, 8), model=SimpleNamespace(history_tbs=7))
        execution = BatchedTrainingExecution(trainer)
        packet = dict(y=np.zeros([2, 4, 1]), ls=np.zeros([2, 4, 1]), bits=np.zeros([2, 4, 1]),
                      ue_ids=np.tile([[[0, 1]]], [2, 4, 1]), active=np.ones([2, 4, 2]),
                      coded=np.zeros([2, 4, 1]), channel_targets=np.zeros([2, 4, 1]), no=np.ones([2, 4]), metadata={"paired_suffix_tbs":2})
        with mock.patch.object(execution, "describe", return_value=tf.zeros([2, 4, 2, 6, 9])):
            _, _, den = execution.prepare([packet])
        self.assertEqual(den, 8.)
        for key in ("y", "ls", "bits", "coded", "channel_targets"):
            packet[key][1, 2, 0] = 1
            with self.assertRaisesRegex(ValueError, "identical future"):
                execution.prepare([packet])
            packet[key][1, 2, 0] = 0

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
                    for a, b in zip(actual, expected):
                        np.testing.assert_array_equal(a, b)
                    state = manager.scatter(state, ids[lane:lane+1, t], written[lane:lane+1, t], active[lane:lane+1, t], times[lane, t])
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
    def test_chunked_gradients_match_sequential_reference_with_both_losses(self):
        from test_memory_receiver import build_model, inputs, activate_physical_path
        from pilot_history_receiver.training import per_user_bce, per_user_payload_loss
        tf.keras.utils.set_random_seed(205)
        tf32 = tf.config.experimental.tensor_float_32_execution_enabled()
        self.addCleanup(tf.config.experimental.enable_tensor_float_32_execution, tf32)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        model = build_model(); activate_physical_path(model, inputs())
        model.blend_output.assign(tf.random.normal(model.blend_output.shape, stddev=.1))
        memory_variables, backbone_variables = model.optimization_variables()
        variables = memory_variables+backbone_variables
        receiver = SimpleNamespace(_neural_rx=None, _tb_encoders=[None],
            _tb_decoders=[SimpleNamespace(_decoder=SimpleNamespace(_num_iter=20))])
        trainer = SimpleNamespace(model=model, memory_variables=memory_variables, optimization_variables=variables,
            contexts={1:(None, SimpleNamespace(_receiver=receiver))}, manager=MemoryPolicy(4, 7, 8),
            settings={"architecture":{"ldpc_iterations":20,"ldpc_first_iterations":8}},
            recipe={"coded_loss_weight":1., "payload_loss_weight":1., "reset_weight":.5, "channel_loss_weight":.1})
        layout = model.layouts[(2,12,4)]
        positions = layout.code_positions
        def gather(grid):
            return tf.gather(tf.reshape(grid, [tf.shape(grid)[0], 2, -1, grid.shape[-1]]), positions, axis=2)
        mapping = lambda grid: tf.reshape(gather(grid), [tf.shape(grid)[0], 2, -1])
        coded_length = len(positions)*4
        payload = lambda llr: llr[..., :coded_length//2]+.3*tf.tanh(llr[..., coded_length//2:])
        to_blocks = lambda x: tf.reshape(x, [tf.shape(x)[0], 2, -1, 4])
        from_blocks = lambda x: tf.reshape(x, [tf.shape(x)[0], 2, coded_length])
        decoder = SimpleNamespace(evidence=lambda x: (to_blocks(x), to_blocks(.3*tf.tanh(x)), x,
                                                      to_blocks(x+.3*tf.tanh(x))),
            finish=lambda x,state: SimpleNamespace(logits=payload(from_blocks(x))+.1*payload(state)),
            first=SimpleNamespace(from_code_blocks=from_blocks, llr_limit=20.),
            symbol_indices=tf.tile(tf.reshape(tf.range(coded_length), [1, -1, 4]), [2, 1, 1]),
            symbol_signs=tf.ones([2, coded_length//4, 4]))
        code_map = SimpleNamespace(grid_to_coded=mapping)
        def prepare(receiver, y, ls, active):
            return [y, tf.zeros([2, 12, 4, 2]), ls, active, tf.ones([tf.shape(y)[0], 2, 1])]
        rng = np.random.RandomState(27)
        packets = []
        for length, lanes in ((6, 2), (3, 1)):
            ids = np.tile([[[0, 1]]], [lanes, length, 1]).astype(np.int32)
            ids[:, 1::2] = ids[:, 1::2, ::-1]
            active = np.ones_like(ids, np.float32); active[:, 1:3, 0] = 0
            y = rng.normal(size=[lanes, length, 12, 4, 8]).astype(np.float32)
            ls = rng.normal(size=[lanes, length, 2, 12, 4, 8]).astype(np.float32)
            coded = rng.randint(2, size=[lanes, length, 2, coded_length]).astype(np.float32)
            bits = coded[..., :coded_length//2].copy(); noise = np.full([lanes, length], .1, np.float32)
            if lanes == 2:
                for value in (y, ls, bits, coded):
                    value[1, -2:] = value[0, -2:]
            targets = layout.channel_targets(tf.reshape(ls, [-1, 2, 12, 4, 8]))
            targets = tf.reshape(targets, [lanes, length, *targets.shape[1:]])
            packets.append(dict(y=y, ls=ls, bits=bits, no=noise, active=active, ue_ids=ids,
                                coded=coded, channel_targets=targets, metadata={"paired_suffix_tbs":2}))
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
                    for weight, saved in ((2/3, history), (1/3, None)):
                        observation = model.observe(sample[2], packet["no"][:, t])
                        saved = model.empty_history(observation) if saved is None else saved
                        output, logits, corrected, posterior, _, estimate = model.decode(sample, packet["no"][:, t],
                            tf.constant(packet["ue_ids"][:, t]), saved, decoder, code_map, observation)
                        nll = channel_nll(estimate, packet["channel_targets"][:, t], layout)
                        coded = packet["coded"][:, t]
                        loss = ((per_user_bce(coded, logits)+per_user_bce(coded, corrected)
                                 +per_user_bce(coded, posterior))/3.+
                                per_user_payload_loss(packet["bits"][:, t], output.logits)+.1*nll)/2
                        total += weight*tf.reduce_sum(active*loss)/(den*lanes)
                    state = manager.scatter(state, packet["ue_ids"][:, t], observation, active, t)
        expected = tape.gradient(total, variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        with mock.patch('pilot_history_receiver.training_execution.prepare_cgnn_inputs', prepare), \
             mock.patch('pilot_history_receiver.training_execution.OneLayerGridCodeMap', return_value=SimpleNamespace(grid_to_coded=mapping, gather_stream_grid=gather, num_data_symbols=len(positions), bits_per_symbol=4)), \
             mock.patch('pilot_history_receiver.training_execution.DecoderAssistedTB', return_value=decoder):
            for batch in (1, 4, 16):
                execution = BatchedTrainingExecution(trainer, batch)
                actual, metrics, fingerprints, used_den = execution.run(1, packets)
                self.assertEqual(used_den, den)
                np.testing.assert_allclose(metrics[0], total, rtol=1e-5, atol=2e-6)
                for a, b in zip(actual, expected):
                    np.testing.assert_allclose(a, b, rtol=3e-3, atol=3e-6)
                self.assertGreater(metrics[8], 0.)


if __name__ == '__main__':
    unittest.main()
