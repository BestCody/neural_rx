import json
from pathlib import Path
import unittest
import tempfile
from unittest import mock

import numpy as np
import tensorflow as tf

from pilot_history_receiver.training import MemoryTrainer, validate_recipe
from pilot_history_receiver.training_data import domain_seed, noise_schedule, sample_schedule, TrainingEpisodes
from temporal_benchmark_reference import frozen_result_cache

ROOT = Path(__file__).resolve().parent.parent


class MemoryTrainingTest(unittest.TestCase):
    def setUp(self):
        tf.keras.utils.set_random_seed(52)
        self.recipe = json.loads((ROOT/"config/memory_training.json").read_text())


    def test_history_settings_reject_invalid_episode_lengths_and_gaps(self):
        for changes in ({"episode_tbs": [2], "episode_probabilities": [1.]},
                        {"episode_tbs": [4], "episode_probabilities": [1.]},
                        {"schedule_gap_tbs": [0]}, {"paired_suffix_tbs": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_recipe(dict(self.recipe, **changes))
        validate_recipe(dict(self.recipe, episode_tbs=[8], episode_probabilities=[1.]))
        validate_recipe(dict(self.recipe, schedule_gap_tbs=[100]))

    def test_invalid_loss_and_plateau_settings_are_rejected(self):
        for changes in ({"learning_rate": 0.}, {"payload_loss_weight": 0.},
                        {"coded_loss_weight": float("nan")}, {"reset_weight": -1.},
                        {"minimum_learning_rate": 1.}, {"plateau_patience": 0},
                        {"plateau_factor": 1.}, {"learning_rate_warmup": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_recipe(dict(self.recipe, **changes))

    def test_data_domains_and_updates_have_repeatable_distinct_seeds(self):
        values=[domain_seed(9,2,3,d) for d in ("channel","payload","noise","schedule","paired_payload","paired_noise")]
        self.assertEqual(len(set(values)),6)
        self.assertEqual(values[0],domain_seed(9,2,3,"channel"))
        self.assertNotEqual(values[0],domain_seed(9,3,3,"channel"))

    def test_repeated_schedules_keep_gaps_attached_to_physical_users(self):
        switched, returned, empty, gaps = False, False, False, set()
        for seed in range(60):
            result = sample_schedule(self.recipe, 64, 4, 2, np.random.RandomState(seed), True)
            ids, active, metadata = result
            replay = sample_schedule(self.recipe, 64, 4, 2, np.random.RandomState(seed), True)
            np.testing.assert_array_equal(ids, replay[0])
            np.testing.assert_array_equal(active, replay[1])
            self.assertEqual(metadata, replay[2])
            np.testing.assert_array_equal(active[:, 0], 1.)
            self.assertTrue(np.all(ids[:, :, 0] != ids[:, :, 1]))
            switched |= np.count_nonzero(np.any(np.diff(np.sort(ids[0], axis=1), axis=0), axis=1)) > 1
            empty |= bool(np.any(active.sum(-1) == 0))
            for event in metadata['gaps']:
                first, last = event['start_tb'], min(64,event['start_tb']+event['duration_tbs'])
                selected = ids[0, first:last] == event['ue_id']
                np.testing.assert_array_equal(active[0, first:last][selected], 0.)
                returned |= bool(np.any((ids[0, last:] == event['ue_id']) & (active[0, last:] > 0)))
                gaps.add(event['duration_tbs'])
        self.assertTrue(switched and returned and empty)
        self.assertEqual(gaps, set(self.recipe['schedule_gap_tbs']))
        ids, active, _ = sample_schedule(self.recipe, 64, 4, 2, np.random.RandomState(9), False)
        np.testing.assert_array_equal(ids, np.repeat(ids[:, :1],64,axis=1))
        np.testing.assert_array_equal(active,1.)

    def test_channel_changes_and_weak_link_noise_are_independent_of_scheduling(self):
        from types import SimpleNamespace
        shape = [1,8,1,1,4,1,1,1]
        generator = SimpleNamespace(num_scheduled_tx=2, p=SimpleNamespace(pilots=np.zeros([20])),
            _activate_seed_batch=mock.Mock(), sample_pool_channel=mock.Mock(return_value=tf.ones(shape,tf.complex64)),
            gather_scheduled_channel=lambda pool,ids: pool)
        common = dict(self.recipe, cold_fraction=0., paired_history_fraction=0., episode_tbs=[8],
            episode_probabilities=[1.], channel_families=['uma'], channel_probabilities=[1.],
            power_imbalance_probability=1., power_gap_db=[12.,12.], weak_link_noise_probability=1.)
        episodes = TrainingEpisodes({},common,123)
        for schedule in (0.,1.):
            for change in (0.,1.):
                episodes.recipe = dict(common,schedule_probability=schedule,channel_change_probability=change)
                with mock.patch.object(episodes,'_generator',return_value=generator):
                    plan = episodes.plan(4,2,132)
                metadata = plan['metadata']
                self.assertEqual(metadata['schedule']['varying'],bool(schedule))
                self.assertEqual(metadata['channel_change'],bool(change))
                self.assertEqual(metadata['family'],'uma')
                self.assertEqual(metadata['noise_reference'],'weakest_pool_ue')
                np.testing.assert_allclose(plan['ebno_db'][0]+min(metadata['power_db_by_id']),
                    metadata['nominal_weak_ebno_db_by_tb'],atol=2e-6)
                if change:
                    regime = generator.sample_pool_channel.call_args.args[2].numpy()
                    self.assertFalse(np.any(regime[:, :metadata['change_tb']]))
                    self.assertTrue(np.all(regime[:, metadata['change_tb']:, metadata['change_ue_id']]))
                else:
                    self.assertIsNone(generator.sample_pool_channel.call_args.args[2])
                self.assertEqual(set(plan),{'channel','active','ue_ids','slot_number','ebno_db','metadata'})
                seed = domain_seed(123,4,2,'schedule')
                expected = sample_schedule(episodes.recipe,8,4,2,np.random.RandomState(seed),bool(schedule))
                np.testing.assert_array_equal(plan['ue_ids'],expected[0])
                np.testing.assert_array_equal(plan['active'],expected[1])
                episodes.recipe = dict(episodes.recipe,weak_link_noise_probability=0.)
                with mock.patch.object(episodes,'_generator',return_value=generator):
                    common_reference = episodes.plan(4,2,132)
                np.testing.assert_allclose(common_reference['ebno_db'],
                    plan['ebno_db']+min(metadata['power_db_by_id']),atol=2e-6)
                np.testing.assert_array_equal(common_reference['ue_ids'],plan['ue_ids'])

    def test_noise_profiles_are_repeatable_bounded_and_include_both_directions(self):
        settings = self.recipe["noise_schedule"]
        changes, constants = [], 0
        for seed in range(100):
            def draw():
                return noise_schedule(settings, 64, np.random.RandomState(seed), np.random.RandomState(seed+100))
            value = draw()
            np.testing.assert_array_equal(value, draw())
            self.assertGreaterEqual(value.min(), settings["range_db"][0])
            self.assertLessEqual(value.max(), settings["range_db"][1])
            delta = float(value[0,-1]-value[0,0])
            self.assertLessEqual(abs(delta), settings["max_change_db"]+1e-6)
            if delta:
                changes.append(delta)
                self.assertEqual(np.count_nonzero(np.diff(value[0])), 1)
            else:
                constants += 1
        self.assertGreater(constants, 0)
        self.assertGreater(min(changes)*max(changes)*-1, 0)
        one = noise_schedule(settings, 1, np.random.RandomState(1), np.random.RandomState(2))
        self.assertEqual(one.shape, (1,1))

    def test_invalid_noise_distributions_fail_validation(self):
        for change in ({"change_probability":float("nan")}, {"core_probability":2},
                       {"max_change_db":0}, {"core_range_db":[-3,6]}, {"range_db":[0]}):
            noise = dict(self.recipe["noise_schedule"], **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_recipe(dict(self.recipe, noise_schedule=noise))

    def test_frozen_cache_reuses_results_and_rejects_corrupt_bytes(self):
        calls = []
        expected = {"errors":np.zeros([3],bool)}
        def compute():
            calls.append(1)
            return expected, {"checked":True}, 1.
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("numpy.load", side_effect=AssertionError("fresh results must stay in memory")):
                arrays, _, _ = frozen_result_cache(directory,{"reference":"frozen"},compute)
                self.assertIs(arrays, expected)
            arrays, checks, _ = frozen_result_cache(directory,{"reference":"frozen"},compute)
            self.assertEqual(len(calls),1)
            self.assertTrue(checks["checked"])
            np.testing.assert_array_equal(arrays["errors"], expected["errors"])
            raw = next(Path(directory).glob("*/errors.npz"))
            raw.write_bytes(b"damaged")
            with self.assertRaises(ValueError):
                frozen_result_cache(directory,{"reference":"frozen"},compute)

    def test_failed_reference_write_does_not_publish_cache(self):
        def compute():
            return {"errors":np.zeros([3],bool)}, {"checked":True}, 1.
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("temporal_benchmark_reference.atomic_json", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    frozen_result_cache(directory, {"reference":"frozen"}, compute)
            self.assertFalse(any(p.is_dir() for p in Path(directory).iterdir()))
            arrays, _, _ = frozen_result_cache(directory, {"reference":"frozen"}, compute)
            self.assertFalse(np.any(arrays["errors"]))



class JointUpdateTest(unittest.TestCase):
    def test_real_eight_example_update_and_checkpoint_continuation(self):
        self.check_update(12)

    def test_full_bandwidth_eight_example_update_and_checkpoint_continuation(self):
        self.check_update(132)

    def check_update(self, width):
        tf.config.experimental.enable_op_determinism()
        recipe = json.loads((ROOT/"config/memory_training.json").read_text())
        recipe.update(prbs=[width], prb_probabilities=[1.], episode_tbs=[8], episode_probabilities=[1.],
                      channel_families=["tdl"], channel_probabilities=[1.], cold_fraction=0.,
                      paired_history_fraction=0., schedule_probability=0., channel_change_probability=0.)
        tf32 = tf.config.experimental.tensor_float_32_execution_enabled()
        self.addCleanup(tf.config.experimental.enable_tensor_float_32_execution,tf32)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        trainer = MemoryTrainer(recipe,detector_batch_tbs=4)
        packets = []
        for index in range(recipe["effective_examples"]):
            plan = trainer.episodes.plan(1,index,width)
            if index == 0:
                plan["metadata"]["paired"] = True
                plan["metadata"]["paired_suffix_tbs"] = 4
            if index == 1:
                plan["active"][:,2:5,0] = 0
            packets.append(trainer.episodes.render(plan))
        for packet in packets:
            bits = packet["bits"]
            expected = trainer.contexts[width][1]._receiver._tb_encoders[0](
                tf.reshape(bits,[-1,*bits.shape[2:]]))
            np.testing.assert_array_equal(packet["coded"],tf.reshape(expected,packet["coded"].shape))
        paired = packets[0]
        self.assertFalse(np.array_equal(paired["bits"][0,:4],paired["bits"][1,:4]))
        for key in ("y","ls","bits","coded","no","channel_targets"):
            np.testing.assert_array_equal(paired[key][0,-4:],paired[key][1,-4:])
        groups = {"filter":trainer.model.channel_memory.filter.trainable_variables,
                  "uncertainty":trainer.model.channel_memory.error.trainable_variables,
                  "correction_gain":[trainer.model.correction_logit],
                  "source_calibration":[trainer.model.evidence_calibration],
                  "blend_head":[trainer.model.blend_output, trainer.model.blend_output_bias],
                  "initializer":trainer.model.base._s_init[0].trainable_variables,
                  "round1":trainer.model.base._iterations[0].trainable_variables,
                  "round2":trainer.model.base._iterations[1].trainable_variables,
                  "readout":trainer.model.base._readout_llrs[0].trainable_variables}
        from receiver_decoding import OneLayerGridCodeMap, DecoderAssistedTB
        from temporal_receiver_core import prepare_cgnn_inputs
        from pilot_history_receiver.pilot_history import PilotHistory
        receiver = trainer.contexts[width][1]._receiver
        mapping = OneLayerGridCodeMap(receiver._neural_rx, trainer.contexts[width][0].max_num_tx,
                                      trainer.model.base._num_bits_per_symbol[0])
        layout = trainer.model.layouts[(trainer.contexts[width][0].max_num_tx, 12*width, 14)]
        for row in np.asarray(mapping.data_indices):
            np.testing.assert_array_equal((row % layout.frequency)*layout.symbols+row//layout.frequency,
                                          layout.code_positions)
        architecture = trainer.settings['architecture']
        decoder = DecoderAssistedTB(receiver._tb_encoders[0], receiver._tb_decoders[0],
                                    architecture['ldpc_iterations'], architecture['ldpc_first_iterations'])
        length = trainer.model.history_tbs
        observed = tf.stack([trainer.model.observe(paired['ls'][:, t], paired['no'][:, t]) for t in range(length)], 2)
        metadata = observed.shape[:3]
        history = PilotHistory(observed,
            tf.broadcast_to(tf.range(length, 0, -1)[None, None], metadata),
            tf.broadcast_to(tf.range(metadata[1])[None, :, None], metadata), tf.ones(metadata, tf.bool))
        sample = prepare_cgnn_inputs(receiver, paired['y'][:, length], paired['ls'][:, length], paired['active'][:, length])
        from test_memory_receiver import activate_physical_path
        activate_physical_path(trainer.model, sample)
        @tf.function
        def payload_gradients():
            with tf.GradientTape() as tape:
                output, _, _, _, _, _ = trainer.model.decode(sample, paired['no'][:, length],
                    paired['ue_ids'][:, length], history, decoder, mapping)
                logits = output.logits
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=paired['bits'][:, length], logits=logits))
            return tape.gradient(loss, trainer.optimization_variables)
        gradients = payload_gradients()
        gradient_map = {id(v): g for v, g in zip(trainer.optimization_variables, gradients)}
        for variable in (trainer.model.blend_kernel, trainer.model.blend_bias):
            np.testing.assert_array_equal(gradient_map[id(variable)], 0.)
        for name, variables in groups.items():
            with self.subTest(payload_gradient=name, width=width):
                selected = [gradient_map[id(v)] for v in variables]
                self.assertTrue(all(g is not None and np.isfinite(g).all() for g in selected))
                values = np.concatenate([np.asarray(g,np.float64).ravel() for g in selected])
                self.assertGreater(float(np.linalg.norm(values)), 0.)
        before = {name:[v.numpy().copy() for v in variables] for name,variables in groups.items()}
        record, gradients = trainer.update(packets)
        self.assertEqual(int(trainer.step.numpy()),1)
        self.assertEqual(int(trainer.optimizer.iterations.numpy()),1)
        self.assertEqual(record["active_user_blocks"],sum(float(tf.reduce_sum(p["active"][0])) for p in packets))
        self.assertGreater(record["memory_gradient_norm"],0.)
        self.assertTrue(all(np.isfinite(g).all() for g in gradients))
        for name,variables in groups.items():
            with self.subTest(component=name):
                self.assertTrue(any(np.any(v.numpy()!=old) for v,old in zip(variables,before[name])))
        self.assertAlmostEqual(record["learning_rate"],recipe["learning_rate"]/recipe["learning_rate_warmup"],places=10)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = trainer.save(directory/"before_feedback")
            feedback = dict(phase="development",checkpoint=str(checkpoint),bank_sha256="fixed-bank",
                result_sha256="result-1",summary={"candidate_persistent":dict(core_errors=3,core_blocks=13,core_tbler=3/13)})
            trainer.observe_development(feedback)
            self.assertEqual(trainer.observe_development(feedback),trainer.schedule_state())
            trainer.learning_rate.assign(recipe["learning_rate"]*recipe["plateau_factor"])
            trainer.plateau_count.assign(recipe["plateau_patience"]-1)
            checkpoint = trainer.save(directory/"continued")
            saved = [v.numpy().copy() for v in trainer.optimizer.variables()]
            resumed = MemoryTrainer(recipe,detector_batch_tbs=7)
            resumed.restore(checkpoint)
            self.assertEqual(resumed.schedule_state(),trainer.schedule_state())
            for variable,expected in zip(resumed.optimizer.variables(),saved):
                np.testing.assert_array_equal(variable,expected)
            reference, reference_gradients = trainer.update(packets)
            for variable in (trainer.model.blend_kernel, trainer.model.blend_bias):
                index = next(i for i,v in enumerate(trainer.optimization_variables) if v is variable)
                self.assertGreater(float(tf.linalg.norm(reference_gradients[index])), 0.)
            actual, actual_gradients = resumed.update(packets)
            self.assertEqual(reference["episode_fingerprints"],actual["episode_fingerprints"])
            self.assertEqual(int(resumed.optimizer.iterations.numpy()),2)
            for key in ("loss","persistent_coded_bce","persistent_payload_loss","reset_coded_bce","reset_payload_loss",
                        "persistent_channel_nll","reset_channel_nll"):
                np.testing.assert_allclose(actual[key],reference[key],rtol=1e-5,atol=2e-6)
            expected_flat = np.concatenate([np.asarray(g).ravel() for g in reference_gradients])
            actual_flat = np.concatenate([np.asarray(g).ravel() for g in actual_gradients])
            difference = actual_flat-expected_flat
            gradient_l2 = float(np.linalg.norm(difference)/max(np.linalg.norm(expected_flat), 1e-12))
            gradient_max = float(np.max(np.abs(difference)))
            print(json.dumps(dict(check='restart_with_different_batch_size', prbs=width,
                                  relative_gradient_l2=gradient_l2, maximum_gradient_difference=gradient_max)))
            for a,b in zip(actual_gradients,reference_gradients):
                a, b = np.asarray(a), np.asarray(b)
                self.assertLessEqual(np.linalg.norm(a-b), 1e-3*np.linalg.norm(b)+1e-6)
                self.assertLessEqual(np.max(np.abs(a-b)), 1e-3*np.max(np.abs(b))+1e-6)
            for a,b in zip(resumed.optimization_variables,trainer.optimization_variables):
                np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
            self.assertEqual(resumed.schedule_state(),trainer.schedule_state())

    def test_plateau_uses_pooled_counts_and_rejects_mismatched_feedback(self):
        recipe = json.loads((ROOT/"config/memory_training.json").read_text())
        trainer = MemoryTrainer.__new__(MemoryTrainer)
        trainer.recipe = recipe
        trainer.step = tf.Variable(0,dtype=tf.int64)
        trainer.learning_rate = tf.Variable(recipe["learning_rate"],dtype=tf.float32)
        trainer.best_tbler = tf.Variable(np.inf,dtype=tf.float64)
        for name,value in (("best_update",-1),("plateau_count",0),("development_update",-1)):
            setattr(trainer,name,tf.Variable(value,dtype=tf.int64))
        for name in ("development_bank","development_result"):
            setattr(trainer,name,tf.Variable("",dtype=tf.string))
        from pilot_history_receiver.training import recipe_hash
        binding = dict(recipe=recipe,recipe_sha256=recipe_hash(recipe),execution={})
        trainer.binding = lambda: binding
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            def feedback(update, errors, blocks):
                trainer.step.assign(update)
                (checkpoint/"manifest.json").write_text(json.dumps(dict(update=update,binding=binding)))
                return dict(phase="development",checkpoint=str(checkpoint),bank_sha256="fixed-bank",
                    result_sha256=f"result-{update}",summary={"candidate_persistent":
                    dict(core_errors=errors,core_blocks=blocks,core_tbler=errors/blocks)})
            state = trainer.observe_development(feedback(0,3,13))
            for update in range(1,recipe["plateau_patience"]+1):
                result = feedback(update,4,13)
                state = trainer.observe_development(result)
            self.assertAlmostEqual(state["learning_rate"],recipe["learning_rate"]*recipe["plateau_factor"],places=10)
            self.assertEqual(state["plateau_count"],0)
            self.assertEqual(trainer.observe_development(result),state)
            with self.assertRaisesRegex(ValueError,"changed"):
                trainer.observe_development(dict(result,result_sha256="different"))
            result = feedback(update+1,2,13)
            for changed in (dict(result,phase="confirmation"),dict(result,bank_sha256="other-bank"),
                dict(result,summary={"candidate_persistent":dict(core_errors=2,core_blocks=13,core_tbler=.5)})):
                with self.assertRaises(ValueError):
                    trainer.observe_development(changed)
                self.assertEqual(trainer.schedule_state(),state)
            trainer.observe_development(result)
            self.assertAlmostEqual(float(trainer.best_tbler.numpy()),2/13)
            trainer.learning_rate.assign(recipe["minimum_learning_rate"])
            for extra in range(1,recipe["plateau_patience"]+1):
                trainer.observe_development(feedback(update+1+extra,4,13))
            self.assertAlmostEqual(float(trainer.learning_rate.numpy()),recipe["minimum_learning_rate"],places=12)

if __name__ == "__main__":
    unittest.main()
