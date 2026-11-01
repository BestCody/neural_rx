import json
from pathlib import Path
from threading import Lock, get_ident
import subprocess
import os
import shutil
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from pilot_history_receiver.training import MemoryTrainer
from pilot_history_receiver.training_parallel import GradientReplica, partition_episodes
from pilot_history_receiver.training_transport import SharedArrayStore


class ParallelUpdateTests(unittest.TestCase):
    def test_two_gpu_gradients_match_one_gpu_for_the_same_eight_episodes(self):
        tf.config.experimental.enable_op_determinism()
        if shutil.which('nvidia-smi') is None:
            self.skipTest('NVIDIA GPUs are required')
        devices = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                                 capture_output=True, text=True)
        if devices.returncode or len(devices.stdout.splitlines()) < 2:
            self.skipTest('Two GPUs are required')
        primary = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
        if not primary.isdecimal():
            self.skipTest('This test requires a numeric primary GPU selection')
        secondary = next(int(i) for i in devices.stdout.splitlines() if i.strip() != primary)
        recipe = json.loads((Path(__file__).resolve().parents[1]/'config/memory_training.json').read_text())
        recipe.update(prbs=[4], prb_probabilities=[1.], channel_families=['tdl'],
                      channel_probabilities=[1.], cold_fraction=0., paired_history_fraction=0.,
                      episode_tbs=[8], episode_probabilities=[1.],
                      schedule_probability=0., channel_change_probability=0.)
        tf32 = tf.config.experimental.tensor_float_32_execution_enabled()
        self.addCleanup(tf.config.experimental.enable_tensor_float_32_execution, tf32)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        trainer = MemoryTrainer(recipe, detector_batch_tbs=4)
        trainer.model.blend_output.assign(tf.random.stateless_normal(trainer.model.blend_output.shape,[317,9],stddev=.1))
        packets = [trainer.episodes.sample(1, index, 4) for index in range(8)]
        expected, gradients = trainer.update(packets, apply=False)
        gradient_map = {id(v): g for v, g in zip(trainer.optimization_variables, gradients)}
        for query in (trainer.model.channel_memory.filter, trainer.model.channel_memory.error):
            self.assertGreater(float(tf.linalg.global_norm([gradient_map[id(v)]
                for v in query.trainable_variables])), 0.)
        self.assertGreater(float(tf.linalg.norm(gradient_map[id(trainer.model.evidence_calibration)])), 0.)
        self.assertGreater(float(tf.linalg.norm(gradient_map[id(trainer.model.blend_kernel)])), 0.)
        with GradientReplica(trainer, secondary, memory_mib=8192) as replica:
            trainer.replica = replica
            actual, distributed = trainer.update(packets, apply=False)
        trainer.replica = None
        self.assertEqual(int(trainer.step), 0)
        self.assertEqual(actual['episode_fingerprints'], expected['episode_fingerprints'])
        self.assertEqual(actual['active_user_blocks'], expected['active_user_blocks'])
        for key in ('loss', 'persistent_coded_bce', 'persistent_payload_loss', 'reset_coded_bce', 'reset_payload_loss',
                    'persistent_channel_nll', 'reset_channel_nll'):
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-5, atol=2e-6)
        for a, b in zip(distributed, gradients):
            np.testing.assert_allclose(a, b, rtol=3e-3, atol=3e-6)

    def test_partition_keeps_paired_episodes_whole_and_covers_update_once(self):
        packets = [dict(active=np.ones([lanes, length, 2])) for lanes, length in
                   ((2, 64), (1, 8), (2, 32), (1, 16), (1, 4), (2, 8), (1, 64), (1, 16))]
        local, remote = partition_episodes(packets)
        self.assertEqual(sorted(local+remote), list(range(8)))
        self.assertEqual((local, remote), partition_episodes(packets))
        self.assertTrue(local and remote)
        total = sum(p["active"][0].sum() for p in packets)
        self.assertEqual(sum(packets[i]["active"][0].sum() for i in local)+
                         sum(packets[i]["active"][0].sum() for i in remote), total)

    def test_failed_or_stale_replica_never_applies_partial_update(self):
        recipe = json.loads((Path(__file__).resolve().parents[1]/"config/memory_training.json").read_text())
        recipe.update(effective_examples=2, prbs=[12], prb_probabilities=[1.])
        trainer = MemoryTrainer.__new__(MemoryTrainer)
        trainer.recipe, trainer.seed = recipe, recipe["seed"]
        trainer.step = tf.Variable(0, dtype=tf.int64)
        trainer.learning_rate = tf.Variable(recipe["learning_rate"])
        trainer._update_lock = Lock()
        trainer.apply = mock.Mock()
        trainer.flat_weights = lambda: tf.constant([1.])
        trainer.execution = SimpleNamespace(run=mock.Mock(return_value=((), tf.zeros([10]), ["local"], 4.)))
        packets = [dict(active=np.ones([1, 2, 2])) for _ in range(2)]
        for response in (RuntimeError("worker exited"), dict(kind="gradients", update=0, indices=[1])):
            replica = GradientReplica.__new__(GradientReplica)
            replica.owner, replica.closed, replica.pending = get_ident(), False, None
            replica.arrays = SharedArrayStore()
            self.addCleanup(replica.arrays.close)
            replica.connection = mock.Mock()
            replica._receive = mock.Mock(side_effect=response if isinstance(response, Exception) else None,
                                         return_value=response)
            trainer.replica = replica
            with self.assertRaises((RuntimeError, ValueError)):
                trainer.update(packets)
            trainer.apply.assert_not_called()
            self.assertEqual(int(trainer.step.numpy()), 0)
            self.assertFalse(trainer._update_lock.locked())


if __name__ == "__main__":
    unittest.main()
