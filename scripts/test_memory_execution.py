import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from unittest import mock

import numpy as np
import tensorflow as tf

from pilot_history_receiver.pilot_history import validate_memory_metadata, MemoryPolicy, PilotMemoryBuffer
from receiver_decoding import OneLayerGridCodeMap
from verification.pilot_history import ReferenceMemoryManager




class PilotHistoryExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.config.experimental.enable_op_determinism()

    def test_shared_mapping_matches_demapper_values_and_gradients(self):
        from verification.radio import demap_llr
        from utils import Parameters, E2E_Model
        tf.keras.utils.set_random_seed(413)
        for prbs in (12,48,132):
            parameters=Parameters("nrx_large.cfg",training=True,n_size_bwp_override=prbs)
            model=E2E_Model(parameters,training=True)
            ofdm=model._receiver._neural_rx
            grid=model._transmitters[0].resource_grid
            users=grid.num_tx
            bits=ofdm._cgnn._num_bits_per_symbol[0]
            mapping=OneLayerGridCodeMap(ofdm,users,bits)
            for batch in (1,2):
                values=tf.Variable(tf.random.normal([batch,users,grid.fft_size,grid.num_ofdm_symbols,bits]))
                reference=demap_llr(ofdm,values,users)
                weighted=tf.random.normal(reference.shape)
                outputs=[]
                for call in (lambda:demap_llr(ofdm,values,users),lambda:mapping.grid_to_coded(values)):
                    with tf.GradientTape() as tape:
                        result=call()
                        loss=tf.reduce_sum(result*weighted)
                    outputs.append((result,tape.gradient(loss,values)))
                for left,right in zip(outputs[0],outputs[1]):
                    np.testing.assert_array_equal(left,right)



    def test_boundary_validation_rejects_invalid_inputs(self):
        validate_memory_metadata([[2,2]],[[1,0]],5)
        invalid=[([[-1,2]],[[1,1]]),([[0,5]],[[1,1]]),([[1,1]],[[1,1]]),
                 ([[1.,2.]],[[1,1]]),([[1,2]],[[1,np.nan]]),([[1,2]],[[1,.5]]),
                 ([1,2],[1,1]),([[1,2]],[[1]])]
        for ids,active in invalid:
            with self.subTest(ids=ids,active=active):
                with self.assertRaises(ValueError):
                    validate_memory_metadata(ids,active,5)


    def test_ring_matches_independent_chronological_history_and_rollback(self):
        tf.keras.utils.set_random_seed(411)
        rng = np.random.RandomState(411)
        for expiry in (None, 3):
            policy = MemoryPolicy(7, 4, expiry)
            manager = ReferenceMemoryManager(policy, 5, 9)
            buffer = PilotMemoryBuffer(policy, 2, 5, 9)
            head, count = buffer.empty_metadata()
            reference = manager.zero_state(2)
            gather = tf.function(buffer.gather, jit_compile=True)
            def read(head, count, ids, active, step):
                tables = tuple(v.read_value() for v in (buffer.observations, buffer.timestamps, buffer.ports))
                return gather(tables, head, count, ids, active, step)
            stage = tf.function(buffer.stage)
            for step in range(40):
                ids = np.stack([rng.choice(7, 3, replace=False) for _ in range(2)]).astype(np.int32)
                active = rng.rand(2, 3) > .4
                if step % 5 == 0:
                    ids[0, 1] = ids[0, 0]
                    active[0, 1] = False
                if step % 9 == 0:
                    active[:] = False
                validate_memory_metadata(ids, active, 7)
                expected = manager.gather(reference, ids, step, active)
                actual = read(head, count, ids, active, tf.constant(step))
                for a, b in zip(expected, actual):
                    np.testing.assert_array_equal(a, b)
                written = tf.random.normal([2, 3, 5, 9])
                if step % 4 == 0:
                    _, _, complete = stage(written*2., head, count, ids, active, tf.constant(step))
                    complete.numpy()
                    rolled_back = read(head, count, ids, active, tf.constant(step))
                    for a, b in zip(actual, rolled_back):
                        np.testing.assert_array_equal(a, b)
                next_head, next_count, complete = stage(written, head, count, ids, active, tf.constant(step))
                complete.numpy()
                head, count = next_head, next_count
                reference = manager.scatter(reference, ids, written, active, step)

    def test_expiry_is_per_record_and_reentry_does_not_rejuvenate_old_data(self):
        policy = MemoryPolicy(3, 4, 3)
        buffer = PilotMemoryBuffer(policy, 1, 2, 9)
        head, count = buffer.empty_metadata()
        ids, active = tf.constant([[0, 1]]), tf.constant([[True, True]])
        for time in (0, 2, 4):
            observed = tf.ones([1, 2, 2, 9])*float(time+1)
            head, count, complete = buffer.stage(observed, head, count, ids, active, tf.constant(time))
            complete.numpy()
        history = buffer.read(head, count, ids[:, ::-1], active, tf.constant(5))
        np.testing.assert_array_equal(history.valid[0, 0], [True, True, False, False])
        np.testing.assert_array_equal(history.age[0, 0], [1, 3, 0, 0])
        np.testing.assert_array_equal(history.ports[0, 0], [1, 1, 0, 0])
        empty = buffer.read(head, tf.zeros_like(count), ids, active, tf.constant(5))
        self.assertFalse(tf.reduce_any(empty.valid))
        np.testing.assert_array_equal(empty.observations, 0.)

    def test_pending_update_is_private_and_nonfinite_active_input_cannot_commit(self):
        for batch, users, capacity, history, cells, components in (
                (1, 2, 4, 7, 792, 9), (2, 3, 5, 3, 17, 5), (1, 1, 2, 1, 7, 3)):
            with self.subTest(batch=batch, users=users, cells=cells):
                buffer = PilotMemoryBuffer(MemoryPolicy(capacity, history, 8), batch, cells, components)
                head, count = buffer.empty_metadata()
                ids = np.broadcast_to(np.arange(users, dtype=np.int32), [batch, users]).copy()
                active = np.ones([batch, users], bool)
                observations = np.ones([batch, users, cells, components], np.float32)
                if users > 1:
                    ids[:, -1] = ids[:, 0]
                    active[:, -1] = False
                    observations[:, -1] = np.nan
                validate_memory_metadata(ids, active, capacity)
                tables = tuple(v.read_value() for v in (buffer.observations, buffer.timestamps, buffer.ports))
                before = [v.numpy().copy() for v in tables]
                pending, finite = buffer.prepare_stage(observations, head, count, ids, active)
                self.assertTrue(finite.numpy())
                for expected, current in zip(before, buffer.snapshot(head, count)[:3]):
                    np.testing.assert_array_equal(expected, current)
                head, count, complete = buffer.commit(pending, finite, tf.constant(0))
                complete.numpy()
                for expected, retained_snapshot in zip(before, tables):
                    np.testing.assert_array_equal(expected, retained_snapshot)
                for value in (buffer.observations, buffer.timestamps, buffer.ports, head, count):
                    self.assertIn('GPU:', value.device)
                retained = buffer.read(head, count, ids, active, tf.constant(1))
                np.testing.assert_array_equal(retained.valid[..., 0], active)
                np.testing.assert_array_equal(retained.observations[:, 0, 0], 1.)
                before = [v.numpy().copy() for v in buffer.snapshot(head, count)]
                observations[:, 0, 0, 0] = np.nan
                tables = tuple(v.read_value() for v in (buffer.observations, buffer.timestamps, buffer.ports))
                pending, finite = buffer.prepare_stage(observations, head, count, ids, active)
                self.assertFalse(finite.numpy())
                with self.assertRaises(tf.errors.InvalidArgumentError):
                    buffer.commit(pending, finite, tf.constant(1))
                for expected, current in zip(before, buffer.snapshot(head, count)):
                    np.testing.assert_array_equal(expected, current)

    def test_adapter_serializes_calls_and_failed_output_does_not_commit(self):
        from pilot_history_receiver.benchmark_adapter import PilotHistoryBenchmarkAdapter, CommittedMemory
        adapter = PilotHistoryBenchmarkAdapter.__new__(PilotHistoryBenchmarkAdapter)
        adapter.tf, adapter._lock, adapter._batch_size, adapter._users = tf, RLock(), 1, 2
        adapter._received_shape, adapter._channel_shape = (1, 4, 1, 1), (2, 1, 1, 8)
        adapter.include_logits = False
        adapter.manager = MemoryPolicy(4, 2, 8)
        adapter._buffer = PilotMemoryBuffer(adapter.manager, 1, 1, 9)
        adapter._committed = CommittedMemory(*adapter._buffer.empty_metadata(), 0)
        calls = []
        def execute(y, channel, noise, control, active, head, count):
            ids, step = tf.cast(control[:, :-1], tf.int32), control[0, -1]
            calls.append(int(step))
            observation = tf.concat([tf.reshape(channel, [1, 2, 1, 8]), tf.ones([1, 2, 1, 1])], -1)
            head, count, complete = adapter._buffer.stage(observation, head, count, ids, active, step)
            return tf.zeros([1, 2, 4]), tf.zeros([1, 2, 4], tf.bool), tf.ones([1, 2], tf.bool), head, count, complete
        adapter._execute = execute
        point = dict(ue_ids=np.array([0, 1]), active=np.ones(2), noise_variance=.1,
                     received_grid=np.ones(adapter._received_shape, np.complex64),
                     ls_channel=np.ones(adapter._channel_shape, np.float32))
        with ThreadPoolExecutor(max_workers=2) as workers:
            list(workers.map(adapter.decode_tb, [point, point]))
        self.assertEqual(calls, [0, 1])
        self.assertEqual(adapter._step, 2)
        before = adapter._buffer.read(adapter._committed.head, adapter._committed.count,
            tf.constant([[0, 1]]), tf.ones([1, 2], tf.bool), tf.constant(2))
        changed = dict(point, ls_channel=point['ls_channel']*3.)
        with mock.patch('pilot_history_receiver.benchmark_adapter.decoder_output', side_effect=RuntimeError('output failed')):
            with self.assertRaisesRegex(RuntimeError, 'output failed'):
                adapter.decode_tb(changed)
        self.assertEqual(adapter._step, 2)
        after = adapter._buffer.read(adapter._committed.head, adapter._committed.count,
            tf.constant([[0, 1]]), tf.ones([1, 2], tf.bool), tf.constant(2))
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)
        adapter.decode_tb(changed)
        self.assertEqual(adapter._step, 3)
        adapter.reset_memory([0])
        np.testing.assert_array_equal(adapter._committed.count, [[0, 2, 0, 0]])


if __name__ == "__main__":
    unittest.main()
