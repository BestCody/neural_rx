from __future__ import annotations

import unittest
import gc
import multiprocessing as mp
import os
from pathlib import Path
from queue import Full
import subprocess
import time
from unittest import mock

import numpy as np

from pilot_history_receiver.training_prefetch import EpisodePrefetch, _channel_packet
from pilot_history_receiver.training_transport import BoundedArrayQueue, reserve_packet, SharedArrayStore, open_shared_message
from pilot_history_receiver.training_options import preparation_arguments, resolve_preparation, visible_gpu_indices
from train_memory_receiver import parser as trainer_parser
from run_memory_training import parser as pilot_parser


class PreparationOptionsTests(unittest.TestCase):
    def test_launchers_select_other_gpu_and_preserve_resolved_child_options(self):
        for parser in (trainer_parser, pilot_parser):
            args = parser().parse_args(["--steps", "6000", "--output-dir", "unused", "--gpu", "5"])
            with mock.patch("pilot_history_receiver.training_options.visible_gpu_indices", return_value=[2, 5]), \
                    mock.patch("pilot_history_receiver.training_options.subprocess.run", return_value=mock.Mock(stdout="46068")):
                resolve_preparation(args, 8)
            self.assertEqual((args.data_gpu, args.channel_workers, args.prefetch_examples,
                              args.prefetch_memory_mib), (2, 1, 0, 4096))
            child = trainer_parser().parse_args(["--steps", "6000", "--output-dir", "unused", "--gpu", "5"] +
                                               preparation_arguments(args))
            with mock.patch("pilot_history_receiver.training_options.visible_gpu_indices") as discover, \
                    mock.patch("pilot_history_receiver.training_options.subprocess.run", return_value=mock.Mock(stdout="46068")):
                resolve_preparation(child, 8)
            discover.assert_not_called()
            self.assertEqual(preparation_arguments(child), preparation_arguments(args))

    def test_single_gpu_and_explicit_disable_stay_disabled_in_child(self):
        for extra in ([], ["--data-gpu", "none"]):
            args = pilot_parser().parse_args(["--steps", "6000", "--output-dir", "unused"] + extra)
            with mock.patch("pilot_history_receiver.training_options.visible_gpu_indices", return_value=[0]):
                resolve_preparation(args, 8)
            child = trainer_parser().parse_args(["--steps", "6000", "--output-dir", "unused"] + preparation_arguments(args))
            with mock.patch("pilot_history_receiver.training_options.visible_gpu_indices", return_value=[0, 1]) as discover:
                resolve_preparation(child, 8)
            discover.assert_not_called()
            self.assertIsNone(child.data_gpu)
            self.assertEqual(child.channel_workers, 1)

    def test_small_updates_cap_automatic_workers_but_reject_invalid_override(self):
        for extra in ([], ["--channel-workers", "2"]):
            args = trainer_parser().parse_args(["--steps", "6000", "--output-dir", "unused", "--data-gpu", "1"] + extra)
            if extra:
                with self.assertRaisesRegex(ValueError, "channel slot"):
                    resolve_preparation(args, 1)
            else:
                resolve_preparation(args, 1)
                self.assertEqual(args.channel_workers, 1)

    def test_invalid_resource_assignments_fail_before_training(self):
        for extra, message in ((["--data-gpu", "0"], "separate GPU"),
                               (["--data-gpu", "none", "--channel-workers", "2"], "separate data GPU"),
                               (["--prefetch-memory-mib", "0"], "positive"),
                               (["--prefetch-examples", "-1"], "capacity")):
            with self.subTest(extra=extra):
                args = trainer_parser().parse_args(["--steps", "6000", "--output-dir", "unused"] + extra)
                with self.assertRaisesRegex(ValueError, message):
                    resolve_preparation(args, 8)

    def test_discovery_respects_gpu_visibility_and_uuid_masks(self):
        output = mock.Mock(stdout="2, GPU-aaaa\n5, GPU-bbbb\n")
        for mask, expected in ((None, [2, 5]), ("5,2", [5, 2]), ("GPU-bbbb", [5]), ("", []), ("-1", [])):
            with self.subTest(mask=mask), mock.patch.dict(os.environ, {}, clear=True), \
                    mock.patch("pilot_history_receiver.training_options.subprocess.run", return_value=output):
                if mask is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = mask
                self.assertEqual(visible_gpu_indices(), expected)

    def test_discovery_failure_does_not_require_extra_gpu(self):
        for error in (FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 5)):
            with mock.patch("pilot_history_receiver.training_options.subprocess.run", side_effect=error):
                args = trainer_parser().parse_args(["--steps", "6000", "--output-dir", "unused"])
                resolve_preparation(args, 8)
            self.assertIsNone(args.data_gpu)
            self.assertEqual(args.channel_workers, 1)



def packets_worker(recipe, seed, first, stop, gpu, queue, errors, cancelled):
    for update in range(first, stop+1):
        for example in range(recipe["effective_examples"]):
            queue.put((update, example, {"metadata": {"update": update, "example": example},
                "value": np.array([seed, update, example])}))
    errors.send({"status": "complete"})


def failed_worker(recipe, seed, first, stop, gpu, queue, errors, cancelled):
    errors.send({"status": "failed", "traceback": "deliberate producer failure"})


def wrong_order_worker(recipe, seed, first, stop, gpu, queue, errors, cancelled):
    queue.put((first+1, 0, {}))


def waiting_worker(recipe, seed, first, stop, gpu, queue, errors, cancelled):
    cancelled.wait(10.)


def silent_worker(recipe, seed, first, stop, gpu, queue, errors, cancelled):
    errors.close()


def forward_worker(recipe, seed, first, stop, gpu, queue, errors, cancelled, channel_queue):
    import traceback
    try:
        for update in range(first, stop+1):
            for example in range(recipe["effective_examples"]):
                packet = _channel_packet(channel_queue, update, example, cancelled)
                if packet is None:
                    return
                queue.put((update, example, packet))
        errors.send({"status": "complete"})
    except BaseException:
        errors.send({"status": "failed", "traceback": traceback.format_exc()})
    finally:
        errors.close()


def sharded_worker(recipe, seed, first, stop, gpu, queue, errors, cancelled, shard=(0, 1)):
    import traceback
    try:
        if shard[0] == 0:
            cancelled.wait(.15)
        count = recipe["effective_examples"]
        for ordinal in range(shard[0], (stop-first+1)*count, shard[1]):
            if not reserve_packet(queue, cancelled):
                return
            update, index = first+ordinal//count, ordinal % count
            if recipe.get("fail") and shard[0] == 0:
                raise ValueError("deliberate parallel failure")
            if recipe.get("silent") and shard[0] == 0:
                return
            value = np.zeros(recipe.get("elements", 3), np.int64)
            value[:3] = seed, update, index
            message = (update, index, {"metadata": {"update": update, "example": index}, "value": value})
            if recipe.get("duplicate") and ordinal == shard[1]:
                message = (first, 0, message[2])
            while not cancelled.is_set():
                try:
                    queue.put(message, timeout=.05)
                    break
                except Full:
                    continue
        errors.send({"status": "complete"})
    except BaseException:
        errors.send({"status": "failed", "traceback": traceback.format_exc()})
    finally:
        queue.abandon()
        errors.close()


class BoundedTransportTests(unittest.TestCase):
    def test_shared_views_retain_readonly_storage_and_borrow_without_copying(self):
        store = SharedArrayStore()
        self.addCleanup(store.close)
        source = np.arange(48, dtype=np.float32).reshape(6, 8)
        message = store.pack(dict(value=source))
        packet = open_shared_message(message, owner=True)
        view = np.asarray(packet["value"])[1:4]
        borrowed = store.pack(dict(value=packet["value"][1:4]), borrow=True)
        self.assertIsNone(borrowed.owned_path)
        copied = store.pack(packet["value"].copy(), borrow=True)
        self.assertIsNotNone(copied.owned_path)
        np.testing.assert_array_equal(open_shared_message(copied, owner=True), source)
        remote = open_shared_message(borrowed)
        self.assertFalse(remote["value"].flags.writeable)
        del packet
        gc.collect()
        self.assertTrue(Path(message.owned_path).exists())
        np.testing.assert_array_equal(view, source[1:4])
        np.testing.assert_array_equal(remote["value"], source[1:4])
        del view
        gc.collect()
        self.assertFalse(Path(message.owned_path).exists())
        np.testing.assert_array_equal(remote["value"], source[1:4])

    def test_shared_queue_handoff_and_cancellation_own_their_buffers(self):
        first = BoundedArrayQueue(mp.get_context("spawn"), 1, 128)
        second = BoundedArrayQueue(mp.get_context("spawn"), 1, 128)
        try:
            first.put(np.arange(16))
            value = first.get(timeout=1.)
            first_path = value.lease.path
            second.put(value)
            del value
            gc.collect()
            self.assertFalse(Path(first_path).exists())
            np.testing.assert_array_equal(second.get(timeout=1.), np.arange(16))
            second.put(np.ones(16))
        finally:
            for queue in (first, second):
                queue.close()
                queue.cancel_join_thread()
                self.assertFalse(Path(queue.arrays.directory).exists())

    def test_bytes_block_and_release_separately_from_count(self):
        queue = BoundedArrayQueue(mp.get_context("spawn"), 3, 32)
        try:
            queue.put(np.ones(3, np.int64), timeout=.1)
            with self.assertRaises(Full):
                queue.put(np.zeros(3, np.int64), timeout=.05)
            np.testing.assert_array_equal(queue.get(timeout=1.), np.ones(3))
            queue.put(np.zeros(3, np.int64), timeout=.1)
            queue.get(timeout=1.)
            snapshot = queue.snapshot()
            self.assertEqual(snapshot["array_bytes"], 0)
            self.assertEqual(snapshot["admitted"], 0)
            self.assertLessEqual(snapshot["peak_array_bytes"], 32)
        finally:
            queue.close()
            queue.cancel_join_thread()

    def test_oversized_packet_fails_without_leaking_credit(self):
        queue = BoundedArrayQueue(mp.get_context("spawn"), 1, 8)
        try:
            with self.assertRaisesRegex(ValueError, "exceeding queue budget"):
                queue.put(np.zeros(2, np.int64))
            self.assertEqual(queue.snapshot()["admitted"], 0)
            queue.put(np.zeros(1, np.int64))
            queue.get(timeout=1.)
        finally:
            queue.close()
            queue.cancel_join_thread()

    def source(self, **recipe):
        return EpisodePrefetch(dict(effective_examples=4, **recipe), 6, 2, 3, 1,
            capacity=None, byte_capacity=2048, worker=forward_worker,
            channel_worker=sharded_worker, channel_workers=2, timeout=3.)

    def test_delayed_earliest_result_preserves_order_and_bounds(self):
        with self.source() as source:
            for update in (2, 3):
                packets = source.get(update)
                for index, packet in enumerate(packets):
                    np.testing.assert_array_equal(packet["value"], [6, update, index])
                    self.assertFalse(packet["value"].flags.writeable)
            stats = source.buffer_stats()
            self.assertEqual(sum(row["byte_capacity"] for row in stats.values()), 2048)
            for row in stats.values():
                self.assertLessEqual(row["peak_admitted"], row["capacity"])
                self.assertLessEqual(row["peak_array_bytes"], row["byte_capacity"])
                self.assertEqual(row["admitted"], 0)
                self.assertEqual(row["array_bytes"], 0)
        self.assertEqual(len(source.processes), 3)
        self.assertTrue(all(not p.is_alive() for p in source.processes))

    def test_pool_failure_silent_exit_duplicate_and_oversize_cancel_every_child(self):
        for recipe, message in [({"fail": True}, "parallel failure"),
                                ({"silent": True}, "without completion"),
                                ({"duplicate": True}, "out-of-order|duplicate"),
                                ({"elements": 128}, "exceeding queue budget")]:
            with self.subTest(recipe=recipe), self.source(**recipe) as source:
                with self.assertRaisesRegex((RuntimeError, ValueError), message):
                    source.get(2)
                self.assertTrue(source.closed)
                self.assertTrue(all(not p.is_alive() for p in source.processes))

    def test_cancel_with_full_pool_queues(self):
        source = self.source()
        time.sleep(.5)
        source.close()
        self.assertTrue(all(not p.is_alive() for p in source.processes))

    def test_parallel_requires_explicit_resources(self):
        with self.assertRaisesRegex(TypeError, "byte_capacity"):
            EpisodePrefetch({"effective_examples": 4}, 1, 1, 2, 0, channel_workers=2)
        with self.assertRaisesRegex(ValueError, "one channel slot"):
            EpisodePrefetch({"effective_examples": 4}, 1, 1, 2, 0, capacity=1,
                            channel_workers=2, byte_capacity=1024)


class ProducerTests(unittest.TestCase):
    def source(self, worker, **kwargs):
        return EpisodePrefetch({"effective_examples": 3}, 6, 2, 3, 1,
                               capacity=1, byte_capacity=1024, worker=worker, **kwargs)

    def test_bounded_ordered_immutable_packets_and_resume(self):
        with self.source(packets_worker) as source:
            for update in (2, 3):
                packets = source.get(update)
                for index, packet in enumerate(packets):
                    np.testing.assert_array_equal(packet["value"], [6, update, index])
                    self.assertFalse(packet["value"].flags.writeable)
                with self.assertRaises(ValueError):
                    source.get(update)
            with self.assertRaises(ValueError):
                source.get(4)
        self.assertFalse(source.process.is_alive())
        source.close()

    def test_failure_is_forwarded(self):
        with self.source(failed_worker) as source:
            with self.assertRaisesRegex(RuntimeError, "deliberate producer failure"):
                source.get(2)
        self.assertFalse(source.process.is_alive())

    def test_out_of_order_packets_fail(self):
        with self.source(wrong_order_worker) as source:
            with self.assertRaisesRegex(ValueError, "out-of-order"):
                source.get(2)

    def test_timeout_and_cancel(self):
        with self.source(waiting_worker, timeout=.3) as source:
            with self.assertRaises(TimeoutError):
                source.get(2)
        self.assertFalse(source.process.is_alive())

    def test_stages_preserve_order(self):
        with self.source(forward_worker, channel_worker=packets_worker) as source:
            for update in (2, 3):
                packets = source.get(update)
                for index, packet in enumerate(packets):
                    np.testing.assert_array_equal(packet["value"], [6, update, index])
                    self.assertFalse(packet["value"].flags.writeable)
        self.assertTrue(all(not p.is_alive() for p in source.processes))

    def test_channel_failure_stops_both_workers(self):
        with self.source(forward_worker, channel_worker=failed_worker) as source:
            with self.assertRaisesRegex(RuntimeError, "deliberate producer failure"):
                source.get(2)
        self.assertTrue(all(not p.is_alive() for p in source.processes))

    def test_silent_channel_exit_is_reported(self):
        with self.source(forward_worker, channel_worker=silent_worker) as source:
            with self.assertRaisesRegex(RuntimeError, "without completion"):
                source.get(2)
        self.assertTrue(all(not p.is_alive() for p in source.processes))

    def test_channel_order_violation_is_reported(self):
        with self.source(forward_worker, channel_worker=wrong_order_worker) as source:
            with self.assertRaisesRegex(RuntimeError, "out-of-order|without completion"):
                source.get(2)
        self.assertTrue(all(not p.is_alive() for p in source.processes))

    def test_cancel_stops_both_workers_with_bounded_queues(self):
        source = self.source(forward_worker, channel_worker=packets_worker)
        source.close()
        self.assertTrue(all(not p.is_alive() for p in source.processes))

    def test_staged_timeout_stops_both_workers(self):
        with self.source(forward_worker, channel_worker=waiting_worker, timeout=.3) as source:
            with self.assertRaises(TimeoutError):
                source.get(2)
        self.assertTrue(all(not p.is_alive() for p in source.processes))


if __name__ == "__main__":
    unittest.main()
