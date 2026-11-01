from __future__ import annotations

import multiprocessing as mp
from queue import Empty, Full
from threading import get_ident
import time
import traceback

from .training_transport import BoundedArrayQueue, OrderedChannelQueues, reserve_packet
from .training_options import DEFAULT_RENDER_BATCH_TBS


def _configure(gpu, seed, memory_mib=None):
    from .training_options import configure_gpu
    tf = configure_gpu(gpu, memory_mib)
    tf.keras.utils.set_random_seed(seed)
    return tf


def _host(packet, tf):
    import numpy as np
    return {k: np.asarray(v) if tf.is_tensor(v) else v for k, v in packet.items()}


def _send(queue, message, cancelled):
    while not cancelled.is_set():
        try:
            queue.put(message, timeout=.25)
            return True
        except Full:
            continue
    return False


def _channel_packet(queue, update, index, cancelled, timeout=600.):
    deadline = time.monotonic()+timeout
    while not cancelled.is_set():
        try:
            actual, example, packet = queue.get(timeout=.25)
            if (actual, example) != (update, index):
                raise ValueError("Channel producer delivered an out-of-order example")
            metadata = packet["metadata"]
            if (metadata["update"], metadata["example"]) != (update, index):
                raise ValueError("Channel metadata differs from its queue identity")
            return packet
        except Empty:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Channel producer timed out at update {update}, example {index}")
    return None


def _produce_channels(recipe, seed, first, stop, gpu, queue, errors, cancelled,
                      shard=(0, 1), memory_mib=None):
    try:
        tf = _configure(gpu, seed, memory_mib)
        from utils.parameters import Parameters
        from .training_data import TrainingEpisodes, update_width
        contexts = {w: (Parameters(recipe["radio_config"], training=True, n_size_bwp_override=w), None)
                    for w in sorted(set(recipe["prbs"]))}
        episodes = TrainingEpisodes(contexts, recipe, seed)
        examples, previous_update = recipe["effective_examples"], None
        for ordinal in range(shard[0], (stop-first+1)*examples, shard[1]):
            update, index = first+ordinal//examples, ordinal % examples
            if not reserve_packet(queue, cancelled):
                return
            if update != previous_update:
                width = update_width(recipe, seed, update)
                previous_update = update
            plan = _host(episodes.plan(update, index, width), tf)
            if not _send(queue, (update, index, plan), cancelled):
                return
            del plan
        errors.send({"status": "complete"})
    except BaseException:
        errors.send({"status": "failed", "traceback": traceback.format_exc()})
        raise
    finally:
        queue.abandon()
        errors.close()


def _produce(recipe, seed, first, stop, gpu, queue, errors, cancelled, channel_queue, memory_mib=None,
             render_batch_tbs=DEFAULT_RENDER_BATCH_TBS):
    try:
        tf = _configure(gpu, seed, memory_mib)
        from temporal_receiver_core import build_backbone
        from .training_data import TrainingEpisodes
        contexts = {w: build_backbone(recipe["radio_config"], 2, True, n_size_bwp_override=w)
                    for w in sorted(set(recipe["prbs"]))}
        episodes = TrainingEpisodes(contexts, recipe, seed, render_batch_tbs)
        for update in range(first, stop+1):
            for index in range(recipe["effective_examples"]):
                if not reserve_packet(queue, cancelled):
                    return
                plan = _channel_packet(channel_queue, update, index, cancelled)
                if plan is None:
                    return
                packet = episodes.render(plan)
                del plan
                packet = _host(packet, tf)
                if not _send(queue, (update, index, packet), cancelled):
                    return
                del packet
        errors.send({"status": "complete"})
    except BaseException:
        errors.send({"status": "failed", "traceback": traceback.format_exc()})
        raise
    finally:
        queue.abandon()
        errors.close()


class EpisodePrefetch:
    def __init__(self, recipe, seed, first, stop, gpu, *, byte_capacity, capacity=2, timeout=600., worker=_produce,
                 channel_worker=None, channel_workers=1, gpu_memory_mib=None,
                 render_batch_tbs=DEFAULT_RENDER_BATCH_TBS):
        capacity = recipe["effective_examples"] if capacity is None else int(capacity)
        if capacity < 1 or timeout <= 0 or first < 1 or stop < first or channel_workers < 1:
            raise ValueError("Invalid producer bounds")
        if channel_workers > capacity:
            raise ValueError("Reserve at least one channel slot per worker")
        if byte_capacity < 2*channel_workers:
            raise ValueError("Byte budget cannot provide space for every stage")
        if gpu_memory_mib and gpu_memory_mib-max(1, gpu_memory_mib//5) < channel_workers:
            raise ValueError("GPU memory budget must provide a positive limit for every producer")
        self.recipe, self.seed = dict(recipe), int(seed)
        self.next_update, self.stop = int(first), int(stop)
        self.timeout = float(timeout)
        self.owner = get_ident()
        self.closed = False
        self.stage_status = {}
        context = mp.get_context("spawn")
        self.queue = BoundedArrayQueue(context, capacity, byte_capacity//2)
        render_errors, child_errors = context.Pipe(duplex=False)
        self.cancelled = context.Event()
        self.channel_queues = []
        self.stage_handles = []
        self.processes = []
        kwargs = {}
        if worker is _produce:
            kwargs["render_batch_tbs"] = render_batch_tbs
        if gpu_memory_mib:
            kwargs["memory_mib"] = max(1, gpu_memory_mib//5)
        child_pipes = [child_errors]
        pending = []
        if worker is _produce or channel_worker is not None:
            for index in range(channel_workers):
                count = capacity//channel_workers + (index < capacity % channel_workers)
                budget = (byte_capacity-byte_capacity//2)//channel_workers
                channel_queue = BoundedArrayQueue(context, count, budget)
                parent_errors, channel_errors = context.Pipe(duplex=False)
                child_pipes.append(channel_errors)
                channel_kwargs = {} if channel_workers == 1 else {"shard": (index, channel_workers)}
                if gpu_memory_mib:
                    channel_kwargs["memory_mib"] = (gpu_memory_mib-kwargs["memory_mib"])//channel_workers
                process = context.Process(target=channel_worker or _produce_channels,
                    args=(self.recipe, self.seed, first, stop, gpu, channel_queue,
                          channel_errors, self.cancelled), kwargs=channel_kwargs,
                    name=f"memory-training-channels-{index}")
                self.channel_queues.append(channel_queue)
                self.stage_handles.append((f"channel_{index}", parent_errors, process))
                pending.append(process)
            kwargs["channel_queue"] = (self.channel_queues[0] if channel_workers == 1 else
                OrderedChannelQueues(self.channel_queues, first, stop, recipe["effective_examples"]))
        self.process = context.Process(target=worker, args=(self.recipe, self.seed, first, stop,
            gpu, self.queue, child_errors, self.cancelled), kwargs=kwargs, name="memory-training-data")
        self.stage_handles.append(("render", render_errors, self.process))
        pending.append(self.process)
        try:
            for process in pending:
                process.start()
                self.processes.append(process)
        except BaseException:
            self.close()
            raise
        finally:
            for pipe in child_pipes:
                pipe.close()

    def _status(self):
        for name, pipe, process in self.stage_handles:
            while pipe.poll():
                try:
                    self.stage_status[name] = pipe.recv()
                except EOFError:
                    break
            status = self.stage_status.get(name)
            if status and status["status"] == "failed":
                raise RuntimeError(f"Training data producer failed ({name}):\n" + status["traceback"])
            if process.exitcode is not None and process.exitcode != 0:
                raise RuntimeError(f"Training data producer {name} exited with code {process.exitcode}")
            if name.startswith("channel") and process.exitcode == 0 and not status:
                raise RuntimeError("Training channel producer exited without completion")

    def get(self, update):
        if self.closed or get_ident() != self.owner:
            raise RuntimeError("The producer must have one live consumer")
        if update != self.next_update or update > self.stop:
            raise ValueError("Updates must be consumed exactly once in order")
        try:
            return self._get(update)
        except BaseException:
            self.close()
            raise

    def _get(self, update):
        import numpy as np
        packets = []
        for index in range(self.recipe["effective_examples"]):
            deadline = time.monotonic()+self.timeout
            while True:
                self._status()
                try:
                    actual, example, packet = self.queue.get(timeout=.25)
                    self._status()
                    break
                except Empty:
                    self._status()
                    if not self.process.is_alive():
                        raise RuntimeError(f"Training data producer exited before update {update}, example {index}")
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Training data producer timed out at update {update}, example {index}")
            if (actual, example) != (update, index):
                raise ValueError("Training data producer delivered an out-of-order example")
            metadata = packet["metadata"]
            if (metadata["update"], metadata["example"]) != (update, index):
                raise ValueError("Training data metadata differs from its queue identity")
            for value in packet.values():
                if isinstance(value, np.ndarray):
                    value.flags.writeable = False
            packets.append(packet)
        self.next_update += 1
        return packets

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.cancelled.set()
        self.queue.close()
        for queue in self.channel_queues:
            queue.close()
        for process in self.processes:
            process.join(timeout=2.)
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=5.)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.)
        for _, pipe, _ in self.stage_handles:
            pipe.close()
        for queue in self.channel_queues:
            queue.cancel_join_thread()
        self.queue.cancel_join_thread()

    def buffer_stats(self):
        queues = [("render", self.queue)] + [(f"channel_{i}", q) for i, q in enumerate(self.channel_queues)]
        return {name: queue.snapshot() for name, queue in queues}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
