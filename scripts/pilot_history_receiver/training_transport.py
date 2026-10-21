from __future__ import annotations

from dataclasses import dataclass
import mmap
import os
from queue import Empty, Full
import shutil
import tempfile
import time

import numpy as np


@dataclass(frozen=True)
class ArrayReference:
    path: str
    offset: int
    shape: tuple
    dtype: np.dtype


@dataclass(frozen=True)
class SharedMessage:
    value: object
    owned_path: str | None


class _ArrayLease:
    def __init__(self, path, owner):
        self.path, self.owner = path, owner
        with open(path, "rb") as stream:
            self.buffer = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        self.address = np.ndarray([len(self.buffer)], np.uint8, buffer=self.buffer).ctypes.data

    def __del__(self):
        if getattr(self, "owner", False):
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
        if hasattr(self, "buffer"):
            try:
                self.buffer.close()
            except BufferError:
                pass


class SharedArray(np.ndarray):
    def __new__(cls, reference, lease):
        value = np.ndarray.__new__(cls, reference.shape, reference.dtype,
                                   buffer=lease.buffer, offset=reference.offset)
        value.lease = lease
        value.flags.writeable = False
        return value

    def __array_finalize__(self, source):
        self.lease = getattr(source, "lease", None)


def open_shared_message(message, *, owner=False):
    leases = {}

    def visit(value):
        if isinstance(value, ArrayReference):
            if value.path not in leases:
                leases[value.path] = _ArrayLease(value.path, owner and value.path == message.owned_path)
            return SharedArray(value, leases[value.path])
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(visit(item) for item in value)
        return value

    return visit(message.value)


class SharedArrayStore:
    def __init__(self):
        self.directory = tempfile.mkdtemp(prefix="neural_rx_arrays_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
        self.owner = os.getpid()

    def pack(self, message, *, borrow=False):
        arrays, size = [], 0

        def collect(value):
            nonlocal size
            if isinstance(value, np.ndarray):
                if value.dtype.hasobject:
                    raise ValueError("Shared training arrays cannot contain Python objects")
                lease = getattr(value, "lease", None)
                offset = value.ctypes.data-lease.address if lease is not None else -1
                if (borrow and lease is not None and value.flags.c_contiguous and
                        0 <= offset <= len(lease.buffer)-value.nbytes):
                    return ArrayReference(lease.path, offset, value.shape, value.dtype)
                offset = (size+63)//64*64
                size = offset+value.nbytes
                arrays.append((value, offset))
                return ArrayReference("", offset, value.shape, value.dtype)
            if isinstance(value, dict):
                return {key: collect(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return type(value)(collect(item) for item in value)
            return value

        value = collect(message)
        if not arrays:
            return SharedMessage(value, None)
        fd, path = tempfile.mkstemp(dir=self.directory)
        try:
            os.posix_fallocate(fd, 0, max(size, 1))
            with mmap.mmap(fd, max(size, 1)) as buffer:
                for array, offset in arrays:
                    np.copyto(np.ndarray(array.shape, array.dtype, buffer=buffer, offset=offset), array)
        except BaseException:
            os.unlink(path)
            raise
        finally:
            os.close(fd)

        def bind(item):
            if isinstance(item, ArrayReference) and not item.path:
                return ArrayReference(path, item.offset, item.shape, item.dtype)
            if isinstance(item, dict):
                return {key: bind(v) for key, v in item.items()}
            if isinstance(item, (tuple, list)):
                return type(item)(bind(v) for v in item)
            return item

        return SharedMessage(bind(value), path)

    @staticmethod
    def release(message):
        if message is not None and message.owned_path is not None:
            try:
                os.unlink(message.owned_path)
            except FileNotFoundError:
                pass

    def close(self):
        if os.getpid() == self.owner:
            shutil.rmtree(self.directory, ignore_errors=True)


def array_bytes(value):
    if isinstance(value, dict):
        return sum(array_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(array_bytes(v) for v in value)
    return int(getattr(value, "nbytes", 0))


class BoundedArrayQueue:
    def __init__(self, context, capacity, byte_capacity):
        if capacity < 1 or byte_capacity < 1:
            raise ValueError("Queue count and byte limits must be positive")
        self.capacity, self.byte_capacity = int(capacity), int(byte_capacity)
        self.queue = context.Queue(maxsize=self.capacity)
        self.slots = context.BoundedSemaphore(self.capacity)
        self.condition = context.Condition()
        self.used = context.Value("q", 0, lock=False)
        self.peak = context.Value("q", 0, lock=False)
        self.admitted = context.Value("q", 0, lock=False)
        self.peak_admitted = context.Value("q", 0, lock=False)
        self.reserved = False
        self.arrays = SharedArrayStore()

    def reserve(self, timeout=None):
        if self.reserved:
            raise RuntimeError("One producer may reserve only one unfinished packet")
        acquired = self.slots.acquire(timeout=timeout)
        with self.condition:
            if acquired:
                self.admitted.value += 1
                self.peak_admitted.value = max(self.peak_admitted.value, self.admitted.value)
        if not acquired:
            raise Full
        self.reserved = True

    def abandon(self):
        if self.reserved:
            self.reserved = False
            with self.condition:
                self.admitted.value -= 1
            self.slots.release()

    def put(self, message, timeout=None):
        started = time.monotonic()
        if not self.reserved:
            self.reserve(timeout)
        size = array_bytes(message)
        if size > self.byte_capacity:
            self.abandon()
            raise ValueError(f"Packet arrays require {size} bytes, exceeding queue budget {self.byte_capacity}")
        deadline = None if timeout is None else started + timeout
        with self.condition:
            while self.used.value + size > self.byte_capacity:
                remaining = None if deadline is None else deadline-time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise Full
                self.condition.wait(remaining)
            self.used.value += size
            self.peak.value = max(self.peak.value, self.used.value)
        shared = None
        try:
            shared = self.arrays.pack(message)
            remaining = None if deadline is None else max(0., deadline-time.monotonic())
            self.queue.put((size, shared), timeout=remaining)
        except BaseException:
            self.arrays.release(shared)
            with self.condition:
                self.used.value -= size
                self.condition.notify_all()
            raise
        self.reserved = False

    def get(self, timeout=None):
        size, message = self.queue.get(timeout=timeout)
        try:
            return open_shared_message(message, owner=True)
        except BaseException:
            self.arrays.release(message)
            raise
        finally:
            with self.condition:
                self.used.value -= size
                self.admitted.value -= 1
                self.condition.notify_all()
            self.slots.release()

    def snapshot(self):
        with self.condition:
            return dict(capacity=self.capacity, byte_capacity=self.byte_capacity,
                array_bytes=self.used.value, peak_array_bytes=self.peak.value,
                admitted=self.admitted.value, peak_admitted=self.peak_admitted.value)

    def close(self):
        self.abandon()
        self.queue.close()

    def cancel_join_thread(self):
        self.queue.cancel_join_thread()
        self.arrays.close()


class OrderedChannelQueues:
    def __init__(self, queues, first, stop, examples):
        self.queues = tuple(queues)
        self.first, self.stop, self.examples = int(first), int(stop), int(examples)
        self.ordinal = 0

    def get(self, timeout=None):
        if self.ordinal >= (self.stop-self.first+1)*self.examples:
            raise Empty
        source = self.queues[self.ordinal % len(self.queues)]
        message = source.get(timeout=timeout)
        update, index = self.first+self.ordinal//self.examples, self.ordinal % self.examples
        if tuple(message[:2]) != (update, index):
            raise ValueError("Channel producer delivered an out-of-order or duplicate example")
        self.ordinal += 1
        return message


def reserve_packet(queue, cancelled):
    while not cancelled.is_set():
        try:
            queue.reserve(timeout=.25)
            return True
        except Full:
            pass
    return False
