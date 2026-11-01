from __future__ import annotations

import multiprocessing as mp
from threading import get_ident
import time
import traceback

import numpy as np

from .training_transport import SharedArrayStore, open_shared_message


def partition_episodes(packets):
    groups, loads = [[], []], [0, 0]
    work = [int(p["active"].shape[0]*p["active"].shape[1]) for p in packets]
    for index in sorted(range(len(packets)), key=lambda i: (-work[i], i)):
        device = 0 if loads[0] <= loads[1] else 1
        groups[device].append(index)
        loads[device] += work[index]
    return tuple(sorted(group) for group in groups)


def _worker(connection, recipe, seed, gpu, batch_tbs, descriptor_tbs, memory_mib, tf32):
    try:
        from .training_options import configure_gpu
        tf = configure_gpu(gpu, memory_mib)
        tf.config.experimental.enable_tensor_float_32_execution(tf32)
        from .training import TrainingModel
        model = TrainingModel(recipe, seed, batch_tbs, descriptor_tbs)
        variables = model.optimization_variables
        sizes = [int(tf.size(v)) for v in variables]

        @tf.function(input_signature=[tf.TensorSpec([sum(sizes)], tf.float32)])
        def assign(flat):
            for variable, value in zip(variables, tf.split(flat, sizes)):
                variable.assign(tf.reshape(value, variable.shape))

        @tf.function(reduce_retracing=True)
        def flatten(values):
            return tf.concat([tf.reshape(value, [-1]) for value in values], 0)

        connection.send(dict(kind="ready", shapes=[tuple(v.shape) for v in variables]))
        while True:
            request = connection.recv()
            if request["kind"] == "stop":
                break
            if request["kind"] != "gradients":
                raise ValueError("Unknown learner request")
            assign(tf.convert_to_tensor(request["weights"], tf.float32))
            packets = open_shared_message(request["packets"])
            gradients, metrics, fingerprints, den = model.execution.run(
                request["width"], packets, denominator=request["denominator"])
            connection.send(dict(kind="gradients", update=request["update"], indices=request["indices"],
                gradients=flatten(gradients).numpy(), metrics=metrics.numpy(),
                fingerprints=fingerprints, denominator=den))
            del packets
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(dict(kind="failed", traceback=traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        connection.close()


class GradientReplica:
    def __init__(self, trainer, gpu, memory_mib=None, timeout=600.):
        self.owner, self.pending, self.closed = get_ident(), None, False
        self.arrays, self.shared_request, self.packet_owners = SharedArrayStore(), None, None
        self.timeout = float(timeout)
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        settings = trainer.execution.settings()
        self.process = context.Process(target=_worker, args=(child, trainer.recipe, trainer.seed, gpu,
            settings["detector_batch_tbs"], settings["descriptor_chunk_tbs"], memory_mib,
            settings["tf32_enabled"]), name="channel-memory-gradient-replica")
        try:
            self.process.start()
            child.close()
            result = self._receive()
            if result["kind"] != "ready" or result["shapes"] != [tuple(v.shape) for v in trainer.optimization_variables]:
                raise ValueError("Learner replica has a different parameter layout")
        except BaseException:
            self.close()
            raise
        finally:
            child.close()

    def _receive(self):
        deadline = time.monotonic()+self.timeout
        while not self.connection.poll(.25):
            if not self.process.is_alive():
                raise RuntimeError(f"Learner replica exited with code {self.process.exitcode}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Learner replica did not complete its request")
        try:
            value = self.connection.recv()
        except EOFError as exc:
            raise RuntimeError("Learner replica closed without a result") from exc
        if value["kind"] == "failed":
            raise RuntimeError("Learner replica failed:\n"+value["traceback"])
        return value

    def submit(self, update, width, packets, indices, weights, denominator):
        if self.closed or self.pending is not None or get_ident() != self.owner:
            raise RuntimeError("A replica accepts one update from its owner at a time")
        host = [{k: np.asarray(v) if hasattr(v, "numpy") else v for k, v in p.items()} for p in packets]
        self.shared_request = self.arrays.pack(host, borrow=True)
        self.packet_owners = host
        self.pending = (int(update), list(indices))
        self.connection.send(dict(kind="gradients", update=int(update), width=width,
            packets=self.shared_request, indices=list(indices), weights=weights, denominator=float(denominator)))

    def result(self):
        if self.pending is None or self.closed or get_ident() != self.owner:
            raise RuntimeError("No pending replica update for this owner")
        value = self._receive()
        if value["kind"] != "gradients" or (value["update"], value["indices"]) != self.pending:
            raise ValueError("Learner returned a stale, duplicate, or incorrectly assigned update")
        self.pending = None
        self.arrays.release(self.shared_request)
        self.shared_request, self.packet_owners = None, None
        return value

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.pid is None:
            self.connection.close()
            self.arrays.close()
            return
        if self.process.is_alive() and self.pending is None:
            try:
                self.connection.send(dict(kind="stop"))
            except (BrokenPipeError, EOFError, OSError):
                pass
        self.process.join(timeout=2.)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5.)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=5.)
        self.connection.close()
        self.arrays.release(self.shared_request)
        self.shared_request, self.packet_owners = None, None
        self.arrays.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
