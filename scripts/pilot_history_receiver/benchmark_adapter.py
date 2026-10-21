from __future__ import annotations

import json
from threading import RLock
from typing import NamedTuple

import numpy as np

from temporal_benchmark_adapters import (
    _ArtifactGuard, _checkpoint_path, _runtime_config, _support_path, _verify_method,
)
from temporal_benchmark_candidate import checked_decoder_tensors, decoder_output, stack_receiver_grids
from .pilot_history import MemoryPolicy, PilotMemoryBuffer, validate_memory_metadata


class CommittedMemory(NamedTuple):
    head: object
    count: object
    step: int


class PilotHistoryBenchmarkAdapter(_ArtifactGuard):
    def __init__(self, *, manifest, method, training_seed, artifacts,
                 config_name, tf_module):
        _verify_method(manifest, method)
        if method not in ("candidate_persistent", "candidate_reset"):
            raise ValueError("Channel memory requires a candidate method")
        if int(manifest["inference"]["iterations"]) != 2:
            raise ValueError("Channel memory requires K=2")
        super().__init__(manifest, artifacts)
        from receiver_decoding import OneLayerGridCodeMap, DecoderAssistedTB
        from temporal_receiver_core import build_backbone, model_checksum, prepare_cgnn_inputs
        from .pilot_layout import PilotLayout
        from .model import PilotHistoryReceiver, architecture_settings

        config_name = _runtime_config(manifest, artifacts, config_name)
        settings = json.loads(_support_path(
            manifest, artifacts, "cnn_autoencoder_finalist.json"
        ).read_text())
        dimensions = architecture_settings(settings)
        self.tf = tf_module
        self.manifest = manifest
        self.method = method
        self._lock = RLock()
        _, self.e2e = build_backbone(
            config_name, 2, False,
            num_tx_eval=int(manifest["radio"]["maximum_active_ues"]),
            n_size_bwp_override=int(manifest["radio"]["n_size_bwp"]),
        )
        self.receiver = self.e2e._receiver
        iterations = int(settings['architecture']['ldpc_iterations'])
        grid = self.e2e._transmitters[0].resource_grid
        self.layout = PilotLayout(self.receiver._ls_est, grid, dimensions["neighbor_radius"])
        self.model = PilotHistoryReceiver(self.receiver._neural_rx._cgnn, self.layout,
                                        self.e2e._transmitters[0]._mapper,
                                        **dimensions, name="channel_memory_k2")
        users = int(manifest["radio"]["maximum_active_ues"])
        antennas = int(manifest["radio"]["receive_antennas"])
        y = self.tf.zeros([1, 1, antennas, grid.num_ofdm_symbols, grid.fft_size], self.tf.complex64)
        channel = self.tf.zeros([1, users, grid.fft_size, grid.num_ofdm_symbols, 2 * antennas])
        self._received_shape = tuple(y.shape[1:])
        self._channel_shape = tuple(channel.shape[1:])
        self._users = users
        self.model(prepare_cgnn_inputs(self.receiver, y, channel, self.tf.ones([1, users])), self.tf.ones([1]),
                   training=False)
        self.grid_map = OneLayerGridCodeMap(
            self.receiver._neural_rx, users, int(manifest["radio"]["bits_per_symbol"]),
        )
        self.checkpoint = _checkpoint_path(manifest, training_seed, artifacts)
        if not self.checkpoint.name.endswith(".weights.h5"):
            raise ValueError("Channel memory requires a complete .weights.h5 checkpoint")
        self.model.load_weights(str(self.checkpoint))
        self.model.trainable = False
        policy = settings["benchmark_memory"]
        self.manager = MemoryPolicy(
            capacity=int(policy["capacity"]), history_tbs=self.model.history_tbs,
            expiry_slots=int(policy["expiry_tbs"]),
        )
        self.decoder = DecoderAssistedTB(
            self.receiver._tb_encoders[0], self.receiver._tb_decoders[0], iterations,
            int(settings['architecture']['ldpc_first_iterations']),
        )
        self._model_checksum = model_checksum
        self._model_hash = model_checksum(self.model)
        self._prepare_inputs = prepare_cgnn_inputs
        self._buffer = None
        self._memory_device = self.model.weights[0].device
        self.start_episode()

    @property
    def _state(self):
        with self._lock:
            return self._buffer.snapshot(self._committed.head, self._committed.count)

    @property
    def _step(self):
        return self._committed.step

    def _build_execution(self):
        tf = self.tf
        buffer = self._buffer
        reset = self.method == "candidate_reset"

        @tf.function(reduce_retracing=True, jit_compile=True)
        def prepare(channel, noise, ids, active, step, head, count, tables):
            history = buffer.gather(tables, head, count, ids, active, step)
            observation = self.model.observe(channel, noise)
            preparation = self.model.prepare([None, None, channel, tf.cast(active, tf.float32)],
                                             observation, history, self.layout)
            pending, finite = buffer.prepare_stage(observation, head, count, ids, active)
            return observation, preparation, pending, finite

        @tf.function(reduce_retracing=True, jit_compile=False)
        def execute(y, channel, noise, control, active, head, count):
            ids, step = tf.cast(control[:, :-1], tf.int32), control[0, -1]
            if reset:
                count = tf.zeros_like(count)
            tables = tuple(v.read_value() for v in (buffer.observations, buffer.timestamps, buffer.ports))
            observation, preparation, pending, finite = prepare(
                channel, noise, ids, active, step, head, count, tables)
            inputs = self._prepare_inputs(self.receiver, y, channel, tf.cast(active, tf.float32))
            output, _, _, _, observation, _ = self.model.decode(inputs, noise, ids, None, self.decoder,
                                                            self.grid_map, observation, preparation)
            logits, hard, crc = checked_decoder_tensors(output)
            with tf.control_dependencies([logits, hard, crc]):
                next_head, next_count, complete = buffer.commit(pending, finite, step)
            with tf.control_dependencies([complete]):
                return tf.identity(logits), tf.identity(hard), tf.identity(crc), next_head, next_count, complete

        return execute

    def start_episode(self):
        self.start_episode_batch(1)

    def start_episode_batch(self, batch_size):
        if int(batch_size) < 1:
            raise ValueError("Episode batch must be positive")
        with self._lock:
            recreate = self._buffer is None or int(batch_size) != self._batch_size
            self._batch_size = int(batch_size)
            if recreate:
                with self.tf.device(self._memory_device):
                    self._buffer = PilotMemoryBuffer(self.manager, self._batch_size,
                        self.layout.count, self._channel_shape[-1]+1)
                self._execute = self._build_execution()
            head, count = self._buffer.empty_metadata()
            self._committed = CommittedMemory(head, count, 0)

    def reset_memory(self, ue_ids=None):
        with self._lock:
            if ue_ids is None:
                head, count = self._buffer.empty_metadata()
            else:
                ids = np.asarray(ue_ids)
                if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or np.any(ids >= self.manager.capacity):
                    raise ValueError("Reset requires valid physical UE IDs")
                head = self._committed.head
                keep = ~np.isin(np.arange(self.manager.capacity), ids)
                count = self.tf.where(keep[None], self._committed.count, 0)
            self._committed = CommittedMemory(head, count, self._step)

    def state_is_empty(self):
        with self._lock:
            return not bool(self.tf.reduce_any(self._committed.count > 0).numpy())

    def decode_tb(self, inputs):
        output = self.decode_tb_batch([inputs])
        return {name: value[0] for name, value in output.items()}

    def decode_tb_batch(self, inputs):
        with self._lock:
            if len(inputs) != self._batch_size:
                raise ValueError("Input and memory batch sizes differ")
            ids = np.stack([p["ue_ids"] for p in inputs])
            active = np.stack([p["active"] for p in inputs])
            if ids.shape != (self._batch_size,self._users):
                raise ValueError("Scheduled user count differs from the receiver configuration")
            validate_memory_metadata(ids,active,self.manager.capacity)
            for point in inputs:
                if np.shape(point["received_grid"]) != self._received_shape or np.shape(point["ls_channel"]) != self._channel_shape:
                    raise ValueError("Received or channel grid differs from the receiver configuration")
            noise = np.asarray([p["noise_variance"] for p in inputs],np.float32)
            if noise.shape != (self._batch_size,) or not np.all(np.isfinite(noise)) or np.any(noise < 0):
                raise ValueError("Noise variance must be finite, nonnegative, and scalar per sample")
            times = [p.get("time_index", self._step) for p in inputs]
            if any(isinstance(t,(bool,np.bool_)) or not isinstance(t,(int,np.integer)) or not self._step <= t < np.iinfo(np.int32).max for t in times) or len(set(times))!=1:
                raise ValueError("Memory calls require matching monotonic TB indices; reset for a new timeline")
            step = int(times[0])
            control = np.concatenate([ids.astype(np.int64, copy=False),
                                      np.full((self._batch_size, 1), step, np.int64)], axis=1)
            tf = self.tf
            complete = None
            try:
                logits, hard, crc, next_head, next_count, complete = self._execute(
                    *stack_receiver_grids(inputs, tf), tf.convert_to_tensor(noise,tf.float32),
                    tf.convert_to_tensor(control, tf.int64), tf.convert_to_tensor(active, tf.bool),
                    self._committed.head, self._committed.count,
                )
                output = decoder_output(logits, hard, crc, include_logits=self.include_logits)
            except BaseException:
                if complete is not None:
                    try:
                        complete.numpy()
                    except BaseException:
                        pass
                raise
            self._committed = CommittedMemory(
                next_head, next_count, step+1)
            return output

    def checksums_unchanged(self):
        with self._lock:
            return self._artifacts_unchanged() and self._model_hash == self._model_checksum(self.model)
