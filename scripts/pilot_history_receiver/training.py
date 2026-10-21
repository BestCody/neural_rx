from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from threading import Lock
from pathlib import Path

import numpy as np
import tensorflow as tf

from temporal_receiver_core import build_backbone, prepare_cgnn_inputs
from temporal_benchmark_scoring import block_error_rate
from artifact_io import atomic_json, file_sha256, sync_directory, validate_checkpoint_files
from artifact_provenance import normalize_training_recipe, training_source_records, validate_training_binding
from .pilot_history import MemoryPolicy
from .pilot_layout import PilotLayout
from .model import PilotHistoryReceiver, architecture_settings
from .training_data import TrainingEpisodes, update_width, validate_history_settings
from .training_execution import BatchedTrainingExecution
from .training_options import DEFAULT_DETECTOR_BATCH_TBS, DEFAULT_DESCRIPTOR_CHUNK_TBS


ROOT = Path(__file__).resolve().parents[2]


def recipe_hash(recipe):
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def validate_recipe(recipe):
    normalize_training_recipe(recipe)
    validate_history_settings(recipe)
    for names, probs in (("prbs", "prb_probabilities"), ("episode_tbs", "episode_probabilities"),
                         ("channel_families", "channel_probabilities")):
        values = np.asarray(recipe[probs], np.float64)
        if (not recipe[names] or len(recipe[names]) != len(values) or
                not np.all(np.isfinite(values)) or np.any(values < 0) or not np.isclose(values.sum(), 1)):
            raise ValueError(f"Invalid sampling distribution: {names}")
    for key in ("learning_rate", "minimum_learning_rate", "gradient_clip_norm",
                "coded_loss_weight", "payload_loss_weight", "channel_loss_weight", "reset_weight"):
        if not np.isfinite(recipe[key]) or recipe[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if recipe["minimum_learning_rate"] > recipe["learning_rate"] or not 0 < recipe["plateau_factor"] < 1:
        raise ValueError("Invalid plateau learning-rate settings")
    for key in ("effective_examples", "checkpoint_every", "learning_rate_warmup",
                "development_interval", "plateau_patience", "memory_capacity", "expiry_tbs"):
        if type(recipe[key]) is not int or recipe[key] < 1:
            raise ValueError(f"Invalid {key}")
    if any(type(width) is not int or width < 1 for width in recipe["prbs"]):
        raise ValueError("PRB counts must be positive integers")
    policy = json.loads((ROOT/"config/memory_receiver.json").read_text())["benchmark_memory"]
    if recipe["memory_capacity"] != policy["capacity"] or recipe["expiry_tbs"] != policy["expiry_tbs"]:
        raise ValueError("Memory lifecycle differs from the adapter contract")


def per_user_bce(labels, logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(labels, tf.float32), logits=logits), -1)


def per_user_payload_loss(labels, logits):
    negative_margin = -(2*tf.cast(labels, tf.float32)-1)*logits
    block_loss = tf.reduce_logsumexp(tf.concat([tf.zeros_like(negative_margin[..., :1]), negative_margin], -1), -1)
    block_loss /= tf.math.log(tf.cast(tf.shape(logits)[-1]+1, tf.float32))
    return .5*tf.reduce_mean(tf.nn.softplus(negative_margin), -1)+.5*block_loss


class TrainingModel:
    def __init__(self, recipe, seed=None, detector_batch_tbs=DEFAULT_DETECTOR_BATCH_TBS,
                 descriptor_chunk_tbs=DEFAULT_DESCRIPTOR_CHUNK_TBS):
        validate_recipe(recipe)
        self.recipe = normalize_training_recipe(recipe)
        self.seed = int(recipe["seed"] if seed is None else seed)
        self.contexts = {}
        tf.keras.utils.set_random_seed(self.seed)
        for width in sorted(set(recipe["prbs"])):
            self.contexts[width] = build_backbone(recipe["radio_config"], 2, True, n_size_bwp_override=width)
        first = min(self.contexts)
        base = self.contexts[first][1]._receiver._neural_rx._cgnn
        self.settings = json.loads((ROOT/"config/memory_receiver.json").read_text())
        dimensions = architecture_settings(self.settings)
        layouts = {width: PilotLayout(e2e._receiver._ls_est, parameters.transmitters[0].resource_grid,
                                     dimensions["neighbor_radius"])
                   for width, (parameters, e2e) in self.contexts.items()}
        self.model = PilotHistoryReceiver(base, layouts[first], self.contexts[first][0].transmitters[0]._mapper,
                                        **dimensions, name="channel_memory_k2")
        for layout in layouts.values():
            self.model.register_layout(layout)
        tf.keras.utils.set_random_seed(self.seed+1)
        receiver = self.contexts[first][1]._receiver
        users, antennas = self.contexts[first][0].max_num_tx, self.contexts[first][0].num_rx_antennas
        if recipe["memory_capacity"] < users:
            raise ValueError("Memory capacity must accommodate every active UE")
        grid = self.contexts[first][0].transmitters[0]._resource_grid
        y = tf.ones([1, 1, antennas, grid.num_ofdm_symbols, grid.fft_size], tf.complex64)
        ls = tf.ones([1, users, grid.fft_size, grid.num_ofdm_symbols, 2*antennas])
        self.model(prepare_cgnn_inputs(receiver, y, ls, tf.ones([1, users])), tf.ones([1]))
        initializer = ROOT/recipe["initializer_checkpoint"]
        if file_sha256(initializer) != recipe["initializer_sha256"]:
            raise ValueError("Receiver initialization checksum differs")
        self.model.load_weights(str(initializer))
        if file_sha256(initializer) != recipe["initializer_sha256"]:
            raise ValueError("Receiver initialization changed while loading")
        self.memory_variables, self.backbone_variables = self.model.optimization_variables()
        self.optimization_variables = self.memory_variables + self.backbone_variables
        self.manager = MemoryPolicy(recipe["memory_capacity"], self.model.history_tbs, recipe["expiry_tbs"])
        self.episodes = TrainingEpisodes(self.contexts, recipe, self.seed)
        self.execution = BatchedTrainingExecution(self, detector_batch_tbs, descriptor_chunk_tbs)


class MemoryTrainer(TrainingModel):
    def __init__(self, recipe, seed=None, detector_batch_tbs=DEFAULT_DETECTOR_BATCH_TBS,
                 descriptor_chunk_tbs=DEFAULT_DESCRIPTOR_CHUNK_TBS):
        super().__init__(recipe, seed, detector_batch_tbs, descriptor_chunk_tbs)
        self.optimizer = tf.keras.optimizers.Adam(recipe["learning_rate"], jit_compile=False)
        self.optimizer.build(self.optimization_variables)
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.learning_rate = tf.Variable(recipe["learning_rate"], trainable=False, dtype=tf.float32)
        self.best_tbler = tf.Variable(np.inf, trainable=False, dtype=tf.float64)
        self.best_update = tf.Variable(-1, trainable=False, dtype=tf.int64)
        self.plateau_count = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.development_update = tf.Variable(-1, trainable=False, dtype=tf.int64)
        self.development_bank = tf.Variable("", trainable=False, dtype=tf.string)
        self.development_result = tf.Variable("", trainable=False, dtype=tf.string)
        self.checkpoint = tf.train.Checkpoint(
            model=self.model, optimizer=self.optimizer, step=self.step,
            learning_rate=self.learning_rate, best_tbler=self.best_tbler, best_update=self.best_update,
            plateau_count=self.plateau_count, development_update=self.development_update,
            development_bank=self.development_bank, development_result=self.development_result)
        self._update_lock = Lock()
        self.replica = None

    @tf.function(reduce_retracing=True)
    def apply(self, gradients, learning_rate):
        clipped, norm = tf.clip_by_global_norm(gradients, self.recipe["gradient_clip_norm"])
        self.optimizer.learning_rate.assign(learning_rate)
        self.optimizer.apply_gradients(zip(clipped, self.optimization_variables))
        self.step.assign_add(1)
        return norm

    def binding(self):
        return {"recipe": self.recipe, "recipe_sha256": recipe_hash(self.recipe),
                "architecture": self.settings["architecture"],
                "seed": self.seed, "execution": self.execution.settings(),
                "sources": training_source_records(self.recipe["radio_config"])}

    def schedule_state(self):
        best = float(self.best_tbler.numpy())
        return dict(learning_rate=float(self.learning_rate.numpy()),
                    best_tbler=best if np.isfinite(best) else None,
                    best_update=int(self.best_update.numpy()), plateau_count=int(self.plateau_count.numpy()),
                    development_update=int(self.development_update.numpy()),
                    development_bank=self.development_bank.numpy().decode(),
                    development_result=self.development_result.numpy().decode())

    def observe_development(self, result):
        if result["phase"] != "development":
            raise ValueError("Only development TBLER may control the learning rate")
        checkpoint = Path(result["checkpoint"])
        manifest = json.loads((checkpoint/"manifest.json").read_text())
        validate_training_binding(manifest["binding"], self.binding())
        update = int(manifest["update"])
        state = self.schedule_state()
        if update == state["development_update"]:
            if result["result_sha256"] != state["development_result"]:
                raise ValueError("An already consumed development result changed")
            return state
        if update <= state["development_update"] or update != int(self.step.numpy()):
            raise ValueError("Development feedback must describe the current checkpoint")
        if state["development_bank"] and state["development_bank"] != result["bank_sha256"]:
            raise ValueError("Development data or scoring changed; use a new selection and schedule record")
        counts = result["summary"]["candidate_persistent"]
        score = float(block_error_rate(counts["core_errors"], counts["core_blocks"]))
        if score != counts["core_tbler"]:
            raise ValueError("Development score differs from its pooled block counts")
        if score < float(self.best_tbler.numpy()):
            self.best_tbler.assign(score)
            self.best_update.assign(update)
            self.plateau_count.assign(0)
        else:
            self.plateau_count.assign_add(1)
            if int(self.plateau_count.numpy()) >= self.recipe["plateau_patience"]:
                self.learning_rate.assign(max(self.recipe["minimum_learning_rate"],
                    float(self.learning_rate.numpy()) * self.recipe["plateau_factor"]))
                self.plateau_count.assign(0)
        self.development_update.assign(update)
        self.development_bank.assign(result["bank_sha256"])
        self.development_result.assign(result["result_sha256"])
        return self.schedule_state()

    def update(self, packets=None, apply=True):
        if not self._update_lock.acquire(blocking=False):
            raise RuntimeError("Training updates must have a single owner")
        try:
            return self._update(packets, apply)
        finally:
            self._update_lock.release()

    def _update(self, packets, apply):
        update = int(self.step.numpy())+1
        learning_rate = float(self.learning_rate.numpy()) * min(1., update/self.recipe["learning_rate_warmup"])
        width = update_width(self.recipe, self.seed, update)
        if packets is None:
            packets = [self.episodes.sample(update, index, width) for index in range(self.recipe["effective_examples"])]
        if len(packets) != self.recipe["effective_examples"]:
            raise ValueError("An update must retain the configured number of base examples")
        if self.replica is None or len(packets) < 2:
            accumulated, metrics, fingerprints, den = self.execution.run(width, packets)
        else:
            from .training_parallel import partition_episodes
            local, remote = partition_episodes(packets)
            den = sum(float(np.asarray(p["active"])[0].sum()) for p in packets)
            weights = self.flat_weights().numpy()
            self.replica.submit(update, width, [packets[i] for i in remote], remote, weights, den)
            accumulated, metrics, local_fingerprints, local_den = self.execution.run(
                width, [packets[i] for i in local], denominator=den)
            result = self.replica.result()
            if local_den+result["denominator"] != den:
                raise ValueError("Learner shards did not cover the complete update")
            accumulated, metrics = self.merge_gradients(accumulated, metrics,
                tf.convert_to_tensor(result["gradients"], tf.float32),
                tf.convert_to_tensor(result["metrics"], tf.float32), tf.constant(len(local)/len(packets)))
            fingerprints = [None]*len(packets)
            for indices, values in ((local, local_fingerprints), (remote, result["fingerprints"])):
                if len(indices) != len(values):
                    raise ValueError("Learner shard lost an episode fingerprint")
                for index, value in zip(indices, values):
                    fingerprints[index] = value
        if apply:
            self.apply(accumulated, tf.constant(learning_rate, tf.float32))
            self.step.numpy()
        values = metrics.numpy()
        record = {
            "update": update, "width": width, "loss": float(values[0]),
            "persistent_coded_bce": float(values[1]), "persistent_payload_loss": float(values[2]),
            "reset_coded_bce": float(values[3]), "reset_payload_loss": float(values[4]),
            "persistent_channel_nll": float(values[5]), "reset_channel_nll": float(values[6]),
            "gradient_norm": float(values[7]), "memory_gradient_norm": float(values[8]),
            "memory_rms": float(values[9]), "active_user_blocks": den,
            "learning_rate": learning_rate,
            "episode_fingerprints": fingerprints, "draws": [p["metadata"] for p in packets],
            "execution": self.execution.settings(),
        }
        return record, accumulated

    @tf.function(reduce_retracing=True)
    def flat_weights(self):
        return tf.concat([tf.reshape(v, [-1]) for v in self.optimization_variables], 0)

    @tf.function(reduce_retracing=True)
    def merge_gradients(self, local, local_metrics, remote, remote_metrics, local_weight):
        sizes = [int(np.prod(v.shape)) for v in self.optimization_variables]
        gradients = tuple(a+tf.reshape(b, a.shape) for a, b in zip(local, tf.split(remote, sizes)))
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient, "Nonfinite combined training gradient")
        norm = tf.linalg.global_norm(gradients)
        tf.debugging.assert_all_finite(norm, "Nonfinite combined gradient norm")
        metrics = tf.concat([local_metrics[:7]+remote_metrics[:7], [norm,
            tf.linalg.global_norm(gradients[:len(self.memory_variables)]),
            local_weight*local_metrics[9]+(1-local_weight)*remote_metrics[9]]], 0)
        return gradients, metrics

    def save(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        step = int(self.step.numpy())
        destination = output/f"checkpoint_{step:06d}"
        if destination.exists():
            raise FileExistsError(destination)
        temp = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=output))
        try:
            self.checkpoint.write(str(temp/"state"))
            self.model.save_weights(str(temp/"receiver.weights.h5"))
            settings = json.loads((ROOT/"config/memory_receiver.json").read_text())
            settings["architecture"] = self.settings["architecture"]
            settings["benchmark_memory"].update(capacity=self.manager.capacity,
                expiry_tbs=self.manager.expiry_slots)
            settings["training_updates"] = step
            atomic_json(temp/"cnn_autoencoder_finalist.json", settings)
            files = {p.name: file_sha256(p) for p in temp.iterdir() if p.is_file()}
            atomic_json(temp/"manifest.json", {"binding": self.binding(), "update": step,
                                              "schedule": self.schedule_state(), "files": files})
            for file in temp.iterdir():
                with file.open("rb") as handle:
                    os.fsync(handle.fileno())
            sync_directory(temp)
            os.replace(temp, destination)
            sync_directory(output)
            atomic_json(output/"latest.json", {"checkpoint": destination.name, "update": step})
        except BaseException:
            shutil.rmtree(temp, ignore_errors=True)
            raise
        return destination

    def restore(self, checkpoint):
        path = Path(checkpoint)
        manifest = json.loads((path/"manifest.json").read_text())
        validate_training_binding(manifest["binding"], self.binding())
        validate_checkpoint_files(path, manifest["files"])
        status = self.checkpoint.read(str(path/"state"))
        status.assert_consumed()
        if int(self.step.numpy()) != manifest["update"] or self.schedule_state() != manifest["schedule"]:
            raise ValueError("Restored update or learning-rate schedule differs")
        return int(self.step.numpy())
