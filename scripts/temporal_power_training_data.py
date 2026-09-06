#!/usr/bin/env python3
"""Add anonymous power variation to training episodes."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from temporal_training_data import TemporalTrainingDataGenerator


class PowerAwareTemporalTrainingDataGenerator(
    TemporalTrainingDataGenerator
):
    """Apply training-only per-UE channel attenuation."""

    def __init__(
        self,
        *args,
        received_power_augmentation,
        seed_streams,
        **kwargs,
    ):
        streams = self._validate_power_seed_streams(seed_streams)
        self.power_seed = streams["power"]
        self.received_power_augmentation = (
            self._validate_power_augmentation(
                received_power_augmentation
            )
        )
        base_streams = {
            name: streams[name]
            for name in ("channel", "payload", "noise", "schedule")
        }
        super().__init__(*args, seed_streams=base_streams, **kwargs)
        self.power_rng = np.random.RandomState(self.power_seed)
        self.last_power_attenuation_db = None
        self.last_power_augmented = None
        self._power_schedule = None

    @staticmethod
    def _validate_power_seed_streams(seed_streams):
        """Validate the five independent training streams."""
        required = {
            "channel", "payload", "noise", "schedule", "power"
        }
        if set(seed_streams or {}) != required:
            raise ValueError("power training needs five seed streams")
        values = {name: int(value) for name, value in seed_streams.items()}
        if any(value < 0 for value in values.values()):
            raise ValueError("power-training seeds cannot be negative")
        if len(set(values.values())) != len(values):
            raise ValueError("power-training seeds must be disjoint")
        return values

    @staticmethod
    def _validate_power_augmentation(value):
        """Validate the power-augmentation contract."""
        required = {
            "mode", "probability", "episode_span_db", "scaling"
        }
        if set(value or {}) != required:
            raise ValueError("received-power fields differ")
        if value["mode"] != "anonymous_episode_span_v1":
            raise ValueError("received-power mode differs")
        if value["scaling"] != "centered_raw_channel":
            raise ValueError("received-power scaling differs")
        probability = float(value["probability"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError("received-power probability is invalid")
        span = value["episode_span_db"]
        if len(span) != 2:
            raise ValueError("received-power span needs two values")
        low, high = (float(item) for item in span)
        if not 0.0 <= low <= high:
            raise ValueError("received-power span is invalid")
        return {
            "mode": value["mode"],
            "probability": probability,
            "episode_span_db": (low, high),
            "scaling": value["scaling"],
        }

    def _reserve_seed_batch(self, batch_size):
        """Attach the independent power seed to one batch."""
        start = self.seed_stream_consumed
        seeds = super()._reserve_seed_batch(batch_size)
        if seeds is not None:
            seeds["power"] = self.power_seed + start
        return seeds

    def _activate_seed_batch(self, seeds):
        """Reset every random stream for one episode."""
        super()._activate_seed_batch(seeds)
        if seeds is not None:
            self.power_rng = np.random.RandomState(seeds["power"])

    def seed_stream_metadata(self):
        """Expose all five deterministic streams."""
        value = super().seed_stream_metadata()
        value["base_seeds"] = {
            **value["base_seeds"],
            "power": self.power_seed,
        }
        return value

    def sample_schedule(self, batch_size, seq_len):
        """Record the identities eligible for power control."""
        schedule = super().sample_schedule(batch_size, seq_len)
        self._power_schedule = schedule.numpy()
        return schedule

    def _sample_power_attenuation(self, batch_size):
        """Assign stable anonymous offsets for one episode."""
        values = np.zeros(
            [int(batch_size), self.ue_pool_size], np.float32
        )
        augmented = np.zeros([int(batch_size)], bool)
        settings = self.received_power_augmentation
        low, high = settings["episode_span_db"]
        if self._power_schedule is None:
            raise RuntimeError("power schedule is unavailable")
        for batch in range(int(batch_size)):
            if self.power_rng.random() >= settings["probability"]:
                continue
            observed = np.unique(self._power_schedule[batch])
            if observed.size < 2:
                continue
            span = float(self.power_rng.uniform(low, high))
            interior = self.power_rng.uniform(
                0.0, span, size=max(0, observed.size - 2)
            )
            levels = np.concatenate(([0.0], interior, [span]))
            self.power_rng.shuffle(levels)
            values[batch, observed] = levels.astype(np.float32)
            augmented[batch] = True
        return values, augmented

    @staticmethod
    def _apply_power_attenuation(h_pool, attenuation_db):
        """Apply a centered per-UE power separation."""
        attenuation = tf.convert_to_tensor(
            attenuation_db, tf.float32
        )
        center = tf.reduce_max(attenuation, axis=1, keepdims=True) / 2.0
        scale = tf.pow(10.0, (center - attenuation) / 20.0)
        shape = [tf.shape(h_pool)[0], 1, 1, 1, -1, 1, 1, 1]
        return h_pool * tf.cast(
            tf.reshape(scale, shape), h_pool.dtype
        )

    def sample_pool_channel(
        self, batch_size, seq_len, channel_regime=None
    ):
        """Scale a raw training-only channel pool."""
        h_pool = super().sample_pool_channel(
            batch_size, seq_len, channel_regime
        )
        attenuation, augmented = self._sample_power_attenuation(
            batch_size
        )
        self.last_power_attenuation_db = attenuation
        self.last_power_augmented = augmented
        return self._apply_power_attenuation(h_pool, attenuation)

    def sample_batch(self, *args, **kwargs):
        """Attach power metadata after signal generation."""
        self._power_schedule = None
        explicit = kwargs.get(
            "ue_ids", args[3] if len(args) > 3 else None
        )
        if explicit is not None:
            self._power_schedule = np.asarray(explicit, np.int32)
        result = super().sample_batch(*args, **kwargs)
        attenuation = tf.convert_to_tensor(
            self.last_power_attenuation_db, tf.float32
        )
        result["ue_power_attenuation_db"] = attenuation
        result["scheduled_power_attenuation_db"] = tf.gather(
            attenuation,
            result["ue_ids"],
            axis=1,
            batch_dims=1,
        )
        result["power_augmented"] = tf.convert_to_tensor(
            self.last_power_augmented, tf.bool
        )
        return result
