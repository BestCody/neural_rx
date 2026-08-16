#!/usr/bin/env python3
"""Pooling backends for temporal Neural RX UE memory.

Pooling and compression are deliberately separate operations:

    final NRX state [B, U, F, T, d_s]
        -> pooler
    per-UE summary [B, U, d_s]
        -> PCA / autoencoder / learned writer
    persistent memory [B, U, d_mem]

Every pooler returns the same d_s-wide per-UE representation so compression
experiments keep exactly the same downstream interface and memory cap.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense


class MeanTemporalPooling(tf.keras.layers.Layer):
    """Exact baseline used by the original temporal-memory implementation."""

    mode = "mean"

    def __init__(self, d_s: int, **kwargs):
        super().__init__(**kwargs)
        self.d_s = int(d_s)

    def call(self, state, training=None):
        del training
        return tf.reduce_mean(state, axis=[2, 3])


class AttentionTemporalPooling(tf.keras.layers.Layer):
    """Learn a scalar importance weight for every time/frequency location.

    The learned scores select *where* to summarize from. The weighted values are
    the original d_s-dimensional NRX states, so the output width remains d_s.
    """

    mode = "attention"

    def __init__(self, d_s: int, hidden_dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.d_s = int(d_s)
        self.hidden_dim = int(hidden_dim)
        self.score_hidden = Dense(
            self.hidden_dim,
            activation="tanh",
            name="attention_score_hidden",
        )
        self.score_out = Dense(1, activation=None, name="attention_score")

    def call(self, state, training=None):
        del training
        state = tf.convert_to_tensor(state)
        shape = tf.shape(state)
        batch_size = shape[0]
        num_ues = shape[1]
        d_s = shape[4]

        score = self.score_out(self.score_hidden(state))
        score = tf.squeeze(score, axis=-1)
        score = tf.reshape(score, [batch_size, num_ues, -1])
        weights = tf.nn.softmax(score, axis=-1)

        values = tf.reshape(state, [batch_size, num_ues, -1, d_s])
        return tf.reduce_sum(values * weights[..., None], axis=2)

    def attention_weights(self, state):
        """Return [B,U,F*T] weights for diagnostics/visualization."""
        state = tf.convert_to_tensor(state)
        shape = tf.shape(state)
        score = self.score_out(self.score_hidden(state))
        score = tf.squeeze(score, axis=-1)
        score = tf.reshape(score, [shape[0], shape[1], -1])
        return tf.nn.softmax(score, axis=-1)


class CNNTemporalPooling(tf.keras.layers.Layer):
    """Learn local time/frequency patterns before global reduction.

    A 3x3 convolution captures neighboring structure and a 1x1 projection
    returns to d_s channels. Global averaging then produces one d_s-wide vector
    per UE. This is intentionally small so pooling does not dominate NRX cost.
    """

    mode = "cnn"

    def __init__(self, d_s: int, hidden_dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.d_s = int(d_s)
        self.hidden_dim = int(hidden_dim)
        self.local = Conv2D(
            self.hidden_dim,
            kernel_size=3,
            padding="same",
            activation="relu",
            name="cnn_pool_local",
        )
        self.project = Conv2D(
            self.d_s,
            kernel_size=1,
            padding="same",
            activation=None,
            name="cnn_pool_project",
        )

    def call(self, state, training=None):
        state = tf.convert_to_tensor(state)
        shape = tf.shape(state)
        batch_size = shape[0]
        num_ues = shape[1]
        freq = shape[2]
        time = shape[3]
        d_s = shape[4]

        x = tf.reshape(state, [batch_size * num_ues, freq, time, d_s])
        x = self.local(x, training=training)
        x = self.project(x, training=training)
        x = tf.reduce_mean(x, axis=[1, 2])
        return tf.reshape(x, [batch_size, num_ues, self.d_s])


def build_pooler(mode: str, d_s: int):
    """Construct a pooling backend with the common [B,U,d_s] interface."""
    mode = str(mode).lower()
    if mode == "mean":
        return MeanTemporalPooling(d_s=d_s, name="pooling_mean")
    if mode == "attention":
        return AttentionTemporalPooling(d_s=d_s, name="pooling_attention")
    if mode == "cnn":
        return CNNTemporalPooling(d_s=d_s, name="pooling_cnn")
    raise ValueError(
        f"Unknown pooling mode {mode!r}; expected mean, attention, or cnn")
