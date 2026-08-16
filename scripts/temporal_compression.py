#!/usr/bin/env python3
"""Compression backends for temporal Neural RX UE memory.

Each backend maps the same pooled final NRX state [B, U, d_s] to exactly
`d_mem` float32 values per scheduled UE. The UE identity/lifecycle manager is
separate and unchanged.

Modes
-----
writer:
    Existing task-aware learned temporal writer. It may mix the new candidate
    with the previous UE memory through a learned keep gate.

pca:
    Frozen linear PCA projection fitted on pooled K-step NRX states before the
    temporal training run. It stores the d_mem PCA coefficients directly.

autoencoder:
    Learned encoder bottleneck of width d_mem. A decoder is used only to define
    a reconstruction loss during training; the decoder is not part of the
    persistent state passed between TBs.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense


class CompressionOutput(NamedTuple):
    memory: tf.Tensor
    aux_loss: tf.Tensor
    reconstruction_mse: tf.Tensor


class LearnedWriterCompression(tf.keras.layers.Layer):
    """Original end-to-end learned temporal writer."""

    mode = "writer"

    def __init__(self, d_mem: int, **kwargs):
        super().__init__(**kwargs)
        self.d_mem = int(d_mem)
        self.hidden = Dense(64, activation="relu", name="writer_hidden")
        self.candidate = Dense(
            self.d_mem, activation="tanh", name="writer_candidate")
        self.keep = Dense(
            self.d_mem,
            activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(1.0),
            name="writer_keep",
        )

    @property
    def temporal_check_variables(self):
        return (
            self.hidden.trainable_variables
            + self.candidate.trainable_variables
            + self.keep.trainable_variables
        )

    def call(
        self,
        pooled_final,
        prev_memory,
        age_feature,
        memory_valid,
        training=None,
    ):
        dtype = pooled_final.dtype
        valid_f = tf.cast(memory_valid, dtype)
        safe_memory = tf.where(
            memory_valid[..., None],
            tf.cast(prev_memory, dtype),
            tf.zeros_like(tf.cast(prev_memory, dtype)),
        )
        write_in = tf.concat(
            [
                pooled_final,
                safe_memory,
                age_feature[..., None],
                valid_f[..., None],
            ],
            axis=-1,
        )
        z = self.hidden(write_in)
        candidate = self.candidate(z)
        keep = self.keep(z) * valid_f[..., None]
        memory = keep * safe_memory + (1.0 - keep) * candidate
        zero = tf.zeros([], dtype=dtype)
        return CompressionOutput(memory, zero, zero)


class PCACompression(tf.keras.layers.Layer):
    """Frozen PCA projection fitted on pooled final NRX states."""

    mode = "pca"

    def __init__(self, d_s: int, d_mem: int, **kwargs):
        super().__init__(**kwargs)
        self.d_s = int(d_s)
        self.d_mem = int(d_mem)
        if self.d_mem > self.d_s:
            raise ValueError(
                f"PCA d_mem={self.d_mem} cannot exceed pooled state "
                f"dimension d_s={self.d_s}"
            )

        self.mean = self.add_weight(
            "pca_mean",
            shape=[self.d_s],
            initializer="zeros",
            trainable=False,
        )
        init = np.zeros([self.d_s, self.d_mem], np.float32)
        init[: self.d_mem, : self.d_mem] = np.eye(
            self.d_mem, dtype=np.float32)
        self.components = self.add_weight(
            "pca_components",
            shape=[self.d_s, self.d_mem],
            initializer=tf.keras.initializers.Constant(init),
            trainable=False,
        )
        self.eigenvalues = self.add_weight(
            "pca_eigenvalues",
            shape=[self.d_mem],
            initializer="zeros",
            trainable=False,
        )
        self.fitted = self.add_weight(
            "pca_fitted",
            shape=[],
            dtype=tf.bool,
            initializer=tf.keras.initializers.Constant(False),
            trainable=False,
        )

    @property
    def temporal_check_variables(self):
        # Non-trainable, but GradientTape can explicitly watch this tensor.
        # In the temporal check TB2's loss can only reach it through the memory
        # produced after TB1.
        return [self.components]

    def set_basis(self, mean, components, eigenvalues):
        mean = np.asarray(mean, np.float32)
        components = np.asarray(components, np.float32)
        eigenvalues = np.asarray(eigenvalues, np.float32)
        if mean.shape != (self.d_s,):
            raise ValueError(
                f"mean has shape {mean.shape}, expected {(self.d_s,)}")
        if components.shape != (self.d_s, self.d_mem):
            raise ValueError(
                f"components has shape {components.shape}, expected "
                f"{(self.d_s, self.d_mem)}"
            )
        if eigenvalues.shape != (self.d_mem,):
            raise ValueError(
                f"eigenvalues has shape {eigenvalues.shape}, expected "
                f"{(self.d_mem,)}"
            )
        self.mean.assign(mean)
        self.components.assign(components)
        self.eigenvalues.assign(eigenvalues)
        self.fitted.assign(True)

    def call(
        self,
        pooled_final,
        prev_memory,
        age_feature,
        memory_valid,
        training=None,
    ):
        centered = pooled_final - tf.cast(self.mean, pooled_final.dtype)
        components = tf.cast(self.components, pooled_final.dtype)
        memory = tf.einsum("bud,dm->bum", centered, components)

        # Reconstruction is diagnostic only and does not alter the PCA basis.
        reconstructed = (
            tf.einsum("bum,dm->bud", memory, components)
            + tf.cast(self.mean, pooled_final.dtype)
        )
        reconstruction_mse = tf.reduce_mean(
            tf.square(reconstructed - pooled_final))
        zero = tf.zeros([], dtype=pooled_final.dtype)
        return CompressionOutput(memory, zero, reconstruction_mse)


class AutoencoderCompression(tf.keras.layers.Layer):
    """Learned autoencoder whose bottleneck is the temporal UE memory."""

    mode = "autoencoder"

    def __init__(self, d_s: int, d_mem: int, hidden_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.d_s = int(d_s)
        self.d_mem = int(d_mem)
        self.hidden_dim = int(hidden_dim)

        self.encoder_hidden = Dense(
            self.hidden_dim, activation="relu", name="ae_encoder_hidden")
        self.encoder_bottleneck = Dense(
            self.d_mem, activation=None, name="ae_bottleneck")
        self.decoder_hidden = Dense(
            self.hidden_dim, activation="relu", name="ae_decoder_hidden")
        self.decoder_out = Dense(
            self.d_s, activation=None, name="ae_reconstruction")

    @property
    def temporal_check_variables(self):
        # Only encoder variables create the memory passed from TB1 to TB2.
        return (
            self.encoder_hidden.trainable_variables
            + self.encoder_bottleneck.trainable_variables
        )

    def call(
        self,
        pooled_final,
        prev_memory,
        age_feature,
        memory_valid,
        training=None,
    ):
        z = self.encoder_hidden(pooled_final)
        memory = self.encoder_bottleneck(z)
        d = self.decoder_hidden(memory)
        reconstructed = self.decoder_out(d)

        # The target is fixed for the reconstruction objective. Gradients still
        # flow through the encoder input, so joint fine-tuning may learn states
        # that are both useful temporally and compressible.
        target = tf.stop_gradient(pooled_final)
        reconstruction_mse = tf.reduce_mean(
            tf.square(reconstructed - target))
        return CompressionOutput(
            memory, reconstruction_mse, reconstruction_mse)


def build_compressor(mode: str, d_s: int, d_mem: int):
    mode = str(mode).lower()
    if mode == "writer":
        return LearnedWriterCompression(
            d_mem=d_mem, name="compression_writer")
    if mode == "pca":
        return PCACompression(
            d_s=d_s, d_mem=d_mem, name="compression_pca")
    if mode == "autoencoder":
        return AutoencoderCompression(
            d_s=d_s,
            d_mem=d_mem,
            name="compression_autoencoder",
        )
    raise ValueError(
        f"Unknown compression mode {mode!r}; expected writer, pca, or autoencoder")


def fit_pca_numpy(samples, d_mem: int):
    """Fit PCA with an eigendecomposition of the feature covariance.

    Parameters
    ----------
    samples : array-like [N, d_s]
        Pooled final NRX states.
    d_mem : int
        Number of principal components retained.

    Returns
    -------
    mean, components, eigenvalues, stats
        components has shape [d_s, d_mem] with columns ordered from highest to
        lowest variance.
    """
    x = np.asarray(samples, np.float64)
    if x.ndim != 2:
        raise ValueError(f"samples must be rank-2, got shape {x.shape}")
    if x.shape[0] < 2:
        raise ValueError("PCA needs at least two state samples")
    if d_mem <= 0 or d_mem > x.shape[1]:
        raise ValueError(
            f"d_mem must be in [1, {x.shape[1]}], got {d_mem}")

    mean = x.mean(axis=0)
    xc = x - mean
    cov = (xc.T @ xc) / float(max(x.shape[0] - 1, 1))
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    all_eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = eigenvectors[:, order[:d_mem]]
    retained = all_eigenvalues[:d_mem]

    total_variance = float(np.sum(all_eigenvalues))
    retained_variance = float(np.sum(retained))
    ratio = (
        retained_variance / total_variance
        if total_variance > 0.0
        else 0.0
    )
    stats = {
        "num_samples": int(x.shape[0]),
        "state_dim": int(x.shape[1]),
        "d_mem": int(d_mem),
        "retained_variance": retained_variance,
        "total_variance": total_variance,
        "retained_variance_ratio": float(ratio),
    }
    return (
        mean.astype(np.float32),
        components.astype(np.float32),
        retained.astype(np.float32),
        stats,
    )
