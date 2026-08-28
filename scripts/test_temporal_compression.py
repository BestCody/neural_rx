#!/usr/bin/env python3
"""Unit checks for temporal memory compression backends."""

import json

import numpy as np
import tensorflow as tf

from temporal_compression import (
    AutoencoderCompression,
    LearnedWriterCompression,
    PCACompression,
    fit_pca_numpy,
)


def main():
    tf.random.set_seed(7)
    np.random.seed(7)

    batch = 3
    users = 2
    d_s = 12
    d_mem = 4
    pooled = tf.Variable(tf.random.normal([batch, users, d_s]))
    prev = tf.random.normal([batch, users, d_mem])
    age = tf.ones([batch, users], tf.float32)
    valid = tf.constant(
        [[True, True], [True, False], [False, True]], tf.bool)

    writer = LearnedWriterCompression(d_mem)
    w = writer(pooled, prev, age, valid, training=True)
    writer_ok = (
        tuple(w.memory.shape) == (batch, users, d_mem)
        and float(w.aux_loss.numpy()) == 0.0
        and len(writer.trainable_variables) > 0
    )

    ae = AutoencoderCompression(d_s, d_mem)
    with tf.GradientTape(persistent=True) as tape:
        a = ae(pooled, prev, age, valid, training=True)
        ae_loss = a.aux_loss
        temporal_probe = tf.reduce_sum(a.memory)
    ae_grads = tape.gradient(ae_loss, ae.trainable_variables)
    ae_grad_norm = tf.linalg.global_norm(
        [g for g in ae_grads if g is not None])
    reconstruction_upstream_grad = tape.gradient(ae_loss, pooled)
    temporal_upstream_grad = tape.gradient(temporal_probe, pooled)
    del tape

    memory_abs_max = float(tf.reduce_max(tf.abs(a.memory)).numpy())
    reconstruction_detached = (
        reconstruction_upstream_grad is None
        or float(tf.linalg.global_norm([reconstruction_upstream_grad]).numpy()) == 0.0
    )
    temporal_connected = (
        temporal_upstream_grad is not None
        and float(tf.linalg.global_norm([temporal_upstream_grad]).numpy()) > 0.0
    )
    autoencoder_ok = (
        tuple(a.memory.shape) == (batch, users, d_mem)
        and np.isfinite(float(a.reconstruction_mse.numpy()))
        and float(a.reconstruction_mse.numpy()) > 0.0
        and np.isfinite(float(a.aux_loss.numpy()))
        and float(a.aux_loss.numpy()) > 0.0
        and float(ae_grad_norm.numpy()) > 0.0
        and memory_abs_max <= 1.000001
        and reconstruction_detached
        and temporal_connected
    )

    # Synthetic data with descending feature scales gives a well-defined PCA
    # spectrum and enough samples for a stable rank-d_mem fit.
    scales = np.linspace(3.0, 0.2, d_s, dtype=np.float32)
    x = np.random.randn(512, d_s).astype(np.float32) * scales
    mean, components, eigenvalues, stats = fit_pca_numpy(x, d_mem)
    pca = PCACompression(d_s, d_mem)
    pca.set_basis(mean, components, eigenvalues)
    p = pca(pooled, prev, age, valid, training=False)

    gram = components.T @ components
    pca_ok = (
        tuple(p.memory.shape) == (batch, users, d_mem)
        and bool(pca.fitted.numpy())
        and np.allclose(gram, np.eye(d_mem), atol=1e-4)
        and np.isfinite(float(p.reconstruction_mse.numpy()))
        and 0.0 < stats["retained_variance_ratio"] <= 1.0
    )

    dimension_guard_ok = False
    try:
        PCACompression(d_s=4, d_mem=5)
    except ValueError:
        dimension_guard_ok = True

    summary = {
        "writer": {
            "shape": list(w.memory.shape),
            "trainable_variables": len(writer.trainable_variables),
            "passed": bool(writer_ok),
        },
        "autoencoder": {
            "shape": list(a.memory.shape),
            "memory_abs_max": memory_abs_max,
            "normalized_reconstruction_loss": float(a.aux_loss.numpy()),
            "reconstruction_mse": float(a.reconstruction_mse.numpy()),
            "gradient_norm": float(ae_grad_norm.numpy()),
            "reconstruction_detached_from_upstream_state": bool(reconstruction_detached),
            "temporal_memory_connected_to_upstream_state": bool(temporal_connected),
            "passed": bool(autoencoder_ok),
        },
        "pca": {
            "shape": list(p.memory.shape),
            "retained_variance_ratio": stats["retained_variance_ratio"],
            "reconstruction_mse": float(p.reconstruction_mse.numpy()),
            "orthonormal_components": bool(
                np.allclose(gram, np.eye(d_mem), atol=1e-4)),
            "dimension_guard": bool(dimension_guard_ok),
            "passed": bool(pca_ok and dimension_guard_ok),
        },
        "same_memory_width": bool(
            w.memory.shape[-1]
            == a.memory.shape[-1]
            == p.memory.shape[-1]
            == d_mem
        ),
    }
    summary["passed"] = bool(
        summary["writer"]["passed"]
        and summary["autoencoder"]["passed"]
        and summary["pca"]["passed"]
        and summary["same_memory_width"]
    )
    print("TEMPORAL_COMPRESSION_TEST=" + json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit("Temporal compression test failed")


if __name__ == "__main__":
    main()
