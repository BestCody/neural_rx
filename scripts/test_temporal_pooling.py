#!/usr/bin/env python3
"""Unit checks for mean, attention, and CNN temporal pooling."""

import json

import numpy as np
import tensorflow as tf

from temporal_pooling import build_pooler


def gradient_norm(pooler, state):
    with tf.GradientTape() as tape:
        out = pooler(state, training=True)
        loss = tf.reduce_sum(tf.square(out))
    grads = tape.gradient(loss, pooler.trainable_variables)
    grads = [g for g in grads if g is not None]
    return float(tf.linalg.global_norm(grads).numpy()) if grads else 0.0


def main():
    tf.random.set_seed(20260816)
    state = tf.random.normal([3, 2, 11, 7, 56])
    expected_shape = [3, 2, 56]
    report = {}

    mean = build_pooler("mean", 56)
    mean_out = mean(state)
    exact = tf.reduce_mean(state, axis=[2, 3])
    report["mean"] = {
        "shape": list(mean_out.shape),
        "matches_original_mean": bool(np.allclose(
            mean_out.numpy(), exact.numpy(), atol=1e-6)),
        "trainable_variables": len(mean.trainable_variables),
    }
    report["mean"]["passed"] = bool(
        report["mean"]["shape"] == expected_shape
        and report["mean"]["matches_original_mean"]
        and report["mean"]["trainable_variables"] == 0
    )

    attention = build_pooler("attention", 56)
    att_out = attention(state, training=True)
    weights = attention.attention_weights(state)
    att_grad = gradient_norm(attention, state)
    report["attention"] = {
        "shape": list(att_out.shape),
        "weights_shape": list(weights.shape),
        "weights_sum_mean": float(tf.reduce_mean(
            tf.reduce_sum(weights, axis=-1)).numpy()),
        "gradient_norm": att_grad,
    }
    report["attention"]["passed"] = bool(
        report["attention"]["shape"] == expected_shape
        and np.allclose(tf.reduce_sum(weights, axis=-1).numpy(), 1.0, atol=1e-6)
        and att_grad > 0.0
    )

    cnn = build_pooler("cnn", 56)
    cnn_out = cnn(state, training=True)
    cnn_grad = gradient_norm(cnn, state)
    report["cnn"] = {
        "shape": list(cnn_out.shape),
        "gradient_norm": cnn_grad,
    }
    report["cnn"]["passed"] = bool(
        report["cnn"]["shape"] == expected_shape
        and cnn_grad > 0.0
    )

    report["same_output_width"] = bool(all(
        report[m]["shape"] == expected_shape
        for m in ["mean", "attention", "cnn"]
    ))
    report["passed"] = bool(
        report["same_output_width"]
        and all(report[m]["passed"] for m in ["mean", "attention", "cnn"])
    )
    print("TEMPORAL_POOLING_TEST=" + json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
