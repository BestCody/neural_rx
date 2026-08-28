#!/usr/bin/env python3
"""Unit checks for long-horizon streaming/TBPTT state handling."""

import json

import numpy as np
import tensorflow as tf

from temporal_streaming_training import (
    detach_and_randomly_reset,
    detach_memory_state,
    iter_tbptt_windows,
    reset_memory_entries,
    validate_streaming_config,
)
from ue_memory_manager import TensorMemoryState


def main():
    windows = list(iter_tbptt_windows(10, 4))
    assert windows == [(0, 4), (4, 8), (8, 10)]

    source = tf.Variable([[[1.0, 2.0], [3.0, 4.0]]])
    with tf.GradientTape() as tape:
        state = TensorMemoryState(
            source * 2.0,
            tf.constant([[True, True]]),
            tf.constant([[5, 7]], tf.int32),
        )
        detached = detach_memory_state(state)
        probe = tf.reduce_sum(detached.memory)
    assert tape.gradient(probe, source) is None
    assert np.allclose(detached.memory.numpy(), (source * 2.0).numpy())

    reset = reset_memory_entries(
        detached, tf.constant([[True, False]], tf.bool)
    )
    assert np.allclose(reset.memory.numpy()[0, 0], 0.0)
    assert np.allclose(reset.memory.numpy()[0, 1], [6.0, 8.0])
    assert reset.valid.numpy().tolist() == [[False, True]]
    assert reset.last_seen.numpy().tolist() == [[-1, 7]]

    reset_all, fraction = detach_and_randomly_reset(detached, 1.0)
    assert np.allclose(reset_all.memory.numpy(), 0.0)
    assert not np.any(reset_all.valid.numpy())
    assert float(fraction.numpy()) == 1.0

    validate_streaming_config(64, 4, 0.05)
    guards = 0
    for args in [
        (1, 1, 0.0),
        (64, 1, 0.0),
        (4, 8, 0.0),
        (10, 4, 0.0),
        (4, 2, 1.1),
    ]:
        try:
            validate_streaming_config(*args)
        except ValueError:
            guards += 1
    assert guards == 5

    summary = {
        "windows_cover_full_episode": True,
        "state_values_cross_windows": True,
        "gradient_history_is_truncated": True,
        "selective_cold_reset": True,
        "configuration_guards": True,
        "passed": True,
    }
    print("TEMPORAL_STREAMING_TRAINING_TEST=" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
