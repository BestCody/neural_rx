#!/usr/bin/env python3
"""TBPTT helpers for streaming temporal training."""

from __future__ import annotations

from typing import Iterator, Tuple

import tensorflow as tf

from ue_memory_manager import TensorMemoryState


def validate_streaming_config(
    stream_len: int,
    tbptt_window: int,
    memory_reset_prob: float,
) -> None:
    """Validate the streaming training settings."""
    if stream_len < 2:
        raise ValueError("stream_len must be at least 2 TBs")
    if tbptt_window < 2:
        raise ValueError(
            "tbptt_window must be at least 2 TBs so future-TB loss can train "
            "the memory writer"
        )
    if tbptt_window > stream_len:
        raise ValueError("tbptt_window cannot exceed stream_len")
    if stream_len % tbptt_window != 0:
        raise ValueError(
            "stream_len must be divisible by tbptt_window so every optimizer "
            "update has the same temporal depth"
        )
    if not 0.0 <= memory_reset_prob <= 1.0:
        raise ValueError("memory_reset_prob must be in [0, 1]")


def iter_tbptt_windows(
    stream_len: int,
    tbptt_window: int,
) -> Iterator[Tuple[int, int]]:
    """Yield half-open TBPTT windows."""
    if stream_len <= 0 or tbptt_window <= 0:
        raise ValueError("stream_len and tbptt_window must be positive")
    for start in range(0, int(stream_len), int(tbptt_window)):
        yield start, min(start + int(tbptt_window), int(stream_len))


def detach_memory_state(state: TensorMemoryState) -> TensorMemoryState:
    """Detach memory from its gradient history."""
    return TensorMemoryState(
        tf.stop_gradient(state.memory),
        tf.stop_gradient(state.valid),
        tf.stop_gradient(state.last_seen),
    )


def reset_memory_entries(
    state: TensorMemoryState,
    reset_mask,
) -> TensorMemoryState:
    """Cold-reset selected UE memories."""
    mask = tf.cast(reset_mask, tf.bool)
    tf.debugging.assert_equal(
        tf.shape(mask),
        tf.shape(state.valid),
        message="reset_mask must have shape [batch, memory capacity]",
    )
    return TensorMemoryState(
        tf.where(mask[..., None], tf.zeros_like(state.memory), state.memory),
        tf.where(mask, tf.zeros_like(state.valid), state.valid),
        tf.where(
            mask,
            tf.fill(tf.shape(state.last_seen), tf.constant(-1, tf.int32)),
            state.last_seen,
        ),
    )


def detach_and_randomly_reset(
    state: TensorMemoryState,
    reset_probability: float,
) -> tuple[TensorMemoryState, tf.Tensor]:
    """Detach memory and apply random cold resets."""
    state = detach_memory_state(state)
    probability = float(reset_probability)
    if probability == 0.0:
        mask = tf.zeros_like(state.valid)
    elif probability == 1.0:
        mask = tf.identity(state.valid)
    else:
        mask = tf.logical_and(
            state.valid,
            tf.random.uniform(tf.shape(state.valid), dtype=tf.float32)
            < probability,
        )
    reset_fraction = tf.reduce_mean(tf.cast(mask, tf.float32))
    return reset_memory_entries(state, mask), reset_fraction
