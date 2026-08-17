#!/usr/bin/env python3
"""Temporal UE-memory trainer with independent pooling and compression choices.

This is a thin v4 extension of train_temporal_ue_memory_v3.py. It preserves the
verified UE-ID routing, memory reader, compressor implementations, training
losses, and diagnostics, while making the final NRX-state pooling selectable:

    final NRX state [B,U,F,T,d_s]
        -> mean | attention | cnn pooling
    per-UE summary [B,U,d_s]
        -> writer | PCA | autoencoder
    memory [B,U,d_mem]
        -> identity-owned memory manager

The memory read gate intentionally keeps the original mean-pooled *initial*
state so pooling experiments change only the write/compression side.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


# v3 owns the established CLI. Pull --pooling out before importing v3 so all
# existing arguments remain source-compatible without duplicating its parser.
def _extract_pooling(argv):
    mode = "mean"
    cleaned = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--pooling":
            if i + 1 >= len(argv):
                raise SystemExit("--pooling requires mean, attention, or cnn")
            mode = argv[i + 1].lower()
            i += 2
            continue
        if arg.startswith("--pooling="):
            mode = arg.split("=", 1)[1].lower()
            i += 1
            continue
        cleaned.append(arg)
        i += 1
    if mode not in {"mean", "attention", "cnn"}:
        raise SystemExit(
            f"invalid --pooling {mode!r}; choose mean, attention, or cnn")
    return mode, cleaned


POOLING, _CLEAN_ARGV = _extract_pooling(sys.argv)
sys.argv[:] = _CLEAN_ARGV

# Import v3 first. v3 parses --gpu and sets CUDA_VISIBLE_DEVICES before it
# imports TensorFlow, which is required for true one-process-per-GPU isolation.
import train_temporal_ue_memory_v3 as v3

import tensorflow as tf
from sionna.utils import expand_to_rank

from temporal_pooling import build_pooler


class PooledTemporalUEMemoryCGNN(v3.TemporalUEMemoryCGNN):
    """v3 temporal receiver with a selectable final-state pooler."""

    def __init__(self, base_cgnn, d_mem, d_s, compression, **kwargs):
        super().__init__(
            base_cgnn,
            d_mem=d_mem,
            d_s=d_s,
            compression=compression,
            **kwargs,
        )
        self.pooling = POOLING
        self.pooler = build_pooler(self.pooling, d_s=self.d_s)

    @property
    def memory_variables(self):
        # Pooler parameters are memory-side parameters, so attention/CNN learn
        # during the memory-only warm-up without touching the shipped NRX base.
        return list(super().memory_variables) + list(self.pooler.trainable_variables)

    @property
    def temporal_check_variables(self):
        # Future-TB loss should cross both the chosen compressor and any learned
        # pooling path used to create TB1's memory.
        return (
            list(super().temporal_check_variables)
            + list(self.pooler.trainable_variables)
        )

    def cold_pooled_final(self, inputs):
        """Cold K-step state summarized by the selected pooler for PCA fitting."""
        y, pe, h_hat, active_tx, mcs_ue_mask = inputs
        s = self._initial_state(y, pe, h_hat, mcs_ue_mask)
        s = self._iterations(s, pe, active_tx)
        return self.pooler(s, training=False)

    def call(
        self,
        inputs,
        prev_memory=None,
        memory_gap=None,
        memory_valid=None,
        training=None,
    ):
        y, pe, h_hat, active_tx, mcs_ue_mask = inputs
        base = self.base
        s = self._initial_state(y, pe, h_hat, mcs_ue_mask)

        batch_size = tf.shape(s)[0]
        num_tx = tf.shape(s)[1]
        if prev_memory is None:
            prev_memory = tf.zeros(
                [batch_size, num_tx, self.d_mem], dtype=s.dtype)
        else:
            prev_memory = tf.cast(prev_memory, s.dtype)

        if memory_valid is None:
            memory_valid = tf.zeros([batch_size, num_tx], tf.bool)
        else:
            memory_valid = tf.cast(memory_valid, tf.bool)

        if memory_gap is None:
            memory_gap = tf.zeros([batch_size, num_tx], tf.int32)
        else:
            memory_gap = tf.cast(memory_gap, tf.int32)

        valid_f = tf.cast(memory_valid, s.dtype)
        safe_memory = tf.where(
            memory_valid[..., None],
            prev_memory,
            tf.zeros_like(prev_memory),
        )
        age_feature = tf.math.log1p(
            tf.cast(tf.maximum(memory_gap, 0), s.dtype))
        age_feature *= valid_f

        # Deliberately unchanged from v3: only final-state pooling varies.
        pooled_init = tf.reduce_mean(s, axis=[2, 3])
        gate_in = tf.concat(
            [
                pooled_init,
                safe_memory,
                age_feature[..., None],
                valid_f[..., None],
            ],
            axis=-1,
        )
        gate = self.mem_gate(gate_in) * valid_f[..., None]
        mem_delta = self.mem_in(safe_memory)
        s = s + gate[:, :, None, None, :] * mem_delta[:, :, None, None, :]

        s = self._iterations(s, pe, active_tx)

        if base._var_mcs_masking:
            llr_grid = base._readout_llrs[0](s)
            llr_grid = tf.gather(
                llr_grid,
                indices=tf.range(base._num_bits_per_symbol[0]),
                axis=-1,
            )
        else:
            llr_grid = base._readout_llrs[0](s)
        h_refined = base._readout_chest(s)

        # The only architecture change relative to v3.
        pooled_final = self.pooler(s, training=training)
        compressed = self.compressor(
            pooled_final,
            safe_memory,
            age_feature,
            memory_valid,
            training=training,
        )

        active_bool = tf.cast(active_tx, tf.bool)
        next_memory = tf.where(
            active_bool[..., None], compressed.memory, safe_memory)

        return (
            llr_grid,
            h_refined,
            next_memory,
            compressed.aux_loss,
            compressed.reconstruction_mse,
        )


def _annotate_v4_summary():
    """Add pooling metadata and collision-safe checkpoint naming to v3 output."""
    out = Path(v3.ARGS.output_dir)
    summary_path = out / "training_summary.json"
    if not summary_path.exists():
        return

    summary = json.loads(summary_path.read_text())
    old_checkpoint = Path(summary["checkpoint"])
    new_checkpoint = old_checkpoint.with_name(
        f"ue_memory_{POOLING}_{v3.ARGS.compression}_idaware_"
        f"d{v3.ARGS.d_mem}_k{v3.ARGS.num_it}.weights.h5"
    )
    if old_checkpoint.exists() and old_checkpoint != new_checkpoint:
        os.replace(old_checkpoint, new_checkpoint)

    summary["architecture"] = "ue_identity_aware_temporal_memory_v4_pooling"
    summary["pooling"] = POOLING
    summary["pooling_semantics"] = {
        "mean": "uniform mean over final NRX time/frequency locations",
        "attention": "learned softmax weighting over final NRX time/frequency locations",
        "cnn": "learned local 3x3 time/frequency features followed by global mean",
    }[POOLING]
    summary["checkpoint"] = str(new_checkpoint)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("V4_POOLING_SUMMARY=" + json.dumps({
        "pooling": POOLING,
        "compression": v3.ARGS.compression,
        "d_mem": v3.ARGS.d_mem,
        "checkpoint": str(new_checkpoint),
    }), flush=True)


def main():
    # Make v3's builder instantiate the pooling-aware subclass.
    v3.TemporalUEMemoryCGNN = PooledTemporalUEMemoryCGNN

    # Prevent default runs for different poolers from overwriting one another.
    if v3.ARGS.output_dir is None:
        v3.ARGS.output_dir = str(
            Path.home()
            / "sionna-srsran"
            / "temporal_reuse"
            / "ue_memory"
            / POOLING
            / v3.ARGS.compression
        )

    v3.main()
    _annotate_v4_summary()


if __name__ == "__main__":
    main()
