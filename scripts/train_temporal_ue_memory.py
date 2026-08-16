#!/usr/bin/env python3
"""Train NRX over correlated TB sequences with persistent per-UE memory.

The shipped NVIDIA training loop samples one independent TB per optimizer step.
This experiment changes the sampling/training unit to a short ordered sequence:

    TB1 -> memory -> TB2 -> memory -> TB3 -> ...

A single GradientTape spans the full sequence, so a later TB loss can update the
memory writer that produced an earlier TB's memory. The base NRX remains the
pretrained CGNN; this wrapper only adds a compact per-UE memory read/write path.

Run from neural_rx/scripts, or copy next to temporal_training_data.py and run
with neural_rx/scripts as the working directory.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="nrx_large.cfg")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--d-mem", type=int, default=32)
    p.add_argument("--num-it", type=int, default=2)
    p.add_argument("--train-steps", type=int, default=6000)
    p.add_argument("--memory-only-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=4)
    p.add_argument("--min-ebno-db", type=float, default=1.0)
    p.add_argument("--max-ebno-db", type=float, default=5.0)
    p.add_argument("--memory-lr", type=float, default=1e-3)
    p.add_argument("--joint-lr", type=float, default=2e-5)
    p.add_argument("--chest-weight", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args()


ARGS = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import sionna as sn
from tensorflow.keras.layers import Dense
from sionna.utils import expand_to_rank, flatten_last_dims

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "..")

from temporal_training_data import TemporalTrainingDataGenerator
from utils import Parameters, E2E_Model, load_weights


class TemporalUEMemoryCGNN(tf.keras.Model):
    """Pretrained CGNN plus a compact recurrent memory vector for each UE."""

    def __init__(self, base_cgnn, d_mem, d_s, **kwargs):
        super().__init__(**kwargs)
        self.base = base_cgnn
        self.d_mem = int(d_mem)
        self.d_s = int(d_s)

        # Read old memory into the current TB state. Start with a small gate so
        # the untrained memory path does not destroy the pretrained cold NRX.
        self.mem_in = Dense(d_s, activation="tanh", name="ue_mem_in")
        self.mem_gate = Dense(
            d_s,
            activation="sigmoid",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
            bias_initializer=tf.keras.initializers.Constant(-3.0),
            name="ue_mem_gate",
        )

        # Write the final current-TB state back into a compact per-UE vector.
        self.mem_hidden = Dense(64, activation="relu", name="ue_mem_hidden")
        self.mem_candidate = Dense(
            d_mem, activation="tanh", name="ue_mem_candidate")
        self.mem_keep = Dense(
            d_mem,
            activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(1.0),
            name="ue_mem_keep",
        )

    @property
    def memory_variables(self):
        layers = [
            self.mem_in,
            self.mem_gate,
            self.mem_hidden,
            self.mem_candidate,
            self.mem_keep,
        ]
        return [v for layer in layers for v in layer.trainable_variables]

    @property
    def memory_writer_variables(self):
        layers = [self.mem_hidden, self.mem_candidate, self.mem_keep]
        return [v for layer in layers for v in layer.trainable_variables]

    def zero_memory(self, batch_size, num_tx, dtype=tf.float32):
        return tf.zeros([batch_size, num_tx, self.d_mem], dtype=dtype)

    def _initial_state(self, y, pe, h_hat, mcs_ue_mask):
        base = self.base

        # Exact normalization used by the shipped CGNN.
        norm = tf.reduce_mean(tf.square(y), axis=(1, 2, 3), keepdims=True)
        norm = tf.math.divide_no_nan(1.0, tf.sqrt(norm))
        y = y * norm
        if h_hat is not None:
            h_hat = h_hat * tf.expand_dims(norm, axis=1)

        # Exact pretrained state initialization.
        if base._var_mcs_masking:
            s = base._s_init[0]((y, pe, h_hat))
        else:
            s = base._s_init[0]((y, pe, h_hat)) * expand_to_rank(
                tf.gather(mcs_ue_mask, indices=0, axis=2), 5, axis=-1)
            for idx in range(1, base._num_mcss_supported):
                s += base._s_init[idx]((y, pe, h_hat)) * expand_to_rank(
                    tf.gather(mcs_ue_mask, indices=idx, axis=2), 5, axis=-1)
        return s

    def call(self, inputs, prev_memory=None):
        y, pe, h_hat, active_tx, mcs_ue_mask = inputs
        base = self.base
        s = self._initial_state(y, pe, h_hat, mcs_ue_mask)

        batch_size = tf.shape(s)[0]
        num_tx = tf.shape(s)[1]
        if prev_memory is None:
            prev_memory = self.zero_memory(batch_size, num_tx, s.dtype)
        else:
            prev_memory = tf.cast(prev_memory, s.dtype)

        # Read: combine old compact memory with the fresh state initialized from
        # the current signal/LS estimate. Memory is broadcast over the grid.
        pooled_init = tf.reduce_mean(s, axis=[2, 3])
        gate_in = tf.concat([pooled_init, prev_memory], axis=-1)
        gate = self.mem_gate(gate_in)
        mem_delta = self.mem_in(prev_memory)
        s = s + gate[:, :, None, None, :] * mem_delta[:, :, None, None, :]

        # Preserve the normal NRX iteration blocks. Only the requested first
        # num_it blocks are active (K=2 in the primary experiment).
        for i in range(base._num_it):
            s = base._iterations[i]([s, pe, active_tx])

        # Standard final readouts for the single-MCS experiment.
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

        # Write: summarize the final state of each UE and update only that UE's
        # own compact memory. This tensor stays connected to the graph so later
        # TB losses can backpropagate into this writer.
        pooled_final = tf.reduce_mean(s, axis=[2, 3])
        write_in = tf.concat([pooled_final, prev_memory], axis=-1)
        z = self.mem_hidden(write_in)
        candidate = self.mem_candidate(z)
        keep = self.mem_keep(z)
        next_memory = keep * prev_memory + (1.0 - keep) * candidate
        next_memory *= tf.cast(active_tx[..., None], next_memory.dtype)

        return llr_grid, h_refined, next_memory


def demap_llr(ofdm, llr_grid, num_tx, mcs_idx=0):
    """Use the shipped NRX resource-grid/layer demappers."""
    llr = tf.cast(llr_grid, tf.float32)
    llr = tf.transpose(llr, [0, 1, 3, 2, 4])
    llr = tf.expand_dims(llr, axis=1)
    llr = ofdm._rg_demapper(llr)
    llr = llr[:, :num_tx]
    llr = flatten_last_dims(llr, 2)
    if ofdm._layer_demappers is None:
        llr = tf.squeeze(llr, axis=-2)
    else:
        llr = ofdm._layer_demappers[mcs_idx](llr)
    return llr


def temporal_forward(receiver, temporal_model, y, h_hat, active_tx, memory):
    """Shipped OFDM preprocessing -> temporal CGNN -> shipped demapping."""
    ofdm = receiver._neural_rx
    num_tx = tf.shape(active_tx)[1]

    y2 = y[:, 0]
    y2 = tf.transpose(y2, [0, 3, 2, 1])
    y2 = tf.concat([tf.math.real(y2), tf.math.imag(y2)], axis=-1)
    pe = ofdm._nearest_pilot_dist[:num_tx]

    y2 = tf.cast(y2, ofdm._nrx_dtype)
    pe = tf.cast(pe, ofdm._nrx_dtype)
    h_hat = tf.cast(h_hat, ofdm._nrx_dtype)
    active = tf.cast(active_tx, ofdm._nrx_dtype)
    mcs_mask = tf.ones([tf.shape(y2)[0], num_tx, 1], tf.float32)

    llr_grid, h_ref, next_memory = temporal_model(
        [y2, pe, h_hat, active, mcs_mask], memory)
    llr = demap_llr(ofdm, llr_grid, num_tx, 0)
    return llr, tf.cast(h_ref, tf.float32), next_memory


def build():
    # Use the same small training resource grid as the shipped NRX training.
    # The temporal generator replaces the independent UMi draw with continuous
    # TDL sequences, but transmitter/receiver preprocessing stays NRX-native.
    p = Parameters(ARGS.config, training=True, system="nrx")
    e2e = E2E_Model(p, training=True)
    e2e(1, 1.0)  # build variables before loading the pretrained checkpoint
    load_weights(e2e, f"../weights/{p.label}_weights")

    base = e2e._receiver._neural_rx._cgnn
    base.num_it = ARGS.num_it
    temporal_model = TemporalUEMemoryCGNN(
        base, d_mem=ARGS.d_mem, d_s=p.d_s,
        name=f"temporal_ue_memory_d{ARGS.d_mem}")
    generator = TemporalTrainingDataGenerator(p, e2e)
    return p, e2e, temporal_model, generator


def make_losses(receiver, temporal_model, batch):
    """One connected forward pass over every TB in the sequence."""
    batch_size = tf.shape(batch["bits"])[0]
    seq_len = batch["bits"].shape[1]
    num_tx = tf.shape(batch["active"])[2]
    memory = temporal_model.zero_memory(batch_size, num_tx)

    data_losses = []
    chest_losses = []
    memory_norms = []

    for t in range(seq_len):
        bits_t = batch["bits"][:, t]
        y_t = batch["y"][:, t]
        ls_t = batch["ls"][:, t]
        h_t = batch["h"][:, t]
        active_t = batch["active"][:, t]

        # Labels match the shipped NRX training loss: re-encode payload bits.
        coded_t = receiver._tb_encoders[0](bits_t)
        h_true_t = receiver.preprocess_channel_ground_truth(h_t)

        llr_t, h_ref_t, memory = temporal_forward(
            receiver, temporal_model, y_t, ls_t, active_t, memory)

        data_loss_t = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(coded_t, tf.float32), logits=llr_t))
        chest_loss_t = tf.reduce_mean(tf.square(h_ref_t - h_true_t))

        data_losses.append(data_loss_t)
        chest_losses.append(chest_loss_t)
        memory_norms.append(tf.reduce_mean(tf.norm(memory, axis=-1)))

    loss_data = tf.add_n(data_losses) / float(seq_len)
    loss_chest = tf.add_n(chest_losses) / float(seq_len)
    total = loss_data + ARGS.chest_weight * loss_chest
    return total, loss_data, loss_chest, data_losses, memory_norms


def temporal_gradient_check(receiver, temporal_model, generator):
    """Verify TB2 loss reaches the memory writer used after TB1."""
    batch = generator.sample_batch(2, 2, 3.0)
    with tf.GradientTape() as tape:
        memory = temporal_model.zero_memory(2, 2)

        bits0 = batch["bits"][:, 0]
        _, _, memory = temporal_forward(
            receiver,
            temporal_model,
            batch["y"][:, 0],
            batch["ls"][:, 0],
            batch["active"][:, 0],
            memory,
        )

        bits1 = batch["bits"][:, 1]
        coded1 = receiver._tb_encoders[0](bits1)
        llr1, _, _ = temporal_forward(
            receiver,
            temporal_model,
            batch["y"][:, 1],
            batch["ls"][:, 1],
            batch["active"][:, 1],
            memory,
        )
        future_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(coded1, tf.float32), logits=llr1))

    grads = tape.gradient(future_loss, temporal_model.memory_writer_variables)
    valid = [g for g in grads if g is not None]
    norm = tf.linalg.global_norm(valid) if valid else tf.constant(0.0)
    return float(future_loss.numpy()), float(norm.numpy())


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        sn.config.seed = seed
    except Exception:
        pass


def main():
    set_seed(ARGS.seed)
    p, e2e, model, generator = build()
    receiver = e2e._receiver

    # Build the new memory layers once before selecting variable groups.
    warmup = generator.sample_batch(1, ARGS.seq_len, 3.0)
    _ = make_losses(receiver, model, warmup)

    future_loss, temporal_grad_norm = temporal_gradient_check(
        receiver, model, generator)
    print(
        "TEMPORAL_GRADIENT_CHECK=" + json.dumps({
            "tb2_only_loss": future_loss,
            "memory_writer_grad_norm": temporal_grad_norm,
            "passed": temporal_grad_norm > 0.0,
        }),
        flush=True,
    )
    if not temporal_grad_norm > 0.0:
        raise RuntimeError(
            "Temporal gradient check failed: later TB loss did not reach "
            "the earlier memory writer.")

    memory_opt = tf.keras.optimizers.Adam(ARGS.memory_lr)
    joint_opt = tf.keras.optimizers.Adam(ARGS.joint_lr)

    out = Path(ARGS.output_dir or (
        Path.home() / "sionna-srsran" / "temporal_reuse" / "ue_memory"))
    out.mkdir(parents=True, exist_ok=True)

    history = []
    start = time.time()
    # Reset after layer construction/check so experiment streams are repeatable.
    set_seed(ARGS.seed)

    for step in range(ARGS.train_steps):
        ebno = float(np.random.uniform(ARGS.min_ebno_db, ARGS.max_ebno_db))
        batch = generator.sample_batch(ARGS.batch_size, ARGS.seq_len, ebno)

        with tf.GradientTape() as tape:
            total, loss_data, loss_chest, per_tb, memory_norms = make_losses(
                receiver, model, batch)

        if step < ARGS.memory_only_steps:
            variables = model.memory_variables
            optimizer = memory_opt
            phase = "memory_only"
        else:
            variables = model.trainable_variables
            optimizer = joint_opt
            phase = "joint"

        grads = tape.gradient(total, variables)
        pairs = [(g, v) for g, v in zip(grads, variables) if g is not None]
        if not pairs:
            raise RuntimeError("No gradients reached the selected variables.")
        optimizer.apply_gradients(pairs)
        grad_norm = float(tf.linalg.global_norm([g for g, _ in pairs]).numpy())

        if step % ARGS.log_every == 0 or step == ARGS.train_steps - 1:
            row = {
                "step": step,
                "phase": phase,
                "ebno_db": ebno,
                "loss": float(total.numpy()),
                "loss_data": float(loss_data.numpy()),
                "loss_chest": float(loss_chest.numpy()),
                "loss_per_tb": [float(x.numpy()) for x in per_tb],
                "memory_norm_per_tb": [float(x.numpy()) for x in memory_norms],
                "gradient_norm": grad_norm,
                "seconds": time.time() - start,
            }
            history.append(row)
            print("TRAIN=" + json.dumps(row), flush=True)

    checkpoint = out / f"ue_memory_d{ARGS.d_mem}_k{ARGS.num_it}.weights.h5"
    model.save_weights(str(checkpoint))

    summary = {
        "config": ARGS.config,
        "d_mem": ARGS.d_mem,
        "num_it": ARGS.num_it,
        "train_steps": ARGS.train_steps,
        "memory_only_steps": ARGS.memory_only_steps,
        "batch_size": ARGS.batch_size,
        "seq_len": ARGS.seq_len,
        "temporal_gradient_check": {
            "tb2_only_loss": future_loss,
            "memory_writer_grad_norm": temporal_grad_norm,
            "passed": temporal_grad_norm > 0.0,
        },
        "checkpoint": str(checkpoint),
        "history": history,
    }
    (out / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print("TRAINING_SUMMARY=" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
