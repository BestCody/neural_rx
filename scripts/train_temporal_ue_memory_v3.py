#!/usr/bin/env python3
"""Train UE-aware temporal Neural RX with selectable memory compression.

The UE identity/lifecycle architecture is shared across every experiment.
Only the final-state -> d_mem compression path changes:

    writer       task-aware learned temporal writer (existing method)
    pca          frozen PCA projection fitted before temporal training
    autoencoder  learned encoder bottleneck + reconstruction loss

All modes pass exactly d_mem float32 values per UE through the same stable
UE-ID memory manager.
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
    p.add_argument(
        "--compression",
        choices=["writer", "pca", "autoencoder"],
        default="writer",
    )
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
    p.add_argument("--ae-reconstruction-weight", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--log-every", type=int, default=25)

    # PCA is fitted once on frozen/cold K-step final states, then frozen.
    p.add_argument("--pca-fit-batches", type=int, default=16)
    p.add_argument("--pca-fit-batch-size", type=int, default=8)

    # UE identity/lifecycle architecture.
    p.add_argument("--ue-pool-size", type=int, default=4)
    p.add_argument("--memory-expiry-slots", type=int, default=8)
    p.add_argument("--fixed-scheduling", action="store_true")
    p.add_argument("--schedule-switch-prob", type=float, default=0.65)
    p.add_argument("--schedule-reorder-prob", type=float, default=0.50)
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

from temporal_compression import build_compressor, fit_pca_numpy
from temporal_training_data import TemporalTrainingDataGenerator
from ue_memory_manager import DifferentiableUEMemoryManager
from utils import Parameters, E2E_Model, load_weights


class TemporalUEMemoryCGNN(tf.keras.Model):
    """Pretrained CGNN + shared memory reader + selectable compressor."""

    def __init__(self, base_cgnn, d_mem, d_s, compression, **kwargs):
        super().__init__(**kwargs)
        self.base = base_cgnn
        self.d_mem = int(d_mem)
        self.d_s = int(d_s)
        self.compression = str(compression)

        # Common reader for all compression experiments.
        self.mem_in = Dense(d_s, activation="tanh", name="ue_mem_in")
        self.mem_gate = Dense(
            d_s,
            activation="sigmoid",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
            bias_initializer=tf.keras.initializers.Constant(-3.0),
            name="ue_mem_gate",
        )
        self.compressor = build_compressor(
            self.compression, d_s=self.d_s, d_mem=self.d_mem)

    @property
    def memory_variables(self):
        return (
            self.mem_in.trainable_variables
            + self.mem_gate.trainable_variables
            + self.compressor.trainable_variables
        )

    @property
    def temporal_check_variables(self):
        return list(self.compressor.temporal_check_variables)

    def _initial_state(self, y, pe, h_hat, mcs_ue_mask):
        base = self.base
        norm = tf.reduce_mean(tf.square(y), axis=(1, 2, 3), keepdims=True)
        norm = tf.math.divide_no_nan(1.0, tf.sqrt(norm))
        y = y * norm
        if h_hat is not None:
            h_hat = h_hat * tf.expand_dims(norm, axis=1)

        if base._var_mcs_masking:
            s = base._s_init[0]((y, pe, h_hat))
        else:
            s = base._s_init[0]((y, pe, h_hat)) * expand_to_rank(
                tf.gather(mcs_ue_mask, indices=0, axis=2), 5, axis=-1)
            for idx in range(1, base._num_mcss_supported):
                s += base._s_init[idx]((y, pe, h_hat)) * expand_to_rank(
                    tf.gather(mcs_ue_mask, indices=idx, axis=2),
                    5,
                    axis=-1,
                )
        return s

    def _iterations(self, s, pe, active_tx):
        for i in range(self.base._num_it):
            s = self.base._iterations[i]([s, pe, active_tx])
        return s

    def cold_pooled_final(self, inputs):
        """Final K-step pooled state with no temporal memory injection."""
        y, pe, h_hat, active_tx, mcs_ue_mask = inputs
        s = self._initial_state(y, pe, h_hat, mcs_ue_mask)
        s = self._iterations(s, pe, active_tx)
        return tf.reduce_mean(s, axis=[2, 3])

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

        # Common read path. This is held fixed across compressor comparisons.
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

        pooled_final = tf.reduce_mean(s, axis=[2, 3])
        compressed = self.compressor(
            pooled_final,
            safe_memory,
            age_feature,
            memory_valid,
            training=training,
        )

        # A scheduled UE overwrites its identity-owned row with exactly d_mem
        # numbers. An inactive receiver position is not allowed to write.
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


def demap_llr(ofdm, llr_grid, num_tx, mcs_idx=0):
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


def prepare_cgnn_inputs(receiver, y, h_hat, active_tx):
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
    return [y2, pe, h_hat, active, mcs_mask]


def temporal_forward(
    receiver,
    temporal_model,
    y,
    h_hat,
    active_tx,
    memory,
    memory_gap,
    memory_valid,
    training=True,
):
    inputs = prepare_cgnn_inputs(receiver, y, h_hat, active_tx)
    llr_grid, h_ref, next_memory, aux_loss, reconstruction_mse = (
        temporal_model(
            inputs,
            prev_memory=memory,
            memory_gap=memory_gap,
            memory_valid=memory_valid,
            training=training,
        )
    )
    num_tx = tf.shape(active_tx)[1]
    llr = demap_llr(receiver._neural_rx, llr_grid, num_tx, 0)
    return (
        llr,
        tf.cast(h_ref, tf.float32),
        next_memory,
        aux_loss,
        reconstruction_mse,
    )


def build():
    p = Parameters(ARGS.config, training=True, system="nrx")
    e2e = E2E_Model(p, training=True)
    e2e(1, 1.0)
    load_weights(e2e, f"../weights/{p.label}_weights")

    base = e2e._receiver._neural_rx._cgnn
    base.num_it = ARGS.num_it
    model = TemporalUEMemoryCGNN(
        base,
        d_mem=ARGS.d_mem,
        d_s=p.d_s,
        compression=ARGS.compression,
        name=f"temporal_{ARGS.compression}_d{ARGS.d_mem}",
    )
    generator = TemporalTrainingDataGenerator(
        p,
        e2e,
        ue_pool_size=ARGS.ue_pool_size,
        dynamic_scheduling=not ARGS.fixed_scheduling,
        schedule_switch_prob=ARGS.schedule_switch_prob,
        schedule_reorder_prob=ARGS.schedule_reorder_prob,
    )
    memory_manager = DifferentiableUEMemoryManager(
        capacity=ARGS.ue_pool_size,
        d_mem=ARGS.d_mem,
        expiry_slots=ARGS.memory_expiry_slots,
    )
    return p, e2e, model, generator, memory_manager


def fit_pca_from_generator(receiver, model, generator):
    if ARGS.compression != "pca":
        return None

    samples = []
    for _ in range(ARGS.pca_fit_batches):
        ebno = float(np.random.uniform(
            ARGS.min_ebno_db, ARGS.max_ebno_db))
        batch = generator.sample_batch(
            ARGS.pca_fit_batch_size, ARGS.seq_len, ebno)
        for t in range(ARGS.seq_len):
            inputs = prepare_cgnn_inputs(
                receiver,
                batch["y"][:, t],
                batch["ls"][:, t],
                batch["active"][:, t],
            )
            pooled = model.cold_pooled_final(inputs)
            samples.append(tf.reshape(pooled, [-1, model.d_s]).numpy())

    x = np.concatenate(samples, axis=0)
    mean, components, eigenvalues, stats = fit_pca_numpy(
        x, model.d_mem)
    model.compressor.set_basis(mean, components, eigenvalues)
    stats["fit_batches"] = int(ARGS.pca_fit_batches)
    stats["fit_batch_size"] = int(ARGS.pca_fit_batch_size)
    stats["frozen_after_fit"] = True
    return stats


def make_losses(receiver, model, memory_manager, batch):
    batch_size = tf.shape(batch["bits"])[0]
    seq_len = batch["bits"].shape[1]
    if seq_len is None:
        raise ValueError("Sequence length must be statically known")

    state = memory_manager.zero_state(batch_size, tf.float32)
    data_losses = []
    chest_losses = []
    aux_losses = []
    reconstruction_mses = []
    memory_norms = []
    memory_valid_fractions = []
    memory_gap_means = []

    for t in range(seq_len):
        bits_t = batch["bits"][:, t]
        y_t = batch["y"][:, t]
        ls_t = batch["ls"][:, t]
        h_t = batch["h"][:, t]
        active_t = batch["active"][:, t]
        ue_ids_t = batch["ue_ids"][:, t]

        state, prev_memory, memory_gap, memory_valid = memory_manager.gather(
            state, ue_ids_t, t)

        coded_t = receiver._tb_encoders[0](bits_t)
        h_true_t = receiver.preprocess_channel_ground_truth(h_t)

        (
            llr_t,
            h_ref_t,
            updated_memory,
            aux_loss_t,
            reconstruction_mse_t,
        ) = temporal_forward(
            receiver,
            model,
            y_t,
            ls_t,
            active_t,
            prev_memory,
            memory_gap,
            memory_valid,
            training=True,
        )

        state = memory_manager.scatter(
            state, ue_ids_t, updated_memory, active_t, t)

        data_loss_t = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(coded_t, tf.float32), logits=llr_t))
        chest_loss_t = tf.reduce_mean(tf.square(h_ref_t - h_true_t))

        data_losses.append(data_loss_t)
        chest_losses.append(chest_loss_t)
        aux_losses.append(aux_loss_t)
        reconstruction_mses.append(reconstruction_mse_t)
        memory_norms.append(
            tf.reduce_mean(tf.norm(updated_memory, axis=-1)))
        memory_valid_fractions.append(
            tf.reduce_mean(tf.cast(memory_valid, tf.float32)))

        valid_gap = tf.where(
            memory_valid,
            tf.cast(memory_gap, tf.float32),
            tf.zeros_like(tf.cast(memory_gap, tf.float32)),
        )
        denom = tf.maximum(
            tf.reduce_sum(tf.cast(memory_valid, tf.float32)), 1.0)
        memory_gap_means.append(tf.reduce_sum(valid_gap) / denom)

    loss_data = tf.add_n(data_losses) / float(seq_len)
    loss_chest = tf.add_n(chest_losses) / float(seq_len)
    loss_aux = tf.add_n(aux_losses) / float(seq_len)

    aux_weight = (
        ARGS.ae_reconstruction_weight
        if ARGS.compression == "autoencoder"
        else 0.0
    )
    total = (
        loss_data
        + ARGS.chest_weight * loss_chest
        + aux_weight * loss_aux
    )
    diagnostics = {
        "memory_norms": memory_norms,
        "memory_valid_fractions": memory_valid_fractions,
        "memory_gap_means": memory_gap_means,
        "reconstruction_mses": reconstruction_mses,
        "compression_aux_loss": loss_aux,
    }
    return total, loss_data, loss_chest, data_losses, diagnostics


def identity_routing_check(d_mem, capacity):
    """Prove memory ownership is independent of NRX input position."""
    manager = DifferentiableUEMemoryManager(
        capacity=capacity, d_mem=d_mem, expiry_slots=8)
    state = manager.zero_state(1)
    a = tf.ones([d_mem], tf.float32)
    b = tf.ones([d_mem], tf.float32) * 2.0

    ids0 = tf.constant([[0, 1]], tf.int32)
    state = manager.scatter(
        state,
        ids0,
        tf.stack([tf.stack([a, b], axis=0)], axis=0),
        tf.ones([1, 2], tf.float32),
        0,
    )

    if capacity >= 3:
        ids1 = tf.constant([[1, 2]], tf.int32)
        state, gathered, gaps, valid = manager.gather(state, ids1, 1)
        route_ok = (
            np.allclose(gathered.numpy()[0, 0], 2.0)
            and np.allclose(gathered.numpy()[0, 1], 0.0)
            and valid.numpy().tolist() == [[True, False]]
            and gaps.numpy().tolist() == [[1, 0]]
        )
        state = manager.scatter(
            state,
            ids1,
            tf.ones([1, 2, d_mem], tf.float32) * 3.0,
            tf.ones([1, 2], tf.float32),
            1,
        )
        _, gathered2, gaps2, valid2 = manager.gather(
            state, tf.constant([[0, 1]], tf.int32), 2)
        persistence_ok = (
            np.allclose(gathered2.numpy()[0, 0], 1.0)
            and np.allclose(gathered2.numpy()[0, 1], 3.0)
            and valid2.numpy().tolist() == [[True, True]]
            and gaps2.numpy().tolist() == [[2, 1]]
        )
    else:
        ids1 = tf.constant([[1, 0]], tf.int32)
        _, gathered, gaps, valid = manager.gather(state, ids1, 1)
        route_ok = (
            np.allclose(gathered.numpy()[0, 0], 2.0)
            and np.allclose(gathered.numpy()[0, 1], 1.0)
            and valid.numpy().tolist() == [[True, True]]
            and gaps.numpy().tolist() == [[1, 1]]
        )
        persistence_ok = route_ok

    exp = DifferentiableUEMemoryManager(
        capacity=capacity, d_mem=d_mem, expiry_slots=1)
    exp_state = exp.zero_state(1)
    exp_state = exp.scatter(
        exp_state,
        ids0,
        tf.stack([tf.stack([a, b], axis=0)], axis=0),
        tf.ones([1, 2], tf.float32),
        0,
    )
    _, expired_memory, _, expired_valid = exp.gather(
        exp_state, ids0, 2)
    expiration_ok = (
        not np.any(expired_valid.numpy())
        and np.allclose(expired_memory.numpy(), 0.0)
    )
    return {
        "route_across_positions": bool(route_ok),
        "unscheduled_memory_persists": bool(persistence_ok),
        "expiration_zeroes_stale_memory": bool(expiration_ok),
        "passed": bool(route_ok and persistence_ok and expiration_ok),
    }


def temporal_gradient_check(receiver, model, generator, memory_manager):
    """Prove TB2 loss crosses the selected TB1 compression path."""
    batch_size = 2
    one = (
        [[0, 1], [1, 2]]
        if generator.ue_pool_size >= 3
        else [[0, 1], [1, 0]]
    )
    schedule = tf.constant([one] * batch_size, tf.int32)
    batch = generator.sample_batch(
        batch_size, 2, 3.0, ue_ids=schedule)

    check_variables = model.temporal_check_variables
    if not check_variables:
        raise RuntimeError(
            f"No temporal check variables for compression={model.compression}")

    with tf.GradientTape() as tape:
        # Needed for PCA because its basis is deliberately non-trainable.
        for variable in check_variables:
            tape.watch(variable)

        state = memory_manager.zero_state(batch_size)
        state, memory0, gap0, valid0 = memory_manager.gather(
            state, batch["ue_ids"][:, 0], 0)
        _, _, updated0, _, _ = temporal_forward(
            receiver,
            model,
            batch["y"][:, 0],
            batch["ls"][:, 0],
            batch["active"][:, 0],
            memory0,
            gap0,
            valid0,
            training=True,
        )
        state = memory_manager.scatter(
            state,
            batch["ue_ids"][:, 0],
            updated0,
            batch["active"][:, 0],
            0,
        )

        state, memory1, gap1, valid1 = memory_manager.gather(
            state, batch["ue_ids"][:, 1], 1)
        coded1 = receiver._tb_encoders[0](batch["bits"][:, 1])
        llr1, _, _, _, _ = temporal_forward(
            receiver,
            model,
            batch["y"][:, 1],
            batch["ls"][:, 1],
            batch["active"][:, 1],
            memory1,
            gap1,
            valid1,
            training=True,
        )
        future_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(coded1, tf.float32), logits=llr1))

    grads = tape.gradient(future_loss, check_variables)
    valid_grads = [g for g in grads if g is not None]
    norm = (
        tf.linalg.global_norm(valid_grads)
        if valid_grads
        else tf.constant(0.0)
    )
    return {
        "compression": model.compression,
        "tb2_only_loss": float(future_loss.numpy()),
        "compression_path_grad_norm": float(norm.numpy()),
        "tb2_memory_valid": valid1.numpy().tolist(),
        "tb2_memory_gap": gap1.numpy().tolist(),
        "forced_schedule": schedule.numpy().tolist(),
        "passed": bool(float(norm.numpy()) > 0.0),
    }


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        sn.config.seed = seed
    except Exception:
        pass


def schedule_change_fraction(ue_ids):
    ids = ue_ids.numpy()
    if ids.shape[1] < 2:
        return 0.0
    changes = [
        np.mean(np.any(ids[:, t] != ids[:, t + 1], axis=-1))
        for t in range(ids.shape[1] - 1)
    ]
    return float(np.mean(changes))


def main():
    if ARGS.d_mem <= 0:
        raise ValueError("--d-mem must be positive")
    if ARGS.compression == "pca" and ARGS.pca_fit_batches <= 0:
        raise ValueError("--pca-fit-batches must be positive for PCA")

    set_seed(ARGS.seed)
    p, e2e, model, generator, memory_manager = build()
    receiver = e2e._receiver

    pca_stats = fit_pca_from_generator(
        receiver, model, generator)
    if pca_stats is not None:
        print("PCA_FIT=" + json.dumps(pca_stats), flush=True)

    # Build common reader and selected compressor before variable selection.
    warmup = generator.sample_batch(1, ARGS.seq_len, 3.0)
    _ = make_losses(receiver, model, memory_manager, warmup)

    identity_check = identity_routing_check(
        d_mem=ARGS.d_mem, capacity=ARGS.ue_pool_size)
    print(
        "IDENTITY_ROUTING_CHECK=" + json.dumps(identity_check),
        flush=True,
    )
    if not identity_check["passed"]:
        raise RuntimeError(
            "UE identity/memory routing correctness check failed")

    gradient_check = temporal_gradient_check(
        receiver, model, generator, memory_manager)
    print(
        "TEMPORAL_COMPRESSION_GRADIENT_CHECK="
        + json.dumps(gradient_check),
        flush=True,
    )
    if not gradient_check["passed"]:
        raise RuntimeError(
            "TB2 loss did not cross the selected TB1 compression path")

    memory_opt = tf.keras.optimizers.Adam(ARGS.memory_lr)
    joint_opt = tf.keras.optimizers.Adam(ARGS.joint_lr)

    out = Path(ARGS.output_dir or (
        Path.home()
        / "sionna-srsran"
        / "temporal_reuse"
        / "ue_memory"
        / ARGS.compression
    ))
    out.mkdir(parents=True, exist_ok=True)

    history = []
    start = time.time()
    set_seed(ARGS.seed)

    for step in range(ARGS.train_steps):
        ebno = float(np.random.uniform(
            ARGS.min_ebno_db, ARGS.max_ebno_db))
        batch = generator.sample_batch(
            ARGS.batch_size, ARGS.seq_len, ebno)

        with tf.GradientTape() as tape:
            (
                total,
                loss_data,
                loss_chest,
                per_tb,
                diagnostics,
            ) = make_losses(
                receiver, model, memory_manager, batch)

        if step < ARGS.memory_only_steps:
            variables = model.memory_variables
            optimizer = memory_opt
            phase = "memory_only"
        else:
            variables = model.trainable_variables
            optimizer = joint_opt
            phase = "joint"

        grads = tape.gradient(total, variables)
        pairs = [
            (g, v)
            for g, v in zip(grads, variables)
            if g is not None
        ]
        if not pairs:
            raise RuntimeError(
                "No gradients reached the selected variables.")
        optimizer.apply_gradients(pairs)
        grad_norm = float(
            tf.linalg.global_norm([g for g, _ in pairs]).numpy())

        if step % ARGS.log_every == 0 or step == ARGS.train_steps - 1:
            row = {
                "step": step,
                "phase": phase,
                "compression": ARGS.compression,
                "ebno_db": ebno,
                "loss": float(total.numpy()),
                "loss_data": float(loss_data.numpy()),
                "loss_chest": float(loss_chest.numpy()),
                "compression_aux_loss": float(
                    diagnostics["compression_aux_loss"].numpy()),
                "reconstruction_mse_per_tb": [
                    float(x.numpy())
                    for x in diagnostics["reconstruction_mses"]
                ],
                "loss_per_tb": [
                    float(x.numpy()) for x in per_tb
                ],
                "memory_norm_per_tb": [
                    float(x.numpy())
                    for x in diagnostics["memory_norms"]
                ],
                "memory_valid_fraction_per_tb": [
                    float(x.numpy())
                    for x in diagnostics["memory_valid_fractions"]
                ],
                "memory_gap_mean_per_tb": [
                    float(x.numpy())
                    for x in diagnostics["memory_gap_means"]
                ],
                "schedule_change_fraction": schedule_change_fraction(
                    batch["ue_ids"]),
                "schedule_example": (
                    batch["ue_ids"][0].numpy().tolist()
                ),
                "gradient_norm": grad_norm,
                "seconds": time.time() - start,
            }
            history.append(row)
            print("TRAIN=" + json.dumps(row), flush=True)

    checkpoint = out / (
        f"ue_memory_{ARGS.compression}_idaware_"
        f"d{ARGS.d_mem}_k{ARGS.num_it}.weights.h5"
    )
    model.save_weights(str(checkpoint))

    summary = {
        "architecture": "ue_identity_aware_temporal_memory_v3_compression",
        "compression": ARGS.compression,
        "compression_semantics": {
            "writer": "task-aware learned recurrent writer",
            "pca": "frozen PCA coefficients of pooled final NRX state",
            "autoencoder": (
                "learned encoder bottleneck; decoder used for "
                "reconstruction loss only"
            ),
        }[ARGS.compression],
        "config": ARGS.config,
        "d_mem": ARGS.d_mem,
        "memory_dtype": "float32",
        "memory_cap_bits_per_ue": int(ARGS.d_mem * 32),
        "memory_cap_bytes_per_ue": int(ARGS.d_mem * 4),
        "num_it": ARGS.num_it,
        "train_steps": ARGS.train_steps,
        "memory_only_steps": ARGS.memory_only_steps,
        "batch_size": ARGS.batch_size,
        "seq_len": ARGS.seq_len,
        "ue_pool_size": ARGS.ue_pool_size,
        "dynamic_scheduling": not ARGS.fixed_scheduling,
        "memory_expiry_slots": ARGS.memory_expiry_slots,
        "schedule_switch_prob": ARGS.schedule_switch_prob,
        "schedule_reorder_prob": ARGS.schedule_reorder_prob,
        "ae_reconstruction_weight": (
            ARGS.ae_reconstruction_weight
            if ARGS.compression == "autoencoder"
            else 0.0
        ),
        "pca_fit": pca_stats,
        "identity_routing_check": identity_check,
        "temporal_compression_gradient_check": gradient_check,
        "checkpoint": str(checkpoint),
        "history": history,
    }
    (out / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(
        "TRAINING_SUMMARY="
        + json.dumps(summary, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
