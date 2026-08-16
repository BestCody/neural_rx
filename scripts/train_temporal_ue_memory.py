#!/usr/bin/env python3
"""Train Neural RX over correlated TBs with UE-identity-aware temporal memory.

Architecture:
  current TB -> shipped NRX initialization
             + memory gathered by stable physical UE ID
             + scheduling-gap / memory-validity context
             -> K NRX iterations
             -> decode + per-UE memory writer
             -> scatter updated memories back by the same UE IDs

The neural model learns *what* to remember.  The external memory manager owns
*whose* memory it is.  During training, differentiable gather/scatter routing
keeps the computation graph connected across TBs even when UEs enter, leave, or
change input position.
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

    # New UE-aware architecture.
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

from temporal_training_data import TemporalTrainingDataGenerator
from ue_memory_manager import DifferentiableUEMemoryManager
from utils import Parameters, E2E_Model, load_weights


class TemporalUEMemoryCGNN(tf.keras.Model):
    """Pretrained CGNN plus a compact recurrent memory vector for each UE."""

    def __init__(self, base_cgnn, d_mem, d_s, **kwargs):
        super().__init__(**kwargs)
        self.base = base_cgnn
        self.d_mem = int(d_mem)
        self.d_s = int(d_s)

        # Read old memory into the current-TB state. The gate starts nearly shut
        # so the untrained path stays close to the pretrained cold receiver.
        self.mem_in = Dense(d_s, activation="tanh", name="ue_mem_in")
        self.mem_gate = Dense(
            d_s,
            activation="sigmoid",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
            bias_initializer=tf.keras.initializers.Constant(-3.0),
            name="ue_mem_gate",
        )

        # Write the final current-TB state into a compact UE memory.
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

    def call(
        self,
        inputs,
        prev_memory=None,
        memory_gap=None,
        memory_valid=None,
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
        # A compact monotonic age feature: 1,2,4,... slots do not explode the
        # gate input numerically. New/invalid memory has age feature 0.
        age_feature = tf.math.log1p(
            tf.cast(tf.maximum(memory_gap, 0), s.dtype))
        age_feature *= valid_f

        # Read path: the gate sees fresh state, old memory, whether that memory
        # exists, and how many slots ago the UE was last scheduled.
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

        # Shipped NRX iteration blocks. K=2 is the main experiment.
        for i in range(base._num_it):
            s = base._iterations[i]([s, pe, active_tx])

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

        # Write path. If no old memory exists, force keep=0 so the first memory
        # becomes a fresh candidate rather than mixing with an invalid zero row.
        pooled_final = tf.reduce_mean(s, axis=[2, 3])
        write_in = tf.concat(
            [
                pooled_final,
                safe_memory,
                age_feature[..., None],
                valid_f[..., None],
            ],
            axis=-1,
        )
        z = self.mem_hidden(write_in)
        candidate = self.mem_candidate(z)
        keep = self.mem_keep(z) * valid_f[..., None]
        next_memory = keep * safe_memory + (1.0 - keep) * candidate

        # Inactive receiver positions never overwrite their historical memory.
        active_bool = tf.cast(active_tx, tf.bool)
        next_memory = tf.where(
            active_bool[..., None], next_memory, safe_memory)

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


def temporal_forward(
    receiver,
    temporal_model,
    y,
    h_hat,
    active_tx,
    memory,
    memory_gap,
    memory_valid,
):
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
        [y2, pe, h_hat, active, mcs_mask],
        prev_memory=memory,
        memory_gap=memory_gap,
        memory_valid=memory_valid,
    )
    llr = demap_llr(ofdm, llr_grid, num_tx, 0)
    return llr, tf.cast(h_ref, tf.float32), next_memory


def build():
    # Keep the shipped NRX transmitter, receiver, preprocessing, and weights.
    p = Parameters(ARGS.config, training=True, system="nrx")
    e2e = E2E_Model(p, training=True)
    e2e(1, 1.0)
    load_weights(e2e, f"../weights/{p.label}_weights")

    base = e2e._receiver._neural_rx._cgnn
    base.num_it = ARGS.num_it
    temporal_model = TemporalUEMemoryCGNN(
        base,
        d_mem=ARGS.d_mem,
        d_s=p.d_s,
        name=f"temporal_ue_memory_d{ARGS.d_mem}",
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
    return p, e2e, temporal_model, generator, memory_manager


def make_losses(receiver, temporal_model, memory_manager, batch):
    """One connected forward pass over every TB using stable UE identity."""
    batch_size = tf.shape(batch["bits"])[0]
    seq_len = batch["bits"].shape[1]
    if seq_len is None:
        raise ValueError("Sequence length must be statically known")

    state = memory_manager.zero_state(batch_size, tf.float32)

    data_losses = []
    chest_losses = []
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

        # Resolve ownership before the neural receiver sees the memory.
        state, prev_memory, memory_gap, memory_valid = memory_manager.gather(
            state, ue_ids_t, t)

        coded_t = receiver._tb_encoders[0](bits_t)
        h_true_t = receiver.preprocess_channel_ground_truth(h_t)

        llr_t, h_ref_t, updated_memory = temporal_forward(
            receiver,
            temporal_model,
            y_t,
            ls_t,
            active_t,
            prev_memory,
            memory_gap,
            memory_valid,
        )

        # Write back under stable physical IDs, not input positions.
        state = memory_manager.scatter(
            state,
            ue_ids_t,
            updated_memory,
            active_t,
            t,
        )

        data_loss_t = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(coded_t, tf.float32), logits=llr_t))
        chest_loss_t = tf.reduce_mean(tf.square(h_ref_t - h_true_t))

        data_losses.append(data_loss_t)
        chest_losses.append(chest_loss_t)
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
    total = loss_data + ARGS.chest_weight * loss_chest
    diagnostics = {
        "memory_norms": memory_norms,
        "memory_valid_fractions": memory_valid_fractions,
        "memory_gap_means": memory_gap_means,
    }
    return total, loss_data, loss_chest, data_losses, diagnostics


def identity_routing_check(d_mem, capacity):
    """Prove memory follows physical UE IDs when positions/users change."""
    manager = DifferentiableUEMemoryManager(
        capacity=capacity, d_mem=d_mem, expiry_slots=8)
    state = manager.zero_state(1)

    a = tf.ones([d_mem], tf.float32) * 1.0
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
        # UE B moves from position 1 -> 0; UE C is new at position 1.
        ids1 = tf.constant([[1, 2]], tf.int32)
        state, gathered, gaps, valid = manager.gather(state, ids1, 1)
        route_ok = (
            np.allclose(gathered.numpy()[0, 0], 2.0)
            and np.allclose(gathered.numpy()[0, 1], 0.0)
            and valid.numpy().tolist() == [[True, False]]
            and gaps.numpy().tolist() == [[1, 0]]
        )

        b_new = tf.ones([d_mem], tf.float32) * 20.0
        c_new = tf.ones([d_mem], tf.float32) * 30.0
        state = manager.scatter(
            state,
            ids1,
            tf.stack([tf.stack([b_new, c_new], axis=0)], axis=0),
            tf.ones([1, 2], tf.float32),
            1,
        )
        ids2 = tf.constant([[0, 1]], tf.int32)
        _, gathered2, gaps2, valid2 = manager.gather(state, ids2, 2)
        persistence_ok = (
            np.allclose(gathered2.numpy()[0, 0], 1.0)
            and np.allclose(gathered2.numpy()[0, 1], 20.0)
            and valid2.numpy().tolist() == [[True, True]]
            and gaps2.numpy().tolist() == [[2, 1]]
        )
    else:
        # No spare UE exists; position swap alone must preserve identity.
        ids1 = tf.constant([[1, 0]], tf.int32)
        _, gathered, gaps, valid = manager.gather(state, ids1, 1)
        route_ok = (
            np.allclose(gathered.numpy()[0, 0], 2.0)
            and np.allclose(gathered.numpy()[0, 1], 1.0)
            and valid.numpy().tolist() == [[True, True]]
            and gaps.numpy().tolist() == [[1, 1]]
        )
        persistence_ok = route_ok

    # Separate expiration check: last seen at 0, queried at 2 with TTL 1.
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
    _, expired_memory, _, expired_valid = exp.gather(exp_state, ids0, 2)
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


def temporal_gradient_check(receiver, temporal_model, generator, memory_manager):
    """Verify TB2 loss reaches TB1's writer after identity remapping."""
    batch_size = 2

    if generator.ue_pool_size >= 3:
        one = [[0, 1], [1, 2]]
    else:
        one = [[0, 1], [1, 0]]
    schedule = tf.constant([one] * batch_size, tf.int32)
    batch = generator.sample_batch(
        batch_size, 2, 3.0, ue_ids=schedule)

    with tf.GradientTape() as tape:
        state = memory_manager.zero_state(batch_size)

        # TB1: both memories are new.
        state, memory0, gap0, valid0 = memory_manager.gather(
            state, batch["ue_ids"][:, 0], 0)
        _, _, updated0 = temporal_forward(
            receiver,
            temporal_model,
            batch["y"][:, 0],
            batch["ls"][:, 0],
            batch["active"][:, 0],
            memory0,
            gap0,
            valid0,
        )
        state = memory_manager.scatter(
            state,
            batch["ue_ids"][:, 0],
            updated0,
            batch["active"][:, 0],
            0,
        )

        # TB2: UE 1 is deliberately moved to another input position.
        state, memory1, gap1, valid1 = memory_manager.gather(
            state, batch["ue_ids"][:, 1], 1)
        coded1 = receiver._tb_encoders[0](batch["bits"][:, 1])
        llr1, _, _ = temporal_forward(
            receiver,
            temporal_model,
            batch["y"][:, 1],
            batch["ls"][:, 1],
            batch["active"][:, 1],
            memory1,
            gap1,
            valid1,
        )
        future_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(coded1, tf.float32), logits=llr1))

    grads = tape.gradient(
        future_loss, temporal_model.memory_writer_variables)
    valid_grads = [g for g in grads if g is not None]
    norm = (
        tf.linalg.global_norm(valid_grads)
        if valid_grads
        else tf.constant(0.0)
    )
    return {
        "tb2_only_loss": float(future_loss.numpy()),
        "memory_writer_grad_norm": float(norm.numpy()),
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
    set_seed(ARGS.seed)
    p, e2e, model, generator, memory_manager = build()
    receiver = e2e._receiver

    # Build the memory layers once before selecting variable groups.
    warmup = generator.sample_batch(1, ARGS.seq_len, 3.0)
    _ = make_losses(receiver, model, memory_manager, warmup)

    identity_check = identity_routing_check(
        d_mem=ARGS.d_mem, capacity=ARGS.ue_pool_size)
    print(
        "IDENTITY_ROUTING_CHECK=" + json.dumps(identity_check),
        flush=True,
    )
    if not identity_check["passed"]:
        raise RuntimeError("UE identity/memory routing correctness check failed")

    gradient_check = temporal_gradient_check(
        receiver, model, generator, memory_manager)
    print(
        "TEMPORAL_GRADIENT_CHECK=" + json.dumps(gradient_check),
        flush=True,
    )
    if not gradient_check["passed"]:
        raise RuntimeError(
            "Later-TB loss did not reach the earlier memory writer through "
            "UE-ID gather/scatter routing."
        )

    memory_opt = tf.keras.optimizers.Adam(ARGS.memory_lr)
    joint_opt = tf.keras.optimizers.Adam(ARGS.joint_lr)

    out = Path(ARGS.output_dir or (
        Path.home() / "sionna-srsran" / "temporal_reuse" / "ue_memory"))
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
            ) = make_losses(receiver, model, memory_manager, batch)

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
        grad_norm = float(
            tf.linalg.global_norm([g for g, _ in pairs]).numpy())

        if step % ARGS.log_every == 0 or step == ARGS.train_steps - 1:
            row = {
                "step": step,
                "phase": phase,
                "ebno_db": ebno,
                "loss": float(total.numpy()),
                "loss_data": float(loss_data.numpy()),
                "loss_chest": float(loss_chest.numpy()),
                "loss_per_tb": [float(x.numpy()) for x in per_tb],
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
                "schedule_example": batch["ue_ids"][0].numpy().tolist(),
                "gradient_norm": grad_norm,
                "seconds": time.time() - start,
            }
            history.append(row)
            print("TRAIN=" + json.dumps(row), flush=True)

    checkpoint = out / (
        f"ue_memory_idaware_d{ARGS.d_mem}_k{ARGS.num_it}.weights.h5")
    model.save_weights(str(checkpoint))

    summary = {
        "architecture": "ue_identity_aware_temporal_memory_v2",
        "config": ARGS.config,
        "d_mem": ARGS.d_mem,
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
        "identity_routing_check": identity_check,
        "temporal_gradient_check": gradient_check,
        "checkpoint": str(checkpoint),
        "history": history,
    }
    (out / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(
        "TRAINING_SUMMARY=" + json.dumps(summary, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
