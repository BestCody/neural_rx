#!/usr/bin/env python3
"""Train/evaluate temporal pairwise edge memory on the NRX.

This is an experiment wrapper: the shipped NRX source is not modified. The
wrapper reuses the pretrained StateInit, aggregation MLP, state-update CNNs,
and readout layers, but replaces average multi-user aggregation with a gated
pairwise message whose recurrent pair state has dimension d_e.

Run from neural_rx/scripts so the existing relative weight paths remain valid.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d-edge", type=int, required=True, choices=[1, 4, 8, 16])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--train-steps", type=int, default=6000)
    p.add_argument("--edge-only-steps", type=int, default=1500)
    p.add_argument("--train-batch", type=int, default=8)
    p.add_argument("--train-seq-len", type=int, default=4)
    p.add_argument("--num-it", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260811)
    p.add_argument("--study-dir", type=str,
                   default="/home/h3lou/sionna-srsran/temporal_reuse")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--eval-target-errors", type=int, default=120)
    p.add_argument("--eval-max-batches", type=int, default=24)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
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

STUDY = Path(ARGS.study_dir)
OUT = Path(ARGS.output_dir or (STUDY / "edge_memory"))
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(STUDY))
sys.path.insert(0, "..")

from temporal_pipeline import TemporalSequence, ebno_to_no, NUM_TX, CONFIG
from utils import Parameters, E2E_Model, load_weights


class TemporalEdgeCGNN(tf.keras.Model):
    """Reuse a pretrained CGNN while adding recurrent UE-pair edge state."""

    def __init__(self, base_cgnn, d_edge, d_s, **kwargs):
        super().__init__(**kwargs)
        self.base = base_cgnn
        self.d_edge = int(d_edge)
        self.d_s = int(d_s)

        # Shared across CGNN iterations and TBs. Current pooled node state +
        # previous pair memory -> gated updated pair memory.
        self.edge_hidden = Dense(64, activation="relu", name="edge_hidden")
        self.edge_candidate = Dense(d_edge, activation="tanh",
                                    name="edge_candidate")
        self.edge_keep = Dense(d_edge, activation="sigmoid",
                               bias_initializer="zeros", name="edge_keep")

        # The message gate starts almost as the original average aggregator.
        # A tiny kernel keeps gradients into edge state non-zero from step 1.
        self.msg_hidden = Dense(64, activation="relu", name="msg_hidden")
        self.msg_gate = Dense(
            d_s,
            activation="sigmoid",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
            bias_initializer=tf.keras.initializers.Constant(4.0),
            name="msg_gate")

    @property
    def edge_variables(self):
        layers = [self.edge_hidden, self.edge_candidate, self.edge_keep,
                  self.msg_hidden, self.msg_gate]
        return [v for layer in layers for v in layer.trainable_variables]

    def zero_edge(self, batch_size, num_tx, dtype=tf.float32):
        return tf.zeros([batch_size, num_tx, num_tx, self.d_edge], dtype=dtype)

    def _pair_mask(self, active_tx, dtype):
        # Ordered receiver/sender pairs [B,U,U], excluding self edges.
        active = tf.cast(active_tx, dtype)
        pair = active[:, :, None] * active[:, None, :]
        u = tf.shape(active)[1]
        pair *= (1.0 - tf.eye(u, batch_shape=[tf.shape(active)[0]], dtype=dtype))
        return pair

    def _aggregate(self, s, active_tx, prev_edge, original_agg):
        """Pair-aware replacement for AggregateUserStates."""
        dtype = s.dtype
        b = tf.shape(s)[0]
        u = tf.shape(s)[1]

        # Original sender-state transform: reuse pretrained aggregation MLP.
        sp = s
        for layer in original_agg._hidden_layers:
            sp = layer(sp)
        sp = original_agg._output_layer(sp)

        # Compact current-TB node summaries for pair-state update.
        pooled = tf.reduce_mean(s, axis=[2, 3])  # [B,U,d_s]
        recv = tf.broadcast_to(pooled[:, :, None, :], [b, u, u, self.d_s])
        send = tf.broadcast_to(pooled[:, None, :, :], [b, u, u, self.d_s])

        if prev_edge is None:
            prev_edge = self.zero_edge(b, u, dtype)
        else:
            prev_edge = tf.cast(prev_edge, dtype)

        pair_in = tf.concat([recv, send, prev_edge], axis=-1)
        eh = self.edge_hidden(pair_in)
        cand = self.edge_candidate(eh)
        keep = self.edge_keep(eh)
        edge = keep * prev_edge + (1.0 - keep) * cand

        mask = self._pair_mask(active_tx, dtype)
        edge = edge * mask[..., None]

        # Richer d_edge can preserve more temporal pair information; all
        # variants ultimately modulate the same d_s-dimensional messages.
        mh = self.msg_hidden(tf.concat([recv, send, edge], axis=-1))
        gate = self.msg_gate(mh) * mask[..., None]  # [B,U,U,d_s]

        # sender dimension becomes pair dimension: receiver u receives v.
        sender_sp = tf.broadcast_to(
            sp[:, None, :, :, :, :],
            [b, u, u, tf.shape(sp)[2], tf.shape(sp)[3], self.d_s])
        msg = sender_sp * gate[..., None, None, :]
        agg = tf.reduce_sum(msg, axis=2)

        # Match the original mean-over-other-active-users scaling. Do not
        # normalize by gate magnitude: with two UEs that would erase the only
        # pairwise modulation available.
        count = tf.reduce_sum(mask, axis=2)
        count = tf.maximum(count, 1.0)
        agg = agg / count[..., None, None, None]
        return agg, edge

    def call(self, inputs, prev_edge=None):
        y, pe, h_hat, active_tx, mcs_ue_mask = inputs
        base = self.base

        # Exact normalization/state initialization from CGNN.call.
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
                    tf.gather(mcs_ue_mask, indices=idx, axis=2), 5, axis=-1)

        edge = prev_edge
        llrs = []
        h_hats = []
        for i in range(base._num_it):
            it = base._iterations[i]
            a, edge = self._aggregate(s, active_tx, edge, it._state_aggreg)
            s = it._state_update((s, a, pe))

            if i == base._num_it - 1:
                llrs_i = []
                for idx in range(base._num_mcss_supported):
                    if base._var_mcs_masking:
                        z = base._readout_llrs[0](s)
                        z = tf.gather(z,
                                      indices=tf.range(base._num_bits_per_symbol[idx]),
                                      axis=-1)
                    else:
                        z = base._readout_llrs[idx](s)
                    llrs_i.append(z)
                llrs.append(llrs_i)
                h_hats.append(base._readout_chest(s))

        return llrs, h_hats, edge


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


def temporal_forward(receiver, model, y, h_hat, active_tx, edge_prev):
    """CGNNOFDM preprocessing + temporal CGNN + demapping."""
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

    llrs, h_hats, edge = model([y2, pe, h_hat, active, mcs_mask], edge_prev)
    llr = demap_llr(ofdm, llrs[-1][0], num_tx, 0)
    return llr, tf.cast(h_hats[-1], tf.float32), edge


def build_model(training, d_edge, num_it):
    p = Parameters(CONFIG, training=training,
                   **({} if training else {"num_tx_eval": NUM_TX}),
                   system="nrx")
    e2e = E2E_Model(p, training=training,
                    **({} if training else {"mcs_arr_eval_idx": 0}))
    e2e(1, 1.0)
    load_weights(e2e, f"../weights/{p.label}_weights")
    base = e2e._receiver._neural_rx._cgnn
    base.num_it = num_it
    model = TemporalEdgeCGNN(base, d_edge=d_edge, d_s=p.d_s,
                             name=f"temporal_edge_d{d_edge}")
    return p, e2e, model


def make_tb(seq, p, h_slot, no):
    bsz = tf.shape(h_slot)[0]
    b = seq._source([bsz, p.max_num_tx, seq._tx._tb_size])
    x = seq._tx(b)
    y = seq._apply([x, h_slot, no])
    ls = seq._rx.estimate_channel(y, p.max_num_tx)
    active = tf.ones([bsz, p.max_num_tx], tf.float32)
    coded = seq._rx._tb_encoders[0](b)
    h_true = seq._rx.preprocess_channel_ground_truth(h_slot)
    return b, coded, y, ls, active, h_true


def set_data_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        sn.config.seed = seed
    except Exception:
        pass


def train_variant():
    p, e2e, model = build_model(True, ARGS.d_edge, ARGS.num_it)
    seq = TemporalSequence(p, e2e)

    # Build edge variables before selecting trainable-variable groups.
    h0 = seq.sample_sequence_channel(ARGS.train_batch, 1)[0]
    no0 = ebno_to_no(p, seq._tx, 3.0)
    _, _, y0, ls0, a0, _ = make_tb(seq, p, h0, no0)
    _ = temporal_forward(seq._rx, model, y0, ls0, a0, None)

    edge_opt = tf.keras.optimizers.Adam(1e-3)
    joint_opt = tf.keras.optimizers.Adam(2e-5)
    history = []

    # Reset data RNG after model/layer initialization so every d_edge run sees
    # the same stochastic sequence stream as closely as possible.
    set_data_seed(ARGS.seed)

    t_start = time.time()
    for step in range(ARGS.train_steps):
        ebno = float(np.random.uniform(1.0, 5.0))
        no = ebno_to_no(p, seq._tx, ebno)
        h_seq = seq.sample_sequence_channel(ARGS.train_batch,
                                            ARGS.train_seq_len)
        with tf.GradientTape() as tape:
            edge = None
            losses = []
            chest_losses = []
            for t in range(ARGS.train_seq_len):
                _, coded, y, ls, active, h_true = make_tb(
                    seq, p, h_seq[t], no)
                llr, h_ref, edge = temporal_forward(
                    seq._rx, model, y, ls, active, edge)
                data_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.cast(coded, tf.float32), logits=llr))
                chest_loss = tf.reduce_mean(tf.square(h_ref - h_true))
                if t > 0:  # optimize the positions that can use prior memory
                    losses.append(data_loss)
                    chest_losses.append(chest_loss)
            loss_data = tf.add_n(losses) / len(losses)
            loss_ch = tf.add_n(chest_losses) / len(chest_losses)
            loss = loss_data + 0.01 * loss_ch

        if step < ARGS.edge_only_steps:
            variables = model.edge_variables
            opt = edge_opt
            phase = "edge_only"
        else:
            variables = model.trainable_variables
            opt = joint_opt
            phase = "joint"
        grads = tape.gradient(loss, variables)
        pairs = [(g, v) for g, v in zip(grads, variables) if g is not None]
        opt.apply_gradients(pairs)

        if step % 100 == 0 or step == ARGS.train_steps - 1:
            row = {
                "step": step,
                "phase": phase,
                "ebno_db": ebno,
                "loss": float(loss.numpy()),
                "loss_data": float(loss_data.numpy()),
                "loss_chest": float(loss_ch.numpy()),
                "seconds": time.time() - t_start,
            }
            history.append(row)
            print("TRAIN", json.dumps(row), flush=True)

    ckpt = str(OUT / f"dedge_{ARGS.d_edge}_k{ARGS.num_it}.weights.h5")
    model.save_weights(ckpt)
    meta = {
        "d_edge": ARGS.d_edge,
        "num_it": ARGS.num_it,
        "train_steps": ARGS.train_steps,
        "edge_only_steps": ARGS.edge_only_steps,
        "train_batch": ARGS.train_batch,
        "train_seq_len": ARGS.train_seq_len,
        "seed": ARGS.seed,
        "checkpoint": ckpt,
        "edge_parameters": int(sum(np.prod(v.shape) for v in model.edge_variables)),
        "history": history,
        "train_seconds": time.time() - t_start,
    }
    with open(OUT / f"train_dedge_{ARGS.d_edge}.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def crossing(res, target=0.1):
    pts = sorted((float(k), v) for k, v in res.items())
    for (x0, r0), (x1, r1) in zip(pts, pts[1:]):
        y0, y1 = r0["tbler"], r1["tbler"]
        if y0 >= target > y1 and y0 > 0 and y1 > 0:
            l0, l1 = np.log10(y0), np.log10(y1)
            return float(x0 + (np.log10(target) - l0) *
                         (x1 - x0) / (l1 - l0))
    return None


def eval_mode(seq, p, model, mode):
    res = {}
    ebnos = np.arange(1.0, 4.01, 0.5)
    for ebno in ebnos:
        no = ebno_to_no(p, seq._tx, float(ebno))
        errors = 0
        blocks = 0
        per_user_err = np.zeros(NUM_TX, dtype=np.int64)
        per_user_blocks = 0
        start = time.time()
        for _ in range(ARGS.eval_max_batches):
            h_seq = seq.sample_sequence_channel(ARGS.eval_batch, 8)
            edge = None
            for t in range(8):
                b, _, y, ls, active, _ = make_tb(seq, p, h_seq[t], no)
                use_edge = edge
                if mode == "reset":
                    use_edge = None
                elif mode == "shuffle" and edge is not None:
                    use_edge = tf.roll(edge, shift=1, axis=0)
                llr, _, edge_new = temporal_forward(
                    seq._rx, model, y, ls, active, use_edge)
                edge = edge_new
                b_hat, _ = seq._rx._tb_decoders[0](llr)
                if t > 0:
                    err = tf.reduce_any(
                        tf.not_equal(b, tf.cast(b_hat > 0.5, b.dtype)),
                        axis=-1).numpy()
                    errors += int(err.sum())
                    blocks += err.size
                    per_user_err += err.sum(axis=0)
                    per_user_blocks += err.shape[0]
            if errors >= ARGS.eval_target_errors:
                break
        tbler = errors / max(blocks, 1)
        row = {
            "tbler": tbler,
            "blocks": blocks,
            "errors": errors,
            "tbler_per_user": (per_user_err / max(per_user_blocks, 1)).tolist(),
            "seconds": time.time() - start,
        }
        res[str(float(ebno))] = row
        print(f"EVAL dE={ARGS.d_edge} mode={mode} ebno={ebno:.1f} "
              f"tbler={tbler:.5f} ({errors}/{blocks})", flush=True)
        if tbler < 0.015:
            break
    return res


def latency_measure(seq, p, model):
    no = ebno_to_no(p, seq._tx, 3.0)
    h_seq = seq.sample_sequence_channel(1, 2)
    _, _, y0, ls0, a0, _ = make_tb(seq, p, h_seq[0], no)
    _, _, edge = temporal_forward(seq._rx, model, y0, ls0, a0, None)
    _, _, y, ls, active, _ = make_tb(seq, p, h_seq[1], no)

    @tf.function(jit_compile=False)
    def infer(y_, ls_, active_, edge_):
        llr_, _, edge_new_ = temporal_forward(
            seq._rx, model, y_, ls_, active_, edge_)
        return llr_, edge_new_

    for _ in range(20):
        z, e = infer(y, ls, active, edge)
        _ = z.numpy(); _ = e.numpy()

    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        z, e = infer(y, ls, active, edge)
        _ = z.numpy(); _ = e.numpy()  # GPU synchronization
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return {
        "median_ms": float(np.median(arr)),
        "p10_ms": float(np.percentile(arr, 10)),
        "p90_ms": float(np.percentile(arr, 90)),
        "mean_ms": float(np.mean(arr)),
        "samples_ms": arr.tolist(),
    }


def evaluate_variant():
    p, e2e, model = build_model(False, ARGS.d_edge, ARGS.num_it)
    seq = TemporalSequence(p, e2e)

    # Build variables, then restore the temporal fine-tuned checkpoint.
    h0 = seq.sample_sequence_channel(1, 1)[0]
    no0 = ebno_to_no(p, seq._tx, 3.0)
    _, _, y0, ls0, a0, _ = make_tb(seq, p, h0, no0)
    _ = temporal_forward(seq._rx, model, y0, ls0, a0, None)
    ckpt = str(OUT / f"dedge_{ARGS.d_edge}_k{ARGS.num_it}.weights.h5")
    model.load_weights(ckpt)

    set_data_seed(ARGS.seed + 1000)
    normal = eval_mode(seq, p, model, "normal")
    set_data_seed(ARGS.seed + 1000)
    reset = eval_mode(seq, p, model, "reset")

    # Distribution-preserving negative control at 3 dB only.
    old_max = ARGS.eval_max_batches
    shuffle = {}
    no = ebno_to_no(p, seq._tx, 3.0)
    errors = blocks = 0
    set_data_seed(ARGS.seed + 2000)
    start = time.time()
    for _ in range(min(old_max, 12)):
        h_seq = seq.sample_sequence_channel(ARGS.eval_batch, 8)
        edge = None
        for t in range(8):
            b, _, y, ls, active, _ = make_tb(seq, p, h_seq[t], no)
            use = None if edge is None else tf.roll(edge, shift=1, axis=0)
            llr, _, edge = temporal_forward(seq._rx, model, y, ls, active, use)
            b_hat, _ = seq._rx._tb_decoders[0](llr)
            if t > 0:
                err = tf.reduce_any(
                    tf.not_equal(b, tf.cast(b_hat > 0.5, b.dtype)), axis=-1)
                errors += int(tf.reduce_sum(tf.cast(err, tf.int32)).numpy())
                blocks += int(np.prod(err.shape))
    shuffle["3.0"] = {
        "tbler": errors / max(blocks, 1), "errors": errors,
        "blocks": blocks, "seconds": time.time() - start,
    }

    lat = latency_measure(seq, p, model)
    persistent_bytes_per_sequence = NUM_TX * (NUM_TX - 1) * ARGS.d_edge * 4
    edge_params = int(sum(np.prod(v.shape) for v in model.edge_variables))
    out = {
        "d_edge": ARGS.d_edge,
        "num_it": ARGS.num_it,
        "normal": {
            "tbler_vs_ebno": normal,
            "snr_at_10pct_tbler_db": crossing(normal),
        },
        "reset_each_tb": {
            "tbler_vs_ebno": reset,
            "snr_at_10pct_tbler_db": crossing(reset),
        },
        "shuffle_previous_edge_at_3db": shuffle["3.0"],
        "latency": lat,
        "edge_parameters": edge_params,
        "persistent_edge_bytes_per_sequence": persistent_bytes_per_sequence,
        "persistent_edge_scaling": "U*(U-1)*d_edge*4 bytes for float32 ordered edges",
    }
    with open(OUT / f"eval_dedge_{ARGS.d_edge}.json", "w") as f:
        json.dump(out, f, indent=2)
    print("RESULT", json.dumps({
        "d_edge": ARGS.d_edge,
        "normal_crossing": out["normal"]["snr_at_10pct_tbler_db"],
        "reset_crossing": out["reset_each_tb"]["snr_at_10pct_tbler_db"],
        "shuffle_3db_tbler": out["shuffle_previous_edge_at_3db"]["tbler"],
        "median_latency_ms": lat["median_ms"],
        "edge_parameters": edge_params,
        "persistent_bytes": persistent_bytes_per_sequence,
    }), flush=True)
    return out


def main():
    print(json.dumps({
        "gpu": ARGS.gpu,
        "d_edge": ARGS.d_edge,
        "tensorflow": tf.__version__,
        "sionna": getattr(sn, "__version__", "unknown"),
        "visible_gpus": [x.name for x in tf.config.list_physical_devices("GPU")],
    }), flush=True)
    if not ARGS.skip_train:
        train_variant()
    if not ARGS.skip_eval:
        evaluate_variant()


if __name__ == "__main__":
    main()
