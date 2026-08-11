#!/usr/bin/env python3
"""Compatibility runner for the current-TB edge ablation.

The existing GitHub Actions edge sweep invokes this filename with dE values.
For this ablation only:
  dE=1 -> dynamic recomputation every CGNN iteration (previous round edge zeroed)
  dE=8 -> recurrent refinement across CGNN iterations, reset at each TB boundary
  dE=4/16 -> skipped
Both evaluated modes load the same already-trained dE=1, K=2 checkpoint.
No cross-TB edge state is used.
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
    p.add_argument("--study-dir", default="/home/h3lou/sionna-srsran/temporal_reuse")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--eval-target-errors", type=int, default=120)
    p.add_argument("--eval-max-batches", type=int, default=24)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()

ARGS = parse_args()
MODE = {1: "dynamic", 8: "recurrent"}.get(ARGS.d_edge)
if MODE is None:
    print(json.dumps({"d_edge": ARGS.d_edge, "status": "skipped_for_intra_tb_ablation"}), flush=True)
    raise SystemExit(0)

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


class IntraTBEdgeCGNN(tf.keras.Model):
    def __init__(self, base_cgnn, d_s, mode, **kwargs):
        super().__init__(**kwargs)
        self.base = base_cgnn
        self.d_s = int(d_s)
        self.mode = mode
        self.edge_hidden = Dense(64, activation="relu", name="edge_hidden")
        self.edge_candidate = Dense(1, activation="tanh", name="edge_candidate")
        self.edge_keep = Dense(1, activation="sigmoid", bias_initializer="zeros", name="edge_keep")
        self.msg_hidden = Dense(64, activation="relu", name="msg_hidden")
        self.msg_gate = Dense(
            d_s, activation="sigmoid",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
            bias_initializer=tf.keras.initializers.Constant(4.0), name="msg_gate")

    def zero_edge(self, batch_size, num_tx, dtype=tf.float32):
        return tf.zeros([batch_size, num_tx, num_tx, 1], dtype=dtype)

    def _pair_mask(self, active_tx, dtype):
        active = tf.cast(active_tx, dtype)
        pair = active[:, :, None] * active[:, None, :]
        u = tf.shape(active)[1]
        return pair * (1.0 - tf.eye(u, batch_shape=[tf.shape(active)[0]], dtype=dtype))

    def _aggregate(self, s, active_tx, prev_edge, original_agg):
        dtype = s.dtype
        b = tf.shape(s)[0]
        u = tf.shape(s)[1]
        sp = s
        for layer in original_agg._hidden_layers:
            sp = layer(sp)
        sp = original_agg._output_layer(sp)

        pooled = tf.reduce_mean(s, axis=[2, 3])
        recv = tf.broadcast_to(pooled[:, :, None, :], [b, u, u, self.d_s])
        send = tf.broadcast_to(pooled[:, None, :, :], [b, u, u, self.d_s])
        if prev_edge is None:
            prev_edge = self.zero_edge(b, u, dtype)
        else:
            prev_edge = tf.cast(prev_edge, dtype)

        eh = self.edge_hidden(tf.concat([recv, send, prev_edge], axis=-1))
        cand = self.edge_candidate(eh)
        keep = self.edge_keep(eh)
        edge = keep * prev_edge + (1.0 - keep) * cand
        mask = self._pair_mask(active_tx, dtype)
        edge = edge * mask[..., None]

        mh = self.msg_hidden(tf.concat([recv, send, edge], axis=-1))
        gate = self.msg_gate(mh) * mask[..., None]
        sender_sp = tf.broadcast_to(
            sp[:, None, :, :, :, :],
            [b, u, u, tf.shape(sp)[2], tf.shape(sp)[3], self.d_s])
        msg = sender_sp * gate[..., None, None, :]
        agg = tf.reduce_sum(msg, axis=2)
        count = tf.maximum(tf.reduce_sum(mask, axis=2), 1.0)
        return agg / count[..., None, None, None], edge

    def call(self, inputs):
        y, pe, h_hat, active_tx, mcs_ue_mask = inputs
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
                    tf.gather(mcs_ue_mask, indices=idx, axis=2), 5, axis=-1)

        edge = None
        llrs = []
        h_hats = []
        for i in range(base._num_it):
            it = base._iterations[i]
            edge_in = None if self.mode == "dynamic" else edge
            a, edge = self._aggregate(s, active_tx, edge_in, it._state_aggreg)
            s = it._state_update((s, a, pe))
            if i == base._num_it - 1:
                llrs_i = []
                for idx in range(base._num_mcss_supported):
                    if base._var_mcs_masking:
                        z = base._readout_llrs[0](s)
                        z = tf.gather(z, indices=tf.range(base._num_bits_per_symbol[idx]), axis=-1)
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


def forward(receiver, model, y, h_hat, active_tx):
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
    mask = tf.ones([tf.shape(y2)[0], num_tx, 1], tf.float32)
    llrs, h_hats, edge = model([y2, pe, h_hat, active, mask])
    return demap_llr(ofdm, llrs[-1][0], num_tx, 0), tf.cast(h_hats[-1], tf.float32), edge


def build_model():
    p = Parameters(CONFIG, training=False, num_tx_eval=NUM_TX, system="nrx")
    e2e = E2E_Model(p, training=False, mcs_arr_eval_idx=0)
    e2e(1, 1.0)
    load_weights(e2e, f"../weights/{p.label}_weights")
    base = e2e._receiver._neural_rx._cgnn
    base.num_it = ARGS.num_it
    model = IntraTBEdgeCGNN(base, p.d_s, MODE, name="temporal_edge_d1")
    return p, e2e, model


def make_tb(seq, p, h_slot, no):
    bsz = tf.shape(h_slot)[0]
    b = seq._source([bsz, p.max_num_tx, seq._tx._tb_size])
    x = seq._tx(b)
    y = seq._apply([x, h_slot, no])
    ls = seq._rx.estimate_channel(y, p.max_num_tx)
    active = tf.ones([bsz, p.max_num_tx], tf.float32)
    return b, y, ls, active


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try: sn.config.seed = seed
    except Exception: pass


def crossing(res, target=0.1):
    pts = sorted((float(k), v) for k, v in res.items())
    for (x0, r0), (x1, r1) in zip(pts, pts[1:]):
        y0, y1 = r0["tbler"], r1["tbler"]
        if y0 >= target > y1 and y0 > 0 and y1 > 0:
            l0, l1 = np.log10(y0), np.log10(y1)
            return float(x0 + (np.log10(target)-l0)*(x1-x0)/(l1-l0))
    return None


def evaluate(seq, p, model):
    res = {}
    for ebno in np.arange(1.0, 4.01, 0.5):
        no = ebno_to_no(p, seq._tx, float(ebno))
        errors = blocks = 0
        start = time.time()
        for _ in range(ARGS.eval_max_batches):
            h_seq = seq.sample_sequence_channel(ARGS.eval_batch, 8)
            for t in range(8):
                b, y, ls, active = make_tb(seq, p, h_seq[t], no)
                llr, _, _ = forward(seq._rx, model, y, ls, active)
                b_hat, _ = seq._rx._tb_decoders[0](llr)
                if t > 0:
                    err = tf.reduce_any(tf.not_equal(b, tf.cast(b_hat > 0.5, b.dtype)), axis=-1).numpy()
                    errors += int(err.sum()); blocks += err.size
            if errors >= ARGS.eval_target_errors:
                break
        tbler = errors / max(blocks, 1)
        res[str(float(ebno))] = {"tbler": tbler, "errors": errors, "blocks": blocks, "seconds": time.time()-start}
        print(f"EVAL mode={MODE} ebno={ebno:.1f} tbler={tbler:.6f} ({errors}/{blocks})", flush=True)
        if tbler < 0.015: break
    return res


def latency(seq, p, model):
    no = ebno_to_no(p, seq._tx, 3.0)
    h = seq.sample_sequence_channel(1, 1)[0]
    _, y, ls, active = make_tb(seq, p, h, no)
    @tf.function(jit_compile=False)
    def infer(y_, ls_, active_):
        z, _, e = forward(seq._rx, model, y_, ls_, active_)
        return z, e
    for _ in range(20):
        z, e = infer(y, ls, active); _ = z.numpy(); _ = e.numpy()
    times = []
    for _ in range(100):
        t0 = time.perf_counter(); z, e = infer(y, ls, active); _ = z.numpy(); _ = e.numpy()
        times.append((time.perf_counter()-t0)*1000.0)
    a = np.asarray(times)
    return {"median_ms": float(np.median(a)), "p10_ms": float(np.percentile(a,10)), "p90_ms": float(np.percentile(a,90)), "mean_ms": float(np.mean(a)), "samples_ms": a.tolist()}


def main():
    p, e2e, model = build_model()
    seq = TemporalSequence(p, e2e)
    h0 = seq.sample_sequence_channel(1,1)[0]
    no0 = ebno_to_no(p, seq._tx, 3.0)
    _, y0, ls0, a0 = make_tb(seq, p, h0, no0)
    _ = forward(seq._rx, model, y0, ls0, a0)
    ckpt = STUDY / "edge_memory" / "run_31523159794" / "dedge_1_k2.weights.h5"
    if not ckpt.exists():
        raise FileNotFoundError(f"Expected prior checkpoint not found: {ckpt}")
    model.load_weights(str(ckpt))
    set_seed(ARGS.seed + 1000)
    res = evaluate(seq, p, model)
    lat = latency(seq, p, model)
    cross = crossing(res)
    out = {
        "ablation_mode": MODE,
        "source_checkpoint_d_edge": 1,
        "num_it": ARGS.num_it,
        "normal": {"tbler_vs_ebno": res, "snr_at_10pct_tbler_db": cross},
        "reset_each_tb": {"tbler_vs_ebno": res, "snr_at_10pct_tbler_db": cross},
        "shuffle_previous_edge_at_3db": {"tbler": res.get("3.0",{}).get("tbler"), "errors": 0, "blocks": 0},
        "latency": lat,
        "edge_parameters": 18362,
        "persistent_edge_bytes_per_sequence": 0,
        "cross_tb_edge_state": False,
    }
    with open(OUT / f"eval_dedge_{ARGS.d_edge}.json", "w") as f: json.dump(out, f, indent=2)
    print("RESULT", json.dumps({"mode": MODE, "crossing": cross, "median_latency_ms": lat["median_ms"], "p90_latency_ms": lat["p90_ms"]}), flush=True)

if __name__ == "__main__": main()
