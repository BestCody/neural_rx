#!/usr/bin/env python3
"""Matched 4-UE experiment: original mean aggregation vs dynamic scalar pair weighting."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, required=True, choices=[1, 2])
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--train-steps", type=int, default=1200)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--eval-batch", type=int, default=4)
    p.add_argument("--eval-target-errors", type=int, default=80)
    p.add_argument("--eval-max-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260811)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


ARGS = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

# Run from neural_rx/scripts; expose repository root for utils imports.
sys.path.insert(0, "..")
from utils import Parameters, E2E_Model, load_weights

CONFIG = "nrx_large_4ue.cfg"
NUM_TX = 4
OUT = Path(ARGS.output_dir)
OUT.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        import sionna as sn
        sn.config.seed = seed
    except Exception:
        pass


class DynamicPairAggregation(Layer):
    """Current-round scalar pair weights; no state is carried across rounds/TBs."""
    def __init__(self, original_agg, d_s, **kwargs):
        super().__init__(**kwargs)
        self.original_agg = original_agg
        self.d_s = int(d_s)
        self.score_hidden = Dense(32, activation="relu", name="pair_score_hidden")
        self.score_out = Dense(1, activation=None, kernel_initializer="zeros", bias_initializer="zeros", name="pair_score_out")
        self.last_weights = None

    @property
    def pair_variables(self):
        return self.score_hidden.trainable_variables + self.score_out.trainable_variables

    def call(self, inputs):
        s, active_tx = inputs
        dtype = s.dtype
        b = tf.shape(s)[0]
        u = tf.shape(s)[1]
        sp = s
        for layer in self.original_agg._hidden_layers:
            sp = layer(sp)
        sp = self.original_agg._output_layer(sp)
        pooled = tf.reduce_mean(s, axis=[2, 3])
        recv = tf.broadcast_to(pooled[:, :, None, :], [b, u, u, self.d_s])
        send = tf.broadcast_to(pooled[:, None, :, :], [b, u, u, self.d_s])
        logits = self.score_out(self.score_hidden(tf.concat([recv, send], axis=-1)))
        logits = tf.squeeze(logits, axis=-1)
        active = tf.cast(active_tx, dtype)
        mask = active[:, :, None] * active[:, None, :]
        mask *= (1.0 - tf.eye(u, batch_shape=[b], dtype=dtype))
        masked_logits = tf.where(mask > 0., logits, tf.cast(-1e9, dtype))
        weights = tf.nn.softmax(masked_logits, axis=2) * mask
        denom = tf.reduce_sum(weights, axis=2, keepdims=True)
        weights = tf.math.divide_no_nan(weights, denom)
        self.last_weights = weights
        sender_sp = tf.broadcast_to(sp[:, None, :, :, :, :], [b, u, u, tf.shape(sp)[2], tf.shape(sp)[3], self.d_s])
        return tf.reduce_sum(sender_sp * weights[..., None, None, None], axis=2)


def build_model(training, weighted):
    p = Parameters(CONFIG, training=training, system="nrx")
    model = E2E_Model(p, training=training, mcs_arr_eval_idx=0)
    _ = model(1, 4.0, num_tx=NUM_TX)
    load_weights(model, "../weights/nrx_large_weights")
    cgnn = model._receiver._neural_rx._cgnn
    cgnn.num_it = ARGS.k
    pair_layers = []
    if weighted:
        for i in range(ARGS.k):
            it = cgnn._iterations[i]
            wrapped = DynamicPairAggregation(it._state_aggreg, p.d_s, name=f"dynamic_pair_agg_{i}")
            it._state_aggreg = wrapped
            pair_layers.append(wrapped)
        _ = model(1, 4.0, num_tx=NUM_TX)
    return p, model, pair_layers


def train_variant(name, weighted):
    tf.keras.backend.clear_session()
    p, model, pair_layers = build_model(training=True, weighted=weighted)
    set_seed(ARGS.seed + 100 * ARGS.k)
    base_opt = tf.keras.optimizers.Adam(2e-5)
    pair_opt = tf.keras.optimizers.Adam(1e-3) if weighted else None
    history = []
    t0 = time.time()
    for step in range(ARGS.train_steps):
        ebno = float(np.random.uniform(0.0, 10.0))
        with tf.GradientTape() as tape:
            loss_data, loss_chest = model(ARGS.batch, ebno, num_tx=NUM_TX)
            loss = loss_data + 0.01 * loss_chest
        all_vars = model.trainable_variables
        grads = tape.gradient(loss, all_vars)
        if weighted:
            pvars = []
            for layer in pair_layers:
                pvars.extend(layer.pair_variables)
            pids = {id(v) for v in pvars}
            base_pairs = [(g, v) for g, v in zip(grads, all_vars) if g is not None and id(v) not in pids]
            pair_pairs = [(g, v) for g, v in zip(grads, all_vars) if g is not None and id(v) in pids]
            if base_pairs:
                base_opt.apply_gradients(base_pairs)
            if pair_pairs:
                pair_opt.apply_gradients(pair_pairs)
        else:
            gv = [(g, v) for g, v in zip(grads, all_vars) if g is not None]
            base_opt.apply_gradients(gv)
        if step % 100 == 0 or step == ARGS.train_steps - 1:
            row = {"step": step, "ebno_db": ebno, "loss": float(loss.numpy()), "loss_data": float(loss_data.numpy()), "loss_chest": float(loss_chest.numpy()), "seconds": time.time() - t0}
            history.append(row)
            print("TRAIN", name, json.dumps(row), flush=True)
    ckpt = OUT / f"{name}_k{ARGS.k}.weights.h5"
    model.save_weights(str(ckpt))
    return {"checkpoint": str(ckpt), "history": history, "train_seconds": time.time() - t0, "trainable_parameters": int(sum(np.prod(v.shape) for v in model.trainable_variables)), "pair_parameters": int(sum(np.prod(v.shape) for layer in pair_layers for v in layer.pair_variables))}


def crossing(points, target=0.1):
    xs = sorted((float(k), v) for k, v in points.items())
    for (x0, r0), (x1, r1) in zip(xs, xs[1:]):
        y0, y1 = r0["tbler"], r1["tbler"]
        if y0 >= target > y1 and y0 > 0 and y1 > 0:
            l0, l1 = np.log10(y0), np.log10(y1)
            return float(x0 + (np.log10(target) - l0) * (x1 - x0) / (l1 - l0))
    return None


def eval_variant(name, weighted, ckpt):
    tf.keras.backend.clear_session()
    _, model, pair_layers = build_model(training=False, weighted=weighted)
    model.load_weights(ckpt)
    set_seed(ARGS.seed + 10000 + 100 * ARGS.k)
    ebnos = np.arange(0.0, 12.01, 1.0)
    res = {}
    for ebno in ebnos:
        errors = 0
        blocks = 0
        start = time.time()
        for _ in range(ARGS.eval_max_batches):
            b, b_hat = model(ARGS.eval_batch, float(ebno), num_tx=NUM_TX)
            err = tf.reduce_any(tf.not_equal(b, tf.cast(b_hat > 0.5, b.dtype)), axis=-1)
            errors += int(tf.reduce_sum(tf.cast(err, tf.int32)).numpy())
            blocks += int(np.prod(err.shape))
            if errors >= ARGS.eval_target_errors:
                break
        tbler = errors / max(blocks, 1)
        res[str(float(ebno))] = {"tbler": tbler, "errors": errors, "blocks": blocks, "seconds": time.time() - start}
        print(f"EVAL {name} K={ARGS.k} ebno={ebno:.1f} tbler={tbler:.6f} ({errors}/{blocks})", flush=True)
        if tbler < 0.02:
            break
    pair_diag = None
    if weighted:
        _ = model(2, 4.0, num_tx=NUM_TX)
        vals = []
        for layer in pair_layers:
            if layer.last_weights is not None:
                w = layer.last_weights.numpy()
                eye = np.eye(NUM_TX, dtype=bool)[None, :, :]
                off = w[~np.broadcast_to(eye, w.shape)].reshape(-1)
                vals.append(off)
        if vals:
            z = np.concatenate(vals)
            pair_diag = {"mean_offdiag_weight": float(np.mean(z)), "std_offdiag_weight": float(np.std(z)), "min_offdiag_weight": float(np.min(z)), "max_offdiag_weight": float(np.max(z)), "equal_weight_reference": 1.0 / 3.0}
    return {"tbler_vs_ebno": res, "snr_at_10pct_tbler_db": crossing(res), "pair_weight_diagnostic": pair_diag}


def main():
    print(json.dumps({"k": ARGS.k, "gpu": ARGS.gpu, "num_tx": NUM_TX, "tensorflow": tf.__version__, "visible_gpus": [x.name for x in tf.config.list_physical_devices("GPU")]}), flush=True)
    cold_train = train_variant("cold4", weighted=False)
    weighted_train = train_variant("weighted4", weighted=True)
    cold_eval = eval_variant("cold4", False, cold_train["checkpoint"])
    weighted_eval = eval_variant("weighted4", True, weighted_train["checkpoint"])
    out = {"num_tx": NUM_TX, "channel": "UMi", "k": ARGS.k, "training_steps_each": ARGS.train_steps, "cold": {"train": cold_train, "eval": cold_eval}, "weighted": {"train": weighted_train, "eval": weighted_eval}}
    c = cold_eval["snr_at_10pct_tbler_db"]
    w = weighted_eval["snr_at_10pct_tbler_db"]
    out["weighted_gain_db"] = None if c is None or w is None else c - w
    path = OUT / f"four_ue_pair_k{ARGS.k}.json"
    path.write_text(json.dumps(out, indent=2))
    print("RESULT", json.dumps({"k": ARGS.k, "cold_crossing": c, "weighted_crossing": w, "weighted_gain_db": out["weighted_gain_db"], "pair_parameters": weighted_train["pair_parameters"], "pair_diag": weighted_eval["pair_weight_diagnostic"]}), flush=True)


if __name__ == "__main__":
    main()
