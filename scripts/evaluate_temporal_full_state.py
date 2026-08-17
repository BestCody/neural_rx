#!/usr/bin/env python3
"""Evaluate the raw full-state temporal-memory upper bound at 132 PRBs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="nrx_large.cfg")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--snr-min", type=float, default=1.5)
    p.add_argument("--snr-max", type=float, default=3.75)
    p.add_argument("--snr-step", type=float, default=0.25)
    p.add_argument("--target-errors", type=int, default=120)
    p.add_argument("--max-batches", type=int, default=32)
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--ue-pool-size", type=int, default=4)
    p.add_argument("--dynamic-scheduling", action="store_true")
    p.add_argument("--schedule-switch-prob", type=float, default=0.65)
    p.add_argument("--schedule-reorder-prob", type=float, default=0.50)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


ARGS = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import sionna as sn

from temporal_full_state import build_system, make_manager, temporal_forward

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


def set_seed(seed):
    np.random.seed(seed); tf.random.set_seed(seed)
    try: sn.config.seed = seed
    except Exception: pass


def make_snrs():
    n = int(round((ARGS.snr_max - ARGS.snr_min) / ARGS.snr_step))
    xs = [ARGS.snr_min + i * ARGS.snr_step for i in range(n + 1)]
    if xs[-1] < ARGS.snr_max - 1e-9: xs.append(ARGS.snr_max)
    return [float(round(x, 10)) for x in xs]


def wilson(e, n, z=1.959963984540054):
    if n <= 0: return [None, None]
    p = e / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    r = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return [max(0.0, c-r), min(1.0, c+r)]


def crossing(points, target=0.1):
    pts = sorted((float(p["snr_db"]), float(p["bler_tb2plus"])) for p in points if p["bler_tb2plus"] not in (None, 0))
    for (x0,y0),(x1,y1) in zip(pts[:-1], pts[1:]):
        if (y0-target)*(y1-target) <= 0:
            if y0 == y1: return (x0+x1)/2
            f = (math.log10(target)-math.log10(y0))/(math.log10(y1)-math.log10(y0))
            return float(x0 + f*(x1-x0))
    return None


def count(bits, bhat, active):
    err = tf.reduce_any(tf.not_equal(tf.cast(bits, tf.int32), tf.cast(tf.round(bhat), tf.int32)), axis=-1)
    mask = tf.cast(active, tf.bool); err = tf.logical_and(err, mask)
    return int(tf.reduce_sum(tf.cast(err, tf.int64)).numpy()), int(tf.reduce_sum(tf.cast(mask, tf.int64)).numpy())


def new_counter():
    return {"all_e":0,"all_n":0,"warm_e":0,"warm_n":0,"per_e":[0]*ARGS.seq_len,"per_n":[0]*ARGS.seq_len}


def add(c,t,bits,bhat,active):
    e,n = count(bits,bhat,active)
    c["all_e"] += e; c["all_n"] += n; c["per_e"][t] += e; c["per_n"][t] += n
    if t >= 1: c["warm_e"] += e; c["warm_n"] += n


def finish(c,snr,batches):
    r = lambda a,b: a/b if b else None
    return {
        "snr_db":float(snr),"batches":int(batches),
        "errors_all":c["all_e"],"blocks_all":c["all_n"],"bler_all":r(c["all_e"],c["all_n"]),
        "errors_tb2plus":c["warm_e"],"blocks_tb2plus":c["warm_n"],"bler_tb2plus":r(c["warm_e"],c["warm_n"]),
        "ci95_tb2plus":wilson(c["warm_e"],c["warm_n"]),
        "per_tb":[{"tb":i+1,"errors":e,"blocks":n,"bler":r(e,n),"ci95":wilson(e,n)} for i,(e,n) in enumerate(zip(c["per_e"],c["per_n"]))],
    }


def build_temporal_eval():
    p,e2e,model,generator = build_system(
        config=ARGS.config,num_it=2,training=False,ue_pool_size=ARGS.ue_pool_size,
        dynamic_scheduling=ARGS.dynamic_scheduling,
        schedule_switch_prob=ARGS.schedule_switch_prob,
        schedule_reorder_prob=ARGS.schedule_reorder_prob)
    warm = generator.sample_batch(1, ARGS.seq_len, 3.0)
    manager,d_mem,state_shape = make_manager(e2e._receiver,model,warm,capacity=ARGS.ue_pool_size,expiry_slots=8)
    state = manager.zero_state(1, tf.float32)
    state,mem,gap,valid = manager.gather(state,warm["ue_ids"][:,0],0)
    temporal_forward(e2e._receiver,model,warm["y"][:,0],warm["ls"][:,0],warm["active"][:,0],mem,gap,valid)
    ckpt = Path(ARGS.checkpoint).expanduser()
    if not ckpt.exists(): raise FileNotFoundError(ckpt)
    model.load_weights(str(ckpt))
    return p,e2e,model,generator,manager,d_mem,state_shape,ckpt


def build_cold(k,warm):
    p,e2e,model,_ = build_system(config=ARGS.config,num_it=k,training=False,ue_pool_size=2,dynamic_scheduling=False)
    manager,d_mem,_ = make_manager(e2e._receiver,model,warm,capacity=2,expiry_slots=8)
    b = int(warm["y"].shape[0]); u = int(warm["active"].shape[-1])
    z = tf.zeros([b,u,d_mem],tf.float32); zg=tf.zeros([b,u],tf.int32); inv=tf.zeros([b,u],tf.bool)
    temporal_forward(e2e._receiver,model,warm["y"][:,0],warm["ls"][:,0],warm["active"][:,0],z,zg,inv)
    return e2e._receiver,model,d_mem


def run():
    set_seed(ARGS.seed)
    p,e2e,temporal,generator,manager,d_mem,state_shape,ckpt = build_temporal_eval()
    warm = generator.sample_batch(1,ARGS.seq_len,3.0)
    rx2,cold2,d2 = build_cold(2,warm); rx8,cold8,d8 = build_cold(8,warm)
    if d2 != d_mem or d8 != d_mem: raise RuntimeError("cold and temporal full-state shapes differ")
    set_seed(ARGS.seed)
    methods=["cold_k2","cold_k8","temporal_k2_full_state"]
    curves={m:[] for m in methods}
    for snr in make_snrs():
        cs={m:new_counter() for m in methods}; batches=0
        for bi in range(ARGS.max_batches):
            batch=generator.sample_batch(ARGS.batch_size,ARGS.seq_len,snr)
            b=int(batch["y"].shape[0]); u=int(batch["active"].shape[-1])
            state=manager.zero_state(b,tf.float32)
            z=tf.zeros([b,u,d_mem],tf.float32); zg=tf.zeros([b,u],tf.int32); inv=tf.zeros([b,u],tf.bool)
            for t in range(ARGS.seq_len):
                bits=batch["bits"][:,t]; y=batch["y"][:,t]; ls=batch["ls"][:,t]; active=batch["active"][:,t]
                l2,_,_=temporal_forward(rx2,cold2,y,ls,active,z,zg,inv); b2,_=rx2._tb_decoders[0](l2); add(cs["cold_k2"],t,bits,b2,active)
                l8,_,_=temporal_forward(rx8,cold8,y,ls,active,z,zg,inv); b8,_=rx8._tb_decoders[0](l8); add(cs["cold_k8"],t,bits,b8,active)
                state,mem,gap,valid=manager.gather(state,batch["ue_ids"][:,t],t)
                lt,_,nxt=temporal_forward(e2e._receiver,temporal,y,ls,active,mem,gap,valid); bt,_=e2e._receiver._tb_decoders[0](lt); add(cs["temporal_k2_full_state"],t,bits,bt,active)
                state=manager.scatter(state,batch["ue_ids"][:,t],nxt,active,t)
            batches=bi+1
            if all(cs[m]["warm_e"]>=ARGS.target_errors for m in methods): break
        for m in methods:
            pt=finish(cs[m],snr,batches); curves[m].append(pt); print("EVAL_POINT="+json.dumps({"method":m,**pt}),flush=True)
    cross={m:crossing(curves[m]) for m in methods}
    c2,c8,ct=cross["cold_k2"],cross["cold_k8"],cross["temporal_k2_full_state"]
    gap=c2-c8 if c2 is not None and c8 is not None else None
    imp=c2-ct if c2 is not None and ct is not None else None
    rec=imp/gap if gap is not None and gap>0 and imp is not None else None
    return {
        "experiment":"temporal_raw_full_state_132prb_evaluation_v1","checkpoint":str(ckpt),
        "parameter_mode":"training=False","n_size_bwp":int(p.n_size_bwp),"d_s":int(p.d_s),
        "state_shape_per_ue":list(state_shape),"memory_floats_per_ue":int(d_mem),"memory_bytes_per_ue":int(d_mem*4),"memory_bits_per_ue":int(d_mem*32),
        "seq_len":ARGS.seq_len,"primary_metric":"TB2+ TBLER","snr_db_at_10pct_tbler":cross,
        "cold_iteration_gap_db":gap,"temporal_improvement_over_cold_k2_db":imp,"gap_recovered_fraction":rec,"gap_recovered_percent":100*rec if rec is not None else None,
        "dynamic_scheduling":ARGS.dynamic_scheduling,"ue_pool_size":ARGS.ue_pool_size,"schedule_switch_prob":ARGS.schedule_switch_prob,"schedule_reorder_prob":ARGS.schedule_reorder_prob,
        "seed":ARGS.seed,"curves":curves,
    }


def write(s):
    out=Path(ARGS.output_dir).expanduser(); out.mkdir(parents=True,exist_ok=True)
    (out/"evaluation.json").write_text(json.dumps(s,indent=2)+"\n")
    with (out/"curves.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["method","snr_db","bler_all","bler_tb2plus","errors_tb2plus","blocks_tb2plus"])
        for m,pts in s["curves"].items():
            for p in pts: w.writerow([m,p["snr_db"],p["bler_all"],p["bler_tb2plus"],p["errors_tb2plus"],p["blocks_tb2plus"]])
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.5,5.0))
    for m,pts in s["curves"].items(): ax.semilogy([p["snr_db"] for p in pts],[p["bler_tb2plus"] for p in pts],marker="o",label=m)
    ax.axhline(0.1,linestyle="--",linewidth=1); ax.set_xlabel("Eb/N0 (dB)"); ax.set_ylabel("TB2+ TBLER"); ax.set_title("132-PRB raw full-state temporal upper bound"); ax.grid(True,which="both",alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out/"tbler_vs_snr.png",dpi=180); fig.savefig(out/"tbler_vs_snr.pdf"); plt.close(fig)
    print("FULL_STATE_EVALUATION_SUMMARY="+json.dumps(s,indent=2),flush=True)


if __name__=="__main__": write(run())
