#!/usr/bin/env python3
"""Evaluate K=8 on the trained 4-UE cold checkpoint from the decisive experiment."""
import argparse, json, os, pickle, random, sys, time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--gpu',type=int,default=0)
p.add_argument('--weights',required=True)
p.add_argument('--output',required=True)
p.add_argument('--seed',type=int,default=20260811)
a=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
sys.path.insert(0,'..')

import numpy as np
import tensorflow as tf
import sionna as sn
from utils import Parameters, E2E_Model

for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g,True)

CONFIG='nrx_large_4ue.cfg'; NUM_TX=4

def seed_all(s):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)
    try: sn.config.seed=s
    except Exception: pass

def crossing(points,target=.1):
    xs=sorted((float(k),float(v)) for k,v in points.items())
    for (x0,y0),(x1,y1) in zip(xs,xs[1:]):
        if y0>=target and y1<target and y0>0 and y1>0:
            l0,l1=np.log10(y0),np.log10(y1)
            return float(x0+(np.log10(target)-l0)*(x1-x0)/(l1-l0))
    return None

params=Parameters(CONFIG,training=False,system='nrx')
model=E2E_Model(params,training=False,return_tb_status=True,mcs_arr_eval_idx=0)
_=model(1,8.0,num_tx=NUM_TX)
with open(a.weights,'rb') as f:
    weights=pickle.load(f)
model.set_weights(weights)
model._receiver._neural_rx._cgnn.num_it=8
seed_all(a.seed+80000)

points={}
t0=time.time()
for eb in np.arange(2.,30.01,1.):
    err=blk=0
    for _ in range(120):
        _,_,ok=model(4,float(eb),num_tx=NUM_TX)
        ok=np.asarray(ok.numpy(),dtype=bool)
        err+=int(np.size(ok)-np.count_nonzero(ok)); blk+=int(np.size(ok))
        if err>=150: break
    tbler=err/max(blk,1)
    points[str(float(eb))]={'tbler':tbler,'errors':err,'blocks':blk}
    print(f'EVAL K=8 ebno={eb:.1f} tbler={tbler:.6f} ({err}/{blk})',flush=True)
    if tbler<0.01:
        break
simple={k:v['tbler'] for k,v in points.items()}
result={'num_ues':4,'k':8,'channel':'UMi','checkpoint':a.weights,
        'snr_at_10pct_tbler_db':crossing(simple),'points':points,
        'best_tbler':min(v['tbler'] for v in points.values()),
        'best_ebno_db':float(min(points,key=lambda k:points[k]['tbler'])),
        'seconds':time.time()-t0}
Path(a.output).write_text(json.dumps(result,indent=2))
print('RESULT',json.dumps(result),flush=True)
