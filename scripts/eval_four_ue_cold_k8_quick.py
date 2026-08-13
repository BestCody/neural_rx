#!/usr/bin/env python3
"""Fast high-SNR sanity check for K=8 on trained 4-UE cold checkpoint."""
import argparse, json, os, pickle, random, sys
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--gpu',type=int,default=1); p.add_argument('--weights',required=True); p.add_argument('--output',required=True); p.add_argument('--seed',type=int,default=20260811); a=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu); os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2'); sys.path.insert(0,'..')
import numpy as np, tensorflow as tf, sionna as sn
from utils import Parameters, E2E_Model
for g in tf.config.list_physical_devices('GPU'): tf.config.experimental.set_memory_growth(g,True)
random.seed(a.seed); np.random.seed(a.seed); tf.random.set_seed(a.seed)
try: sn.config.seed=a.seed
except Exception: pass
m=E2E_Model(Parameters('nrx_large_4ue.cfg',training=False,system='nrx'),training=False,return_tb_status=True,mcs_arr_eval_idx=0)
_=m(1,8.0,num_tx=4)
with open(a.weights,'rb') as f: m.set_weights(pickle.load(f))
m._receiver._neural_rx._cgnn.num_it=8
pts={}
for eb in (8.,16.,24.,30.):
    err=blk=0
    for _ in range(30):
        _,_,ok=m(4,eb,num_tx=4); ok=np.asarray(ok.numpy(),dtype=bool)
        err+=int(np.size(ok)-np.count_nonzero(ok)); blk+=int(np.size(ok))
        if err>=50: break
    pts[str(eb)]={'tbler':err/max(blk,1),'errors':err,'blocks':blk}
    print(f'QUICK K8 ebno={eb:.1f} tbler={pts[str(eb)]["tbler"]:.6f} ({err}/{blk})',flush=True)
r={'k':8,'points':pts,'high_snr_30db_tbler':pts['30.0']['tbler']}
Path(a.output).write_text(json.dumps(r,indent=2)); print('RESULT',json.dumps(r),flush=True)
