#!/usr/bin/env python3
"""Proper 4-UE NRX adaptation and decisive dynamic pair-weighting test.

Stage 1 trains a full K=8 cold NRX on 4-UE UMi with multiloss and checks
prefix K=1/K=2 TBLER crossings after staged chunks. Stage 2 trains the
weighted model from the same shipped NRX Large initialization for exactly the
same number of optimizer steps and schedule.
"""
import argparse, json, os, pickle, random, sys, time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['cold','weighted'],required=True)
p.add_argument('--gpu',type=int,required=True)
p.add_argument('--output-dir',required=True)
p.add_argument('--target-steps',type=int,default=0,help='0=adaptive cold training')
p.add_argument('--max-steps',type=int,default=40000)
p.add_argument('--chunk-steps',type=int,default=5000)
p.add_argument('--batch',type=int,default=64)
p.add_argument('--seed',type=int,default=20260811)
a=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
sys.path.insert(0,'..')

import numpy as np
import tensorflow as tf
import sionna as sn
from tensorflow.keras.layers import Dense, Layer
from utils import Parameters, E2E_Model, load_weights

for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g,True)
sn.Config.xla_compat=True
OUT=Path(a.output_dir); OUT.mkdir(parents=True,exist_ok=True)
CONFIG='nrx_large_4ue.cfg'; NUM_TX=4; FULL_K=8


def seed_all(s):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)
    try: sn.config.seed=s
    except Exception: pass

class DynamicPairAggregation(Layer):
    """Scalar current-round pair weights; no recurrent or cross-TB state."""
    def __init__(self, original_agg, d_s, **kw):
        super().__init__(**kw); self.original_agg=original_agg; self.d_s=int(d_s)
        self.score_hidden=Dense(32,activation='relu',name='pair_score_hidden')
        # Zero final logits makes the new model exactly equal-weight at init.
        self.score_out=Dense(1,kernel_initializer='zeros',bias_initializer='zeros',name='pair_score_out')
        self.last_weights=None
    def call(self,inputs):
        s,active_tx=inputs; dtype=s.dtype
        b=tf.shape(s)[0]; u=tf.shape(s)[1]
        sp=s
        for l in self.original_agg._hidden_layers: sp=l(sp)
        sp=self.original_agg._output_layer(sp)
        pooled=tf.reduce_mean(s,axis=[2,3])
        recv=tf.broadcast_to(pooled[:,:,None,:],[b,u,u,self.d_s])
        send=tf.broadcast_to(pooled[:,None,:,:],[b,u,u,self.d_s])
        logits=tf.squeeze(self.score_out(self.score_hidden(tf.concat([recv,send],axis=-1))),-1)
        active=tf.cast(active_tx,dtype)
        mask=active[:,:,None]*active[:,None,:]
        mask*=1.0-tf.eye(u,batch_shape=[b],dtype=dtype)
        logits=tf.where(mask>0,logits,tf.cast(-1e9,dtype))
        w=tf.nn.softmax(logits,axis=2)*mask
        w=tf.math.divide_no_nan(w,tf.reduce_sum(w,axis=2,keepdims=True))
        self.last_weights=w
        sender=tf.broadcast_to(sp[:,None,:,:,:,:],[b,u,u,tf.shape(sp)[2],tf.shape(sp)[3],self.d_s])
        return tf.reduce_sum(sender*w[...,None,None,None],axis=2)


def install_pair_heads(model,params):
    c=model._receiver._neural_rx._cgnn; layers=[]
    for i,it in enumerate(c._iterations):
        w=DynamicPairAggregation(it._state_aggreg,params.d_s,name=f'dynamic_pair_agg_{i}')
        it._state_aggreg=w; layers.append(w)
    return layers


def build_train(weighted):
    params=Parameters(CONFIG,training=True,system='nrx')
    m=E2E_Model(params,training=True,mcs_arr_eval_idx=0)
    _=m(1,8.0,num_tx=NUM_TX)
    load_weights(m,'../weights/nrx_large_weights')
    pair=[]
    if weighted:
        pair=install_pair_heads(m,params)
        _=m(1,8.0,num_tx=NUM_TX)
    c=m._receiver._neural_rx._cgnn
    c.num_it=FULL_K; c.apply_multiloss=True
    return params,m,pair


def build_eval(weighted,weights):
    params=Parameters(CONFIG,training=False,system='nrx')
    m=E2E_Model(params,training=False,return_tb_status=True,mcs_arr_eval_idx=0)
    _=m(1,8.0,num_tx=NUM_TX)
    if weighted:
        install_pair_heads(m,params); _=m(1,8.0,num_tx=NUM_TX)
    m.set_weights(weights)
    return m


def lr_for_step(step):
    if step<10000: return 2e-4
    if step<25000: return 1e-4
    return 3e-5


def crossing(points,target=.1):
    xs=sorted((float(k),float(v)) for k,v in points.items())
    for (x0,y0),(x1,y1) in zip(xs,xs[1:]):
        if y0>=target and y1<target and y0>0 and y1>0:
            l0,l1=np.log10(y0),np.log10(y1)
            return float(x0+(np.log10(target)-l0)*(x1-x0)/(l1-l0))
    return None


def quick_eval(weights,weighted,seed):
    # Coarse points are only for deciding whether training has reached a usable regime.
    model=build_eval(weighted,weights)
    result={}
    for k in (1,2):
        model._receiver._neural_rx._cgnn.num_it=k
        seed_all(seed+1000*k)
        pts={}
        for eb in (4.,8.,12.,16.,20.,24.,28.):
            err=blk=0
            for _ in range(25):
                _,_,ok=model(4,eb,num_tx=NUM_TX)
                ok=np.asarray(ok.numpy(),dtype=bool)
                err+=int(np.size(ok)-np.count_nonzero(ok)); blk+=int(np.size(ok))
                if err>=40: break
            pts[str(eb)]=err/max(blk,1)
        result[f'k{k}']={'tbler':pts,'crossing':crossing(pts)}
    return result


def full_eval(weights,weighted,seed):
    model=build_eval(weighted,weights); out={}
    for k in (1,2):
        model._receiver._neural_rx._cgnn.num_it=k
        seed_all(seed+10000*k)
        pts={}
        for eb in np.arange(2.,30.01,1.):
            err=blk=0
            for _ in range(80):
                _,_,ok=model(4,float(eb),num_tx=NUM_TX)
                ok=np.asarray(ok.numpy(),dtype=bool)
                err+=int(np.size(ok)-np.count_nonzero(ok)); blk+=int(np.size(ok))
                if err>=100: break
            pts[str(float(eb))]={'tbler':err/max(blk,1),'errors':err,'blocks':blk}
            if pts[str(float(eb))]['tbler']<0.015: break
        simple={x:v['tbler'] for x,v in pts.items()}
        out[f'k{k}']={'points':pts,'crossing':crossing(simple)}
    # diagnostic from a current-round forward pass
    diag=[]
    if weighted:
        model._receiver._neural_rx._cgnn.num_it=2
        _=model(2,12.,num_tx=NUM_TX)
        for it in model._receiver._neural_rx._cgnn._iterations[:2]:
            w=getattr(it._state_aggreg,'last_weights',None)
            if w is not None:
                z=w.numpy(); eye=np.eye(NUM_TX,dtype=bool)[None,:,:]
                off=z[~np.broadcast_to(eye,z.shape)]
                diag.append({'mean':float(off.mean()),'std':float(off.std()),'min':float(off.min()),'max':float(off.max())})
    out['pair_weight_diagnostic']=diag
    return out


def main():
    seed_all(a.seed)
    params,model,pair=build_train(a.mode=='weighted')
    opt=tf.keras.optimizers.Adam(learning_rate=lr_for_step(0))
    train_vars=model.trainable_variables

    @tf.function(jit_compile=True)
    def train_100():
        loss=tf.constant(0.,tf.float32); ld=tf.constant(0.,tf.float32); lc=tf.constant(0.,tf.float32)
        for _ in tf.range(100):
            eb=tf.random.uniform([a.batch],minval=2.,maxval=20.,dtype=tf.float32)
            with tf.GradientTape() as tape:
                ld,lc=model(a.batch,eb,num_tx=NUM_TX)
                loss=ld+0.01*lc
            grads=tape.gradient(loss,train_vars)
            gv=[(g,v) for g,v in zip(grads,train_vars) if g is not None]
            opt.apply_gradients(gv)
        return loss,ld,lc

    target=a.target_steps if a.target_steps>0 else a.max_steps
    step=0; hist=[]; staged=[]; t0=time.time()
    # Adaptive cold: inspect every chunk and stop only after both K1 and K2 have crossings.
    # Weighted: target_steps is fixed to the cold model's selected budget.
    while step<target:
        next_stop=min(step+a.chunk_steps,target)
        while step<next_stop:
            opt.learning_rate.assign(lr_for_step(step))
            loss,ld,lc=train_100(); step+=100
            if step%1000==0:
                row={'step':step,'lr':float(opt.learning_rate.numpy()),'loss':float(loss.numpy()),'loss_data':float(ld.numpy()),'loss_chest':float(lc.numpy()),'seconds':time.time()-t0}
                hist.append(row); print('TRAIN',json.dumps(row),flush=True)
        weights=model.get_weights()
        ck=OUT/f'{a.mode}_{step}_weights.pkl'
        with ck.open('wb') as f: pickle.dump(weights,f)
        q=quick_eval(weights,a.mode=='weighted',a.seed+step)
        staged.append({'step':step,'quick_eval':q})
        print('STAGE',json.dumps(staged[-1]),flush=True)
        if a.mode=='cold' and q['k1']['crossing'] is not None and q['k2']['crossing'] is not None:
            break

    weights=model.get_weights()
    final=full_eval(weights,a.mode=='weighted',a.seed+777777)
    pair_params=int(sum(np.prod(v.shape) for l in pair for v in l.trainable_variables)) if pair else 0
    result={'mode':a.mode,'num_ues':4,'channel':'UMi','full_training_k':8,'multiloss':True,
            'initialization':'shipped nrx_large_weights (2-UE pretrained backbone)',
            'steps':step,'batch':a.batch,'snr_training_db':[2.,20.],
            'lr_schedule':{'0-9999':2e-4,'10000-24999':1e-4,'25000+':3e-5},
            'history':hist,'stages':staged,'full_eval':final,'pair_parameters':pair_params,
            'seconds':time.time()-t0}
    path=OUT/f'{a.mode}_result.json'; path.write_text(json.dumps(result,indent=2))
    print('FINAL_RESULT',json.dumps({'mode':a.mode,'steps':step,'k1':final['k1']['crossing'],'k2':final['k2']['crossing'],'pair_parameters':pair_params,'pair_diag':final['pair_weight_diagnostic']}),flush=True)

if __name__=='__main__': main()
