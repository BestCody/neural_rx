"""Per-layer GPU time of the FP16 engines of the Large NRX (N_it = 2) and the proposed NRX with the TensorRT-compiled
filter, from TensorRT's layer profiler (it synchronizes after each layer, so totals are slightly above time_trt.py).
Prints each engine's layers before StateInit: input normalization and format conversions, plus the filter.

    ~/venvs/tensorrt/bin/python profile_layers.py     # GPU 0; uses build/*_fp16.plan from time_trt.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import json, sys, numpy as np, tensorrt as trt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cuda import cudart
from time_trt import Engine, call
logger = trt.Logger(trt.Logger.ERROR); rt = trt.Runtime(logger)
reference = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build/reference.npz'))

class Profiler(trt.IProfiler):
    def __init__(self):
        trt.IProfiler.__init__(self); self.times = {}
    def report_layer_time(self, name, ms):
        self.times.setdefault(name, []).append(ms)

for name in ('large_k2', 'proposed'):
    feeds = {k.split('/')[-1]: reference[k] for k in reference.files if k.startswith(f'{name}/input/')}
    engine = Engine(trt, cudart, np, rt, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'build/{name}_fp16.plan'), 'rb').read(), feeds)
    for _ in range(50): engine.run()
    profiler = Profiler(); engine.context.profiler = profiler
    for _ in range(300): engine.run()
    medians = {k: float(np.median(v)) for k, v in profiler.times.items()}
    total = sum(medians.values())
    groups = {}
    for k, v in medians.items():
        if 'Reformatting' in k: g = 'format conversions'
        elif k.startswith(('state_init', 'cgnn_it', 'readout')) or 'myl1' in k or 'myl2' in k or 'myl3' in k: g = 'CGNN'
        else: g = 'filter + input normalization'
        groups[g] = groups.get(g, 0) + v
    print('==', name, 'sum of layer medians', round(total, 3), 'ms', {g: round(v, 3) for g, v in groups.items()})
    prefix = 'myl6_' if name == 'proposed' else 'myl2_'
    first = [(k, v) for k, v in medians.items() if prefix in k or (k.startswith('Reformatting') and ('Equal_1' in k or 'sub_2/y' in k))]
    for k, v in first:
        print('  ', round(v * 1000, 1), 'us', k[:100])
    print('   total of the part before StateInit:', round(1000*sum(v for _, v in first), 1), 'us')
