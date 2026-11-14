"""Hardware-independent cost per TB of one CGNN iteration, the channel filter, and the IDD detector.

TensorFlow's graph profiler counts the floating-point operations of the TensorFlow parts (one CGNN iteration of the
Large NRX, the channel filter with its MLPs). It does not see the custom CUDA kernels of the IDD detector, so those are
counted by hand from native_ops/pair_correction.cu.cc and pair_prepare.cu.cc (see idd_detector() below). Everything is
for one slot: U=2 UEs, B=4 antennas, 132 PRBs, 14 OFDM symbols, 16-QAM. Needs a GPU to load the custom kernels; graphs are only traced.

    complexity.py            # prints and writes results/complexity.json
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def graph_flops(concrete):
    from tensorflow.python.profiler import model_analyzer
    from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
    options = ProfileOptionBuilder(ProfileOptionBuilder.float_operation()).with_empty_output().build()
    result = model_analyzer.profile(concrete.graph, options=options)
    per_op = {}
    for child in result.children:
        per_op[child.name] = per_op.get(child.name, 0)+child.total_float_ops
    by_type = {}
    for op in concrete.graph.get_operations():
        if op.name in per_op and per_op[op.name]:
            by_type[op.type] = by_type.get(op.type, 0)+per_op[op.name]
    return result.total_float_ops, by_type


def idd_detector(data_res, cells, users=2, antennas=4, points=16, bits=4, groups=9):
    """Real multiplications of the IDD detector per slot, counted from the CUDA kernels.

    Per data RE (pair_correction.cu.cc, physical_scores):
      - whitening of y for each of the 9 energy groups: forward substitution with the 4x4 Cholesky factor,
        r(r-1)/2 complex multiplications and r real scalings of a complex value;
      - for each of the 16x16 symbol pairs, each of the r rows: two complex multiplications (channel times symbol)
        and one squared magnitude (2 real multiplications);
      - the channel vectors at the RE: 9 groups x r rows x 2 UEs x 3 coefficients, complex times real.
    Marginalization (P without priors and Q with priors) needs exponentials and additions, counted separately.
    Per filter cell (pair_prepare.cu.cc): 9 Cholesky factorizations of a 4x4 Hermitian matrix and triangular solves
    for 6 right-hand sides.
    """
    complex_mult = 4
    r = antennas
    whitening = groups*(r*(r-1)//2*complex_mult+r*2)
    hypotheses = points**users*r*(users*complex_mult+2)
    vectors = groups*r*users*3*2
    per_re = whitening+hypotheses+vectors
    cholesky = r**3//3*complex_mult
    solves = 6*(r*(r-1)//2*complex_mult+r*2)
    per_cell = groups*(cholesky+solves)
    exponentials_per_re = 2*(users*points*points+users*bits*points)
    return {"real_mult_per_re": per_re, "real_mult_hypotheses_per_re": hypotheses,
            "real_mult_per_cell": per_cell, "data_res": data_res, "cells": cells,
            "real_mult": per_re*data_res+per_cell*cells, "exponentials": exponentials_per_re*data_res}


def filter_and_covariance(users, cells, neighbors, antennas, filter_inputs, filter_hidden, filter_outputs,
                          error_inputs, error_hidden, error_outputs):
    """Real multiplications of the channel filter and its error covariance per slot, counted from
    channel_filter.py (the profiler does not count its einsums). Complex x complex = 4, complex x real = 2."""
    r = antennas
    per_neighbor_filter = (4*r+2*r+2*r+6+6  # correlation, powers, feature ratios, beta
                           +filter_inputs*filter_hidden+filter_hidden*filter_outputs)
    per_neighbor_filter += 3*r*4  # coefficients einsum: 3 x r complex products
    per_neighbor_covariance = 9*2+3*2+1+2*r+4*r*r  # factor x beta, power, q_j, weights x innovation, outer product
    per_cell_covariance = r*(error_inputs*error_hidden+error_hidden*error_outputs)+4*r*r+4*r*r
    filter_total = users*cells*neighbors*per_neighbor_filter
    covariance_total = users*cells*(neighbors*per_neighbor_covariance+per_cell_covariance)
    return {"users": users, "cells_per_ue": cells, "neighbors": neighbors,
            "per_neighbor_filter": per_neighbor_filter, "per_neighbor_covariance": per_neighbor_covariance,
            "per_cell_covariance": per_cell_covariance, "filter_real_mult": filter_total,
            "covariance_real_mult": covariance_total, "real_mult": filter_total+covariance_total}


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import numpy as np
    import tensorflow as tf
    from neural_rx.benchmark import tbler_sweep as sweep

    sweep.configure_runtime()
    protocol = sweep.sweep_context()[0]
    adapter = sweep.make_receiver("candidate_persistent", protocol)
    model, receiver, layout = adapter.model, adapter.receiver, adapter.layout
    users, antennas = 2, 4
    grid = adapter.e2e._transmitters[0].resource_grid
    y = tf.zeros([1, 1, antennas, grid.num_ofdm_symbols, grid.fft_size], tf.complex64)
    channel = tf.zeros([1, users, grid.fft_size, grid.num_ofdm_symbols, 2*antennas])
    active = tf.ones([1, users])
    noise = tf.ones([1])
    inputs = adapter._prepare_inputs(receiver, y, channel, active)
    base = model.base
    state = base.initialize_state(inputs)

    iteration = tf.function(lambda s, pe, a: base._iterations[0]([s, pe, a])).get_concrete_function(
        tf.TensorSpec(state.shape, state.dtype), tf.TensorSpec(inputs[1].shape, inputs[1].dtype),
        tf.TensorSpec(inputs[3].shape, inputs[3].dtype))
    initial = tf.function(lambda *x: base.initialize_state(list(x))).get_concrete_function(
        *[tf.TensorSpec(v.shape, v.dtype) for v in inputs])
    readout = tf.function(lambda s: base._readout_llrs[0](s)).get_concrete_function(
        tf.TensorSpec(state.shape, state.dtype))

    observation = model.observe(channel, noise)
    history = model.empty_history(observation)
    model.prepare([None, None, channel, active], observation, history, layout)
    prepare = model._preparation_functions[layout.key].get_concrete_function(observation, history, active)

    report = {"grid": {"subcarriers": int(grid.fft_size), "ofdm_symbols": int(grid.num_ofdm_symbols),
                       "users": users, "antennas": antennas}}
    for name, concrete in (("cgnn_iteration", iteration), ("state_init", initial), ("readout", readout),
                           ("channel_filter", prepare)):
        flops, by_type = graph_flops(concrete)
        report[name] = {"flops": int(flops), "by_op_type": {k: int(v) for k, v in sorted(by_type.items())}}
    # Symbols per UE and TB, i.e., data REs carrying one symbol of each UE.
    data_res = int(adapter.decoder.symbol_indices.shape[1])
    report["grid"]["resource_grid_data_symbols"] = int(grid.num_data_symbols)
    report["idd_detector"] = idd_detector(data_res, int(len(layout.cell_positions)))
    context = model.channel_memory(observation, history, active, layout=layout, defer_uncertainty=True)
    _, _, cells, neighbors, _ = context.beta.shape
    memory = model.channel_memory
    f_in, f_hidden = memory.filter.layers[0].kernel.shape
    e_in, e_hidden = memory.error.layers[0].kernel.shape
    report["channel_filter_counted"] = filter_and_covariance(
        users, int(cells), int(neighbors), antennas, int(f_in), int(f_hidden), int(memory.filter.layers[1].units),
        int(e_in), int(e_hidden), int(memory.error.layers[1].units))
    coded_bits = int(np.prod(adapter.decoder.symbol_indices.shape))
    hidden = int(model.blend_kernel.shape[1])
    combining = int(model.blend_kernel.shape[0])*hidden+hidden+int(model.evidence_calibration.shape[0])+2
    report["idd_combining"] = {"coded_bits": coded_bits, "real_mult_per_bit": combining,
                               "real_mult": coded_bits*combining}
    report["weights"] = {"cgnn_iteration": int(sum(np.prod(v.shape) for v in base._iterations[0].weights))}
    (HERE/"results").mkdir(exist_ok=True)
    (HERE/"results/complexity.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
