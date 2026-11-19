"""Export the NRX networks to ONNX for TensorRT timing, as in Wiesmayr et al. (2025).

Scope (as in their code release): the neural network only, with batch size 1, from the received slot and the LS channel
estimate to the LLRs. LS estimation, LDPC decoding, and host-device copies are outside. Networks:

  large_k2, _k3, _k8  released Large NRX (models/pretrained/nrx_large_weights): StateInit, N_it CGNN iterations,
                       LLR readout
  proposed_cgnn        the network of the proposed NRX: StateInit, two CGNN iterations, and LLR readout, with the
                       refined channel as an input. The channel filter runs before it as CUDA kernels
                       (fitted_filter.cu, time_fused.py); build/filter_tables.npz holds the kernels' tables and the
                       TensorFlow receiver's refined channel for the check, build/reference.npz one UMi slot with a
                       full memory (the memory entries are LS estimates, so the network has no second output).

Each exported function is first checked against the original TensorFlow receiver on that slot.

    export.py [--checkpoint DIR]     # writes build/*.onnx, build/*.npz, results/export.json (needs a GPU)

--channels-first exports the same CGNNs (large_k2_cf, _k3_cf, _k8_cf, proposed_cgnn_cf) with every layer in the layout
TensorRT computes in (users as the batch, channels first) instead of TensorFlow's channels-last layout, which removes
the transposes and copies between the layers. The weights and the function are unchanged, and it applies to the Large
NRX and the proposed NRX alike. It writes build/*_cf.onnx, build/reference_cf.npz, and results/export_cf.json and
leaves the other files as they are.
"""
import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWEEP = HERE.parents[1]/"experiments/20261104_tbler_sweep/results"
BUILD = HERE/"build"
CONDITION, SETTING, EPISODE, TB, SNR_INDEX = "natural_umi", "sweep", 0, 24, 8


def channels_first_network(tf, cgnn, iterations, channel, y_value, active_value, pe, pilots=None):
    """CGNN.initialize_state, the first CGNN iterations, and the LLR readout with the users as the batch and the
    channels first: [users, channels, subcarriers, OFDM symbols] throughout. Dense layers are 1x1 convolutions.
    With `pilots` (PilotLayout.pilot_positions) also the memory record: see memory_record()."""
    users = pe.shape[0]

    def first(value):                                    # [U, S, T, C] -> [U, C, S, T]
        return tf.transpose(value, [0, 3, 1, 2])

    def pointwise(value, kernel, bias, relu):
        value = tf.nn.conv2d(value, kernel, 1, "VALID", data_format="NCHW")
        value = tf.nn.bias_add(value, bias, data_format="NCHW")
        return tf.nn.relu(value) if relu else value

    def separable(value, layer):
        value = tf.nn.depthwise_conv2d(value, layer.depthwise_kernel, [1, 1, 1, 1], "SAME", data_format="NCHW")
        return pointwise(value, layer.pointwise_kernel, layer.bias, layer.activation is tf.keras.activations.relu)

    def dense(value, layer):
        return pointwise(value, layer.kernel[None, None], layer.bias, layer.activation is tf.keras.activations.relu)

    def separable_of_parts(parts, layer, order):
        """separable(tf.concat(parts, 1), layer). A depthwise convolution acts on each channel alone, so it is applied
        to each part, and the outputs are concatenated in the given order, with the pointwise kernel's input channels
        in the same order. TensorRT then writes the convolutions' outputs into one buffer without copying them."""
        offset, outputs, kernels = 0, [], []
        for part in parts:
            channels = slice(offset, offset+part.shape[1])
            offset += part.shape[1]
            outputs.append(tf.nn.depthwise_conv2d(part, layer.depthwise_kernel[:, :, channels], [1, 1, 1, 1], "SAME",
                                                  data_format="NCHW"))
            kernels.append(layer.pointwise_kernel[:, :, channels])
        return pointwise(tf.concat([outputs[i] for i in order], 1), tf.concat([kernels[i] for i in order], 2),
                         layer.bias, layer.activation is tf.keras.activations.relu)

    norm = cgnn.input_scale(y_value)
    y_value, channel = y_value*norm, channel*norm[:, None]
    positions = first(tf.constant(pe))
    start = first(tf.concat([tf.tile(y_value, [users, 1, 1, 1]), pe, channel[0]], -1))
    state = start                                        # all users use the first MCS, as in network() of main()
    for layer in [*cgnn._s_init[0]._hidden_conv, cgnn._s_init[0]._output_conv]:
        state = separable(state, layer)
    active = tf.reshape(active_value, [users, 1, 1, 1])
    others = tf.nn.relu(tf.reduce_sum(active, 0, keepdims=True)-1.)
    others = tf.where(others == 0., 1., tf.math.divide_no_nan(1., others))
    for iteration in cgnn._iterations[:iterations]:
        message = state
        for layer in [*iteration._state_aggreg._hidden_layers, iteration._state_aggreg._output_layer]:
            message = dense(message, layer)
        message = (tf.add_n([message[user:user+1]*active[user] for user in range(users)])-message*active)*others
        update = iteration._state_update
        layers = [*update._hidden_conv, update._output_conv]
        # the state first: in this order, TensorRT 10.3 copies neither the message nor the state (profile_layers.py)
        value = separable_of_parts([message, state, positions], layers[0], order=(1, 0, 2))
        for layer in layers[1:]:
            value = separable(value, layer)
        state = value+state
    record = None
    if pilots is not None:
        # the channel-estimate readout only at the pilots: [U, C, observations, pilots per observation]
        flat = tf.reshape(state, [users, state.shape[1], -1])
        value = tf.stack([tf.gather(flat[user], pilots[user].reshape(-1), axis=1) for user in range(users)])
        value = tf.reshape(value, [users, state.shape[1], *pilots.shape[1:]])
        for layer in [*cgnn._readout_chest._hidden_layers, cgnn._readout_chest._output_layer]:
            value = dense(value, layer)
        record = tf.transpose(tf.reduce_mean(value, 3), [0, 2, 1])[None]/norm
    readout = cgnn._readout_llrs[0]
    for layer in [*readout._hidden_layers, readout._output_layer]:
        state = dense(state, layer)
    llr = tf.transpose(state, [0, 2, 3, 1])[None]
    return llr if pilots is None else (llr, record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels-first", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint of the proposed NRX (default: the benchmark's)")
    args = parser.parse_args()
    channels_first = args.channels_first
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import numpy as np
    import tensorflow as tf
    import tf2onnx
    from neural_rx.backbone.core import build_backbone, prepare_cgnn_inputs
    from neural_rx.benchmark import tbler_sweep as sweep
    from neural_rx.benchmark.runtime import validate_episode
    from neural_rx.receiver.pilot_history import PilotHistory

    sweep.configure_runtime()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    if args.checkpoint:
        sweep.CHECKPOINT = args.checkpoint.resolve()
    protocol, plan, parameters, bank, anchors = sweep.sweep_context()
    adapter = sweep.make_receiver("candidate_persistent", protocol)
    model, layout, receiver = adapter.model, adapter.layout, adapter.receiver

    # One UMi slot at a mid SNR, with the memory holding the previous seven TBs of both UEs (ages 1-7, own ports).
    job = sweep.sweep_job(plan, anchors, CONDITION)
    with np.load(SWEEP/SETTING/CONDITION/"candidate_persistent"/f"episode_{EPISODE:05d}.npz") as stored:
        snr = stored["snr_db"]
        fingerprint = stored["fingerprint"]
    job = dict(job, snr_db=[float(v) for v in snr])
    episode = validate_episode(bank.generate(job, EPISODE), protocol, job)
    assert np.array_equal(np.asarray(episode.fingerprint), fingerprint)
    point = episode.receiver_inputs[SNR_INDEX]
    assert episode.active[TB-7:TB+1].all() and all((point[t]["ue_ids"] == point[TB]["ue_ids"]).all()
                                                    for t in range(TB-7, TB))
    y = tf.convert_to_tensor(point[TB]["received_grid"][None], tf.complex64)
    ls = tf.convert_to_tensor(point[TB]["ls_channel"][None], tf.float32)
    noise = tf.constant([point[TB]["noise_variance"]], tf.float32)
    active = tf.ones([1, layout.users])
    # The memory as in operation: the receiver decodes the TBs before this one, each leaving its LS estimate.
    adapter.start_episode()
    for t in range(TB):
        adapter.decode_tb(point[t])
    read = adapter._buffer.read(adapter._committed.head, adapter._committed.count,
                                tf.constant(point[TB]["ue_ids"][None]), tf.ones([1, layout.users], tf.bool),
                                tf.constant(TB))
    assert bool(tf.reduce_all(read.valid)) and np.array_equal(read.age.numpy(), np.arange(1, 8)[None, None].repeat(layout.users, 1))
    history_obs, age, ports = read.observations, read.age, read.ports   # newest first
    valid = tf.ones([1, layout.users, 7])

    inputs = prepare_cgnn_inputs(receiver, y, ls, active)
    y_real, pe = inputs[0], inputs[1].numpy()  # a constant in the exported graphs, not an input
    observation = model.observe(ls, noise)
    history = PilotHistory(history_obs, age, ports, tf.cast(valid, tf.bool))
    reference_channel = model.prepare(inputs, observation, history, layout)[1]
    reference_llr, _ = model.detect(inputs, noise, history, observation)
    base = model.base

    def network(cgnn, iterations, channel, y_value, active_value):
        if channels_first:
            return channels_first_network(tf, cgnn, iterations, channel, y_value, active_value, pe)
        state = cgnn.initialize_state([y_value, pe, channel, active_value, tf.ones([1, layout.users, 1])])
        for iteration in cgnn._iterations[:iterations]:
            state = iteration([state, pe, active_value])
        return cgnn._readout_llrs[0](state)

    y_spec = tf.TensorSpec(y_real.shape, tf.float32, name="rx_slot")
    h_spec = tf.TensorSpec(ls.shape, tf.float32, name="h_hat_ls")
    a_spec = tf.TensorSpec([1, layout.users], tf.float32, name="active")
    c_spec = tf.TensorSpec(ls.shape, tf.float32, name="channel")
    report = {"slot": {"condition": CONDITION, "setting": SETTING, "episode": EPISODE, "tb": TB,
                       "snr_db": float(snr[SNR_INDEX])},
              "checkpoint": str(sweep.CHECKPOINT)}
    BUILD.mkdir(exist_ok=True)
    suffix = "_cf" if channels_first else ""

    # The two CGNN iterations of the proposed NRX with the refined channel as an input; the filter kernels
    # (fitted_filter.cu) run right before this engine.
    @tf.function(input_signature=[y_spec, c_spec, a_spec])
    def proposed_cgnn(y_value, channel_value, active_value):
        return network(base, 2, channel_value, y_value, active_value)

    outputs = {f"proposed_cgnn{suffix}": proposed_cgnn(y_real, reference_channel, active).numpy()}
    report["proposed_cgnn_check"] = {
        "llr_max_abs_diff": float(np.max(np.abs(outputs[f"proposed_cgnn{suffix}"]-reference_llr.numpy()))),
        "llr_max_abs": float(tf.reduce_max(tf.abs(reference_llr)))}
    tf2onnx.convert.from_function(proposed_cgnn, input_signature=[y_spec, c_spec, a_spec], opset=17,
                                  output_path=str(BUILD/f"proposed_cgnn{suffix}.onnx"))
    feeds = {f"proposed_cgnn{suffix}": {"rx_slot": y_real.numpy(), "channel": reference_channel.numpy(),
                                        "active": active.numpy()},
             f"filter{suffix}": {"h_hat_ls": ls.numpy(), "noise_variance": noise.numpy(), "active": active.numpy(),
                                 "memory_observations": history_obs.numpy(), "memory_age": age.numpy(),
                                 "memory_ports": ports.numpy(), "memory_valid": valid.numpy()}}
    cells = layout.grid_indices
    take = lambda table: np.take_along_axis(table, cells, 1)
    factors = layout.slope_factors
    symbol_cells = np.stack([np.stack([np.flatnonzero(layout.cell_symbol[u] == k) for k in range(layout.dmrs_symbols)])
                             for u in range(layout.users)])
    np.savez(BUILD/f"filter_tables{suffix}.npz", pack=layout.pack_indices.astype(np.int32),
             noise_factors=layout.noise_factors.astype(np.float32),
             neighbor_index=layout.neighbor_indices.astype(np.int32),
             neighbor_valid=layout.neighbor_valid.astype(np.uint8),
             neighbor_delta=layout.neighbor_delta.astype(np.float32),
             cell_symbol=layout.cell_symbol.astype(np.int32), symbol_cells=symbol_cells.astype(np.int32),
             slope_cells=layout.slope_cells.astype(np.int32), cell_of_position=cells.astype(np.int32),
             # a position's offsets within its cell's first-order model, in units of the differences to the
             # neighbouring cells in time and in frequency
             time_factor=(take(factors[..., 0])+take(factors[..., 2])*layout.phi[..., 1]).astype(np.float32),
             frequency_factor=(take(factors[..., 1])+take(factors[..., 3])*layout.phi[..., 2]).astype(np.float32),
             fit_tbs=np.int32(model.channel_filter.fit_tbs), ridge=np.float32(model.channel_filter.ridge),
             reference_channel=reference_channel.numpy())

    weights = {}
    for iterations in (2, 3, 8):
        _, e2e = build_backbone(sweep.CONFIG, iterations, False, num_tx_eval=layout.users,
                                n_size_bwp_override=sweep.PRBS)
        cgnn = e2e._receiver._neural_rx._cgnn

        def make(cgnn, iterations):
            @tf.function(input_signature=[y_spec, h_spec, a_spec])
            def large(y_value, ls_value, active_value):
                return network(cgnn, iterations, ls_value, y_value, active_value)
            return large

        large = make(cgnn, iterations)

        reference, _ = cgnn([y_real, pe, ls, active, tf.ones([1, layout.users, 1])])
        name = f"large_k{iterations}{suffix}"
        outputs[name] = large(y_real, ls, active).numpy()
        report[f"large_k{iterations}_check"] = {"llr_max_abs_diff": float(np.max(np.abs(outputs[name]-reference[-1][0].numpy())))}
        tf2onnx.convert.from_function(large, input_signature=[y_spec, h_spec, a_spec], opset=17,
                                      output_path=str(BUILD/f"{name}.onnx"))
        feeds[name] = {"rx_slot": y_real.numpy(), "h_hat_ls": ls.numpy(), "active": active.numpy()}
        weights[f"large_k{iterations}"] = int(sum(np.prod(v.shape) for v in [*cgnn._s_init[0].weights,
                                                           *[w for it in cgnn._iterations[:iterations] for w in it.weights],
                                                           *cgnn._readout_llrs[0].weights]))
    report["weights"] = weights
    arrays = {f"{name}/input/{key}": value for name, feed in feeds.items() for key, value in feed.items()}
    arrays.update({f"{name}/output": value for name, value in outputs.items()})
    np.savez(BUILD/f"reference{suffix}.npz", **arrays)
    (HERE/"results").mkdir(exist_ok=True)
    (HERE/f"results/export{suffix}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
