"""Export the NRX networks to ONNX for TensorRT timing, as in Wiesmayr et al. (2025).

Scope (as in their code release): the neural network only, with batch size 1, from the received slot and the LS channel
estimate to the LLRs. LS estimation, LDPC decoding, and host-device copies are outside. Networks:

  large_k2, _k3, _k8  released Large NRX (models/pretrained/nrx_large_weights): StateInit, N_it CGNN iterations,
                       LLR readout
  proposed             proposed NRX without IDD: pilot packing, channel filter with the pilot memory, StateInit,
                       two CGNN iterations, LLR readout. Its extra inputs are the noise variance and the memory entries
                       of both UEs as read from the ring buffer (observations, ages, ports, validity).
  proposed_cgnn        the same without the filter: StateInit, two CGNN iterations, and LLR readout of the proposed NRX,
                       with the refined channel as an input. In time_fused.py, the hand-written filter kernel
                       (pilot_filter.cu) produces this input; build/filter_tables.npz holds the kernel's tables, the
                       filter weights, and the TensorFlow receiver's refined channel for the check.

ONNX has no complex type, so the channel filter (ChannelMemory.call with deferred uncertainty, plus
PilotLayout.expand_channel) is rewritten here in real arithmetic. The error covariance is only needed for IDD and is
not part of the network. Each exported function is first checked against the original TensorFlow receiver on one
UMi slot with a full memory.

    export.py      # writes build/*.onnx, build/*.npz, results/export.json (needs a GPU for the native ops)
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWEEP = HERE.parents[1]/"experiments/20261104_tbler_sweep/results"
BUILD = HERE/"build"
EPS = 1.1920928955078125e-7
CONDITION, SETTING, EPISODE, TB, SNR_INDEX = "natural_umi", "sweep", 0, 24, 8


def per_user_gather(tf, values, indices):
    """tf.gather(values[:, u], indices[u], axis=1) for each user u, stacked on axis 1 (avoids batch_dims gathers)."""
    return tf.stack([tf.gather(values[:, u], indices[u], axis=1) for u in range(len(indices))], 1)


def pack(tf, layout, channel, noise):
    """PilotLayout.pack: LS estimates at the distinct pilot positions plus their noise variance."""
    flat = tf.reshape(channel, [1, layout.users, -1, channel.shape[-1]])
    observations = per_user_gather(tf, flat, layout.pack_indices)
    variance = tf.reshape(noise, [-1, 1, 1])*layout.noise_factors[None]
    return tf.concat([observations, variance[..., None]], -1)


def neighbors(tf, layout, records, ports):
    """PilotLayout.neighbors with the batch-dims gather written as a one-hot selection over the users' entries."""
    current = tf.broadcast_to(tf.range(layout.users)[None, :, None], tf.shape(ports))
    addresses = tf.stack([current, ports], -1)
    indices = tf.gather_nd(layout.neighbor_indices, addresses)       # [1, U, E, cells, neighbors]
    valid = tf.gather_nd(layout.neighbor_valid, addresses)
    delta = tf.gather_nd(layout.neighbor_delta, addresses)
    users, entries = records.shape[1], records.shape[2]
    selected = tf.stack([tf.stack([tf.gather(records[0, u, e], indices[0, u, e], axis=0)
                                   for e in range(entries)], 0) for u in range(users)], 0)[None]
    return selected, valid, delta


def filter_real(tf, memory, layout, observation, history, age, ports, valid, active):
    """ChannelMemory.call(defer_uncertainty=True) and the channel expansion of PilotHistoryReceiver.prepare,
    in real arithmetic. Returns the refined channel [1, U, S, T, 2B] that replaces the LS estimate."""
    antennas = (observation.shape[-1]-1)//2
    active_mask = active > .5
    valid = (valid > .5) & active_mask[..., None]
    ports0 = tf.broadcast_to(tf.range(layout.users)[None, :, None], [1, layout.users, 1])
    records = tf.concat([observation[:, :, None], tf.where(valid[..., None, None], history, 0.)], 2)
    ports = tf.concat([ports0, ports], 2)
    ages = tf.concat([tf.zeros_like(age[..., :1]), age], 2)
    valid = tf.concat([active_mask[..., None], valid], 2)
    old, local_valid, delta = neighbors(tf, layout, records, ports)
    central = tf.reduce_all(delta == 0., -1)
    is_current = tf.range(records.shape[2])[None, None, :, None, None] == 0
    allowed = valid[..., None, None] & local_valid & ~(is_current & central)
    old = tf.where(allowed[..., None], old, 0.)

    def flatten(value):
        order = [0, 1, 3, 2, 4]+list(range(5, value.shape.rank))
        value = tf.transpose(value, order)
        return tf.reshape(value, [1, layout.users, layout.count, -1, *value.shape[5:]])

    old, allowed = flatten(old), flatten(allowed)
    h0r, h0i = observation[..., :antennas], observation[..., antennas:2*antennas]
    hjr, hji = old[..., :antennas], old[..., antennas:2*antennas]
    q0, qj = tf.maximum(observation[..., -1], 0.), tf.maximum(old[..., -1], 0.)
    p0 = tf.reduce_mean(h0r*h0r+h0i*h0i, -1)
    pj = tf.reduce_mean(hjr*hjr+hji*hji, -1)
    reference = tf.maximum(p0+q0, 1e-30)
    floor = EPS*reference
    scale = tf.sqrt(tf.maximum(p0[..., None], floor[..., None])*tf.maximum(pj, floor[..., None]))
    corr_r = tf.reduce_mean(h0r[..., None, :]*hjr+h0i[..., None, :]*hji, -1)/scale
    corr_i = tf.reduce_mean(h0r[..., None, :]*hji-h0i[..., None, :]*hjr, -1)/scale
    innov_r = tf.where(allowed[..., None], hjr-h0r[..., None, :], 0.)
    innov_i = tf.where(allowed[..., None], hji-h0i[..., None, :], 0.)
    disagreement = tf.reduce_mean(innov_r*innov_r+innov_i*innov_i, -1)
    age_ms = tf.cast(ages, tf.float32)*layout.tb_seconds*1000.
    coordinates = tf.concat([tf.broadcast_to(tf.math.log(1.+age_ms)[..., None, None, None], tf.shape(delta[..., :1])),
                             delta[..., :1]-age_ms[..., None, None, None], delta[..., 1:]], -1)
    coordinates = flatten(coordinates)
    features = tf.stack([corr_r, corr_i, tf.math.log(tf.maximum(pj+qj, floor[..., None])/reference[..., None]),
                         tf.broadcast_to((q0/reference)[..., None], tf.shape(qj)),
                         qj/tf.maximum(pj+qj, floor[..., None]),
                         disagreement/tf.maximum(reference[..., None]+pj+qj, floor[..., None])], -1)
    geometry = tf.broadcast_to(layout.geometry_features[None, ..., None, :],
                               tf.concat([tf.shape(features)[:-1], [4]], 0))
    scores = memory.filter(tf.concat([features, coordinates, geometry], -1))
    logits = tf.where(allowed, scores[..., 0], -1e9)
    weights = tf.nn.softmax(tf.concat([logits, tf.zeros_like(logits[..., :1])], -1), -1)[..., :-1]
    beta_r = weights[..., None]*tf.tanh(scores[..., 1::2])      # [1, U, cells, neighbors, 3]
    beta_i = weights[..., None]*tf.tanh(scores[..., 2::2])
    # coefficients[..., r, k] = sum_j beta[..., j, k]*innovation[..., j, r] (complex)
    def contract(beta, innovation):
        return tf.reduce_sum(beta[..., :, None, :]*innovation[..., :, :, None], -3)
    coef_r = contract(beta_r, innov_r)-contract(beta_i, innov_i)+h0r[..., None]*tf.constant([1., 0., 0.])
    coef_i = contract(beta_r, innov_i)+contract(beta_i, innov_r)+h0i[..., None]*tf.constant([1., 0., 0.])
    mask = active_mask[..., None, None, None]
    coef_r, coef_i = tf.where(mask, coef_r, 0.), tf.where(mask, coef_i, 0.)
    phi = tf.constant(layout.phi.reshape(layout.users, layout.frequency, layout.symbols, 3))

    def expand(coefficients):
        flat = tf.reshape(coefficients, [1, layout.users, layout.count, antennas*3])
        value = per_user_gather(tf, flat, layout.grid_indices)
        value = tf.reshape(value, [1, layout.users, layout.frequency, layout.symbols, antennas, 3])
        return tf.reduce_sum(value*phi[None, ..., None, :], -1)

    return tf.concat([expand(coef_r), expand(coef_i)], -1)


def main():
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
    past = [model.observe(tf.convert_to_tensor(point[t]["ls_channel"][None], tf.float32),
                          tf.constant([point[t]["noise_variance"]], tf.float32)) for t in range(TB-1, TB-8, -1)]
    history_obs = tf.stack(past, 2)                                  # newest first, as PilotMemoryBuffer.gather
    age = tf.constant(np.arange(1, 8, dtype=np.int32)[None, None].repeat(layout.users, 1))
    ports = tf.constant(np.arange(layout.users, dtype=np.int32)[None, :, None].repeat(7, 2))
    valid = tf.ones([1, layout.users, 7])

    inputs = prepare_cgnn_inputs(receiver, y, ls, active)
    y_real, pe = inputs[0], inputs[1].numpy()  # a constant in the exported graphs, not an input
    observation = model.observe(ls, noise)
    history = PilotHistory(history_obs, age, ports, tf.cast(valid, tf.bool))
    reference_channel = model.prepare(inputs, observation, history, layout)[1]
    reference_llr = model.detect(inputs, noise, history, observation)[0]
    base = model.base

    def network(cgnn, iterations, channel, y_value, active_value):
        state = cgnn.initialize_state([y_value, pe, channel, active_value, tf.ones([1, layout.users, 1])])
        for iteration in cgnn._iterations[:iterations]:
            state = iteration([state, pe, active_value])
        return cgnn._readout_llrs[0](state)

    y_spec = tf.TensorSpec(y_real.shape, tf.float32, name="rx_slot")
    h_spec = tf.TensorSpec(ls.shape, tf.float32, name="h_hat_ls")
    a_spec = tf.TensorSpec([1, layout.users], tf.float32, name="active")
    proposed_specs = [y_spec, h_spec, a_spec, tf.TensorSpec([1], tf.float32, name="noise_variance"),
                      tf.TensorSpec(history_obs.shape, tf.float32, name="memory_observations"),
                      tf.TensorSpec(age.shape, tf.int32, name="memory_age"),
                      tf.TensorSpec(ports.shape, tf.int32, name="memory_ports"),
                      tf.TensorSpec(valid.shape, tf.float32, name="memory_valid")]

    @tf.function(input_signature=proposed_specs)
    def proposed(y_value, ls_value, active_value, noise_value, memory, memory_age, memory_ports, memory_valid):
        packed = pack(tf, layout, ls_value, noise_value)
        channel = filter_real(tf, model.channel_memory, layout, packed, memory, memory_age, memory_ports,
                              memory_valid, active_value)
        return network(base, 2, channel, y_value, active_value)

    @tf.function(input_signature=proposed_specs)
    def proposed_channel(y_value, ls_value, active_value, noise_value, memory, memory_age, memory_ports, memory_valid):
        packed = pack(tf, layout, ls_value, noise_value)
        return filter_real(tf, model.channel_memory, layout, packed, memory, memory_age, memory_ports,
                           memory_valid, active_value)

    proposed_args = (y_real, ls, active, noise, history_obs, age, ports, valid)
    report = {"slot": {"condition": CONDITION, "setting": SETTING, "episode": EPISODE, "tb": TB,
                       "snr_db": float(snr[SNR_INDEX])}}
    channel = proposed_channel(*proposed_args)
    llr = proposed(*proposed_args)
    scale = float(tf.reduce_max(tf.abs(reference_llr)))
    report["proposed_check"] = {
        "channel_max_abs_diff": float(tf.reduce_max(tf.abs(channel-reference_channel))),
        "channel_max_abs": float(tf.reduce_max(tf.abs(reference_channel))),
        "llr_max_abs_diff": float(tf.reduce_max(tf.abs(llr-reference_llr))), "llr_max_abs": scale,
        "llr_sign_agreement": float(tf.reduce_mean(tf.cast(tf.sign(llr) == tf.sign(reference_llr), tf.float32)))}
    if report["proposed_check"]["channel_max_abs_diff"] > 1e-4*report["proposed_check"]["channel_max_abs"]:
        raise RuntimeError(f"real-valued filter differs from the receiver: {report['proposed_check']}")

    BUILD.mkdir(exist_ok=True)
    outputs = {"proposed": llr.numpy()}
    tf2onnx.convert.from_function(proposed, input_signature=proposed_specs, opset=17,
                                  output_path=str(BUILD/"proposed.onnx"))
    feeds = {"proposed": dict(zip([s.name for s in proposed_specs], [v.numpy() for v in proposed_args], strict=True))}

    # The two CGNN iterations of the proposed NRX with the refined channel as an input, for the hand-written filter
    # kernel (pilot_filter.cu) that runs right before this engine; and the kernel's tables and weights.
    c_spec = tf.TensorSpec(ls.shape, tf.float32, name="channel")

    @tf.function(input_signature=[y_spec, c_spec, a_spec])
    def proposed_cgnn(y_value, channel_value, active_value):
        return network(base, 2, channel_value, y_value, active_value)

    outputs["proposed_cgnn"] = proposed_cgnn(y_real, reference_channel, active).numpy()
    report["proposed_cgnn_check"] = {"llr_max_abs_diff": float(np.max(np.abs(outputs["proposed_cgnn"]
                                                                              - reference_llr.numpy())))}
    tf2onnx.convert.from_function(proposed_cgnn, input_signature=[y_spec, c_spec, a_spec], opset=17,
                                  output_path=str(BUILD/"proposed_cgnn.onnx"))
    feeds["proposed_cgnn"] = {"rx_slot": y_real.numpy(), "channel": reference_channel.numpy(),
                              "active": active.numpy()}
    order = np.argsort(layout.grid_indices, axis=1, kind="stable").astype(np.int32)
    offsets = np.stack([np.searchsorted(row[o], np.arange(layout.count+1))
                        for row, o in zip(layout.grid_indices, order, strict=True)]).astype(np.int32)
    dense = model.channel_memory.filter.layers
    np.savez(BUILD/"filter_tables.npz", pack=layout.pack_indices.astype(np.int32),
             noise_factors=layout.noise_factors.astype(np.float32),
             neighbor_index=layout.neighbor_indices.astype(np.int32),
             neighbor_valid=layout.neighbor_valid.astype(np.uint8),
             neighbor_delta=layout.neighbor_delta.astype(np.float32),
             geometry=layout.geometry_features.astype(np.float32),
             weights=np.concatenate([v.numpy().reshape(-1) for v in (dense[0].kernel, dense[0].bias,
                                                                      dense[1].kernel, dense[1].bias)]),
             phi=layout.phi.astype(np.float32), cell_offsets=offsets, cell_positions=order,
             tb_ms=np.float32(layout.tb_seconds*1000.), reference_channel=reference_channel.numpy())

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
        name = f"large_k{iterations}"
        outputs[name] = large(y_real, ls, active).numpy()
        report[f"{name}_check"] = {"llr_max_abs_diff": float(np.max(np.abs(outputs[name]-reference[-1][0].numpy())))}
        tf2onnx.convert.from_function(large, input_signature=[y_spec, h_spec, a_spec], opset=17,
                                      output_path=str(BUILD/f"{name}.onnx"))
        feeds[name] = {"rx_slot": y_real.numpy(), "h_hat_ls": ls.numpy(), "active": active.numpy()}
        weights[name] = int(sum(np.prod(v.shape) for v in [*cgnn._s_init[0].weights,
                                                           *[w for it in cgnn._iterations[:iterations] for w in it.weights],
                                                           *cgnn._readout_llrs[0].weights]))
    report["weights"] = weights
    arrays = {f"{name}/input/{key}": value for name, feed in feeds.items() for key, value in feed.items()}
    arrays.update({f"{name}/output": value for name, value in outputs.items()})
    np.savez(BUILD/"reference.npz", **arrays)
    (HERE/"results").mkdir(exist_ok=True)
    (HERE/"results/export.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
