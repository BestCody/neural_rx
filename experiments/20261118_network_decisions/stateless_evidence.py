"""Decode the power-controlled test sequences again with the stateless Large NRX retrained with two iterations
(stage 1, models/neural_rx_k2_20261117) and log the same per-TB evidence as log_evidence.py logs for the proposed NRX.

The evidence has the same definitions (evidence_adapter.FEATURES): the LS channel power and noise variance from the
same pilot packing, the bit error rate implied by the LLRs, and the parity checks after the first 8 BP iterations.
This NRX has no memory and no channel filter, so the memory features are zero and the power of its channel estimate
is that of the LS estimate. Each sequence's error flags must equal the stored test evaluation of this NRX
(experiments/20261118_test_evaluation, "power_control"); any difference stops the run.

    stateless_evidence.py --gpu 1 --condition natural_umi --first-episode 0 --last-episode 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NEURAL_RX = ROOT/"models/neural_rx_k2_20261117/checkpoint_011000"


def stateless_adapter(sweep, protocol, tf, np):
    from neural_rx.backbone.training import load_weights
    from neural_rx.benchmark.candidate import checked_decoder_tensors
    from neural_rx.decoding import SoftTBInformationDecoder
    from neural_rx.provenance.artifacts import file_sha256
    from neural_rx.receiver.pilot_layout import PilotLayout
    from evidence_adapter import FIRST_BP_ITERATIONS
    import scipy.sparse as sparse

    receiver = sweep.make_receiver("frozen_cold_k2", protocol)
    saved = json.loads((NEURAL_RX/"manifest.json").read_text())
    weights = NEURAL_RX/"nrx_weights.pkl"
    if file_sha256(weights) != saved["files"][weights.name]:
        raise ValueError("weights differ from the checkpoint manifest")
    load_weights(receiver.model, str(weights))
    receiver._model_hash = receiver.model_checksum(receiver.model)
    rx = receiver.receiver
    layout = PilotLayout(rx._ls_est, receiver.model._transmitters[0].resource_grid, 1)
    first = SoftTBInformationDecoder(rx._tb_encoders[0], rx._tb_decoders[0], FIRST_BP_ITERATIONS, return_infobits=False)
    parity = sparse.csr_matrix(rx._tb_decoders[0]._decoder.pcm, dtype=np.int32)

    # receiver._build_graph's decoding path, unchanged, plus the evidence
    @tf.function(reduce_retracing=True, jit_compile=False)
    def decode(y, channel, active, noise):
        inputs = receiver.prepare_inputs(rx, y, channel, active)
        state = receiver.backbone.initialize_state(inputs)
        for index in range(receiver.iterations):
            state = receiver.backbone._iterations[index]([state, inputs[1], inputs[3]])
        grid = receiver.readout(receiver.backbone, state)
        llr = receiver.grid_map.grid_to_coded(grid)
        logits, hard, crc = checked_decoder_tensors(receiver.decoder(llr))
        with tf.control_dependencies([logits, hard, crc]):
            observation = layout.pack(channel, noise)
            ls_power = tf.reduce_mean(tf.reduce_sum(tf.square(observation[..., :-1]), -1), -1)
            ls_noise = tf.reduce_mean(observation[..., -1], -1)
            posterior = first.decoder(first.to_code_blocks(llr))
            magnitude, posterior_magnitude = tf.abs(llr), tf.abs(first.from_code_blocks(posterior))
            zero = tf.zeros_like(ls_power)
            evidence = tf.stack([zero, zero, zero, zero, zero, ls_power, ls_noise, ls_power,
                                 tf.reduce_mean(magnitude, -1), tf.reduce_mean(tf.sigmoid(-magnitude), -1),
                                 tf.reduce_mean(tf.sigmoid(-posterior_magnitude), -1)], -1)
        return logits, hard, crc, evidence, tf.cast(posterior > 0., tf.int8)

    def run(inputs):
        from neural_rx.benchmark.candidate import stack_receiver_grids
        y, channel = stack_receiver_grids(inputs, tf)
        active = tf.convert_to_tensor(np.stack([point["active"] for point in inputs]), tf.bool)
        noise = tf.convert_to_tensor(np.asarray([point["noise_variance"] for point in inputs], np.float32))
        _, hard, crc, evidence, bits = decode(y, channel, active, noise)
        bits = bits.numpy()
        unsatisfied = ((parity @ bits.reshape(-1, bits.shape[-1]).astype(np.int32).T) % 2).sum(0).reshape(bits.shape[:-1])
        evidence = np.concatenate([evidence.numpy(), unsatisfied.sum(-1)[..., None], (unsatisfied > 0).sum(-1)[..., None]],
                                  -1).astype(np.float64)
        return np.asarray(hard.numpy(), bool), np.asarray(crc.numpy(), bool), evidence

    return receiver, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--reference-dir", type=Path, default=HERE.parent/"20261118_test_evaluation/results/power_control")
    parser.add_argument("--first-episode", type=int, default=0)
    parser.add_argument("--last-episode", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=HERE/"results/evidence_pc_stateless")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import numpy as np
    import tensorflow as tf
    sys.path[:0] = [str(HERE), str(HERE.parent/"20261112_early_retransmission")]
    from evidence_adapter import FEATURES
    from neural_rx.benchmark import tbler_sweep as sweep
    from neural_rx.benchmark.runtime import validate_episode
    from neural_rx.benchmark.scoring import payload_error_flags

    sweep.configure_runtime()
    from neural_rx.backbone import E2E_Model, Parameters
    from neural_rx.benchmark.frozen import protocol_and_plan
    from power_control import SETTINGS, PowerControlBank
    anchors = json.loads(sweep.ANCHORS.read_text())["conditions"]
    protocol, plan = protocol_and_plan("confirmation", sweep.PRBS, 32, anchors, protocol_path=sweep.PROTOCOL)
    parameters = Parameters(sweep.CONFIG, training=False, num_tx_eval=2, n_size_bwp_override=sweep.PRBS)
    bank = PowerControlBank(parameters, E2E_Model(parameters, training=False, mcs_arr_eval_idx=0), tf, dict(SETTINGS))
    base_job = sweep.sweep_job(plan, anchors, args.condition)
    precoder = parameters.transmitters[0]._precoder._w
    receiver, decode = stateless_adapter(sweep, protocol, tf, np)
    output = args.output_dir/args.condition
    output.mkdir(parents=True, exist_ok=True)
    for index in range(args.first_episode, args.last_episode):
        path = output/f"episode_{index:05d}.npz"
        if path.is_file():
            continue
        started = time.perf_counter()
        with np.load(args.reference_dir/args.condition/"neural_rx_k2_trained"/path.name, allow_pickle=False) as data:
            stored = {name: data[name] for name in data.files}
        job = dict(base_job, snr_db=[float(value) for value in stored["snr_db"]])
        episode = validate_episode(bank.generate(job, index), protocol, job)
        if not np.array_equal(np.asarray(episode.fingerprint), stored["fingerprint"]):
            raise RuntimeError(f"{args.condition}/{path.name}: sequence differs from the stored evaluation")
        power = sweep.effective_power(bank.last_channel, precoder)
        gain = getattr(bank, "last_power_gain_db", np.zeros(episode.active.shape))
        generated = time.perf_counter()-started
        points = len(episode.receiver_inputs)
        tbs, users = episode.active.shape
        errors = np.zeros((points, tbs, users), bool)
        disagreement = np.zeros((points, tbs, users), bool)
        evidence = np.zeros((points, tbs, users, len(FEATURES)))
        noise = np.zeros((points, tbs))
        for tb in range(tbs):                     # all TBs, so the CRC history before the scored TBs is real too
            inputs = [episode.receiver_inputs[point][tb] for point in range(points)]
            hard, crc, evidence[:, tb] = decode(inputs)
            noise[:, tb] = [float(value["noise_variance"]) for value in inputs]
            truth = np.broadcast_to(episode.bits[tb], (points, *episode.bits[tb].shape))
            active = np.broadcast_to(episode.active[tb], (points, users))
            errors[:, tb], disagreement[:, tb] = payload_error_flags(hard, crc, truth, active)
        start = sweep.SCORING_START
        if not (np.array_equal(errors[:, start:], stored["errors"][:, start:])
                and np.array_equal(disagreement[:, start:], stored["crc_disagreement"][:, start:])):
            changed = int((errors[:, start:] != stored["errors"][:, start:]).sum())
            raise RuntimeError(f"{args.condition}/{path.name}: {changed} error flags differ from the stored evaluation")
        if not receiver.checksums_unchanged():
            raise RuntimeError("receiver weights changed during the run")
        slots = np.asarray([int(tb["slot_number"]) for tb in episode.receiver_inputs[0]], np.int32)
        sweep._atomic_savez(path, errors=errors, crc_disagreement=disagreement, evidence=evidence,
                            features=np.asarray(FEATURES), noise_variance=noise, active=episode.active,
                            ue_ids=episode.ue_ids, channel_change_event=episode.channel_change_event,
                            snr_db=np.asarray(job["snr_db"]), slot_number=slots, channel_power=power,
                            power_gain_db=gain, fingerprint=np.asarray(episode.fingerprint))
        print(json.dumps({"condition": args.condition, "episode": index, "generate_s": round(generated, 2),
                          "decode_s": round(time.perf_counter()-started-generated, 2), "matches_evaluation": True,
                          "errors": errors[:, start:].sum((1, 2)).tolist()}), flush=True)


if __name__ == "__main__":
    main()
