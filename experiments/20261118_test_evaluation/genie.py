"""How much is left to gain from a better channel estimate? UMi test sequences decoded with the true channel given
to the two-iteration networks in place of their channel estimate: (a) the retrained Large NRX (stage 1), with the
true channel instead of its LS estimate, and (b) the proposed NRX's network (stage 2), with the true channel instead
of the output of its channel filter. Neither network was trained for this input, so these are lower bounds on what a
perfect estimate would give. Sequences and SNR grid are those of the stored sweep ("sweep" grid).

    genie.py --gpu 0 --first-episode 0 --last-episode 16      # results/genie/episode_NNNNN.npz
"""
import argparse, json, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SWEEP = ROOT/"experiments/20261104_tbler_sweep/results"
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--first-episode", type=int, default=0)
parser.add_argument("--last-episode", type=int, default=32)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from neural_rx.backbone.core import prepare_cgnn_inputs
from neural_rx.backbone.training import load_weights
from neural_rx.benchmark import tbler_sweep as sweep
from neural_rx.benchmark.runtime import validate_episode
from neural_rx.benchmark.scoring import payload_error_flags

sweep.configure_runtime()
protocol, plan, parameters, bank, anchors = sweep.sweep_context()
condition, setting = "natural_umi", "sweep"
base_job = sweep.sweep_job(plan, anchors, condition)
stage1 = sweep.make_receiver("frozen_cold_k2", protocol)
load_weights(stage1.model, str(ROOT/"models/neural_rx_k2_20261117/checkpoint_011000/nrx_weights.pkl"))
stage1._model_hash = stage1.model_checksum(stage1.model)
ours = sweep.make_receiver("candidate_persistent", protocol)


@tf.function(reduce_retracing=True)
def with_true_channel(y, truth, active):
    inputs = prepare_cgnn_inputs(ours.receiver, y, truth, active)
    grid, _ = ours.model.iterate(inputs)
    output = ours.decoder(ours.grid_map.grid_to_coded(grid))
    return output.hard_bits, output.crc_status


output = HERE/"results/genie"
output.mkdir(parents=True, exist_ok=True)
for index in range(args.first_episode, args.last_episode):
    name = f"episode_{index:05d}.npz"
    if (output/name).is_file():
        continue
    with np.load(SWEEP/setting/condition/"frozen_cold_k2"/name) as data:
        snr, fingerprint = data["snr_db"], data["fingerprint"]
    with np.load(HERE/"results"/setting/condition/"neural_rx_k2_trained"/name) as data:
        stored = data["errors"]
    job = dict(base_job, snr_db=[float(v) for v in snr])
    episode = validate_episode(bank.generate(job, index), protocol, job)
    assert np.array_equal(np.asarray(episode.fingerprint), fingerprint)
    channel = bank.last_channel
    errors_ls, _ = sweep.decode_episode("frozen_cold_k2", stage1, episode, None)
    same = bool(np.array_equal(errors_ls, stored))
    points, tbs = len(episode.receiver_inputs), len(episode.receiver_inputs[0])
    errors_ours = np.zeros_like(errors_ls)
    for tb in range(tbs):
        true = stage1.receiver.preprocess_channel_ground_truth(channel[0][tb:tb+1])[0].numpy()
        inputs = [episode.receiver_inputs[point][tb] for point in range(points)]
        if tb >= sweep.SCORING_START:
            y = tf.constant(np.stack([v["received_grid"] for v in inputs]))
            truth = tf.constant(np.repeat(true[None], points, 0).astype(np.float32))
            active = tf.constant(np.repeat(episode.active[tb][None], points, 0).astype(np.float32))
            hard, crc = with_true_channel(y, truth, active)
            errors_ours[:, tb], _ = payload_error_flags(np.asarray(hard), np.asarray(crc),
                np.broadcast_to(episode.bits[tb], (points, *episode.bits[tb].shape)), np.asarray(active).astype(bool))
        for value in inputs:
            assert value["ls_channel"].shape == true.shape
            value["ls_channel"] = true.astype(value["ls_channel"].dtype)
    errors_stage1, _ = sweep.decode_episode("frozen_cold_k2", stage1, episode, None)
    np.savez(output/name, errors_stage1_true=errors_stage1, errors_ours_true=errors_ours, snr_db=snr,
             active=episode.active, fingerprint=fingerprint)
    print(json.dumps({"episode": index, "stage1_ls_matches_stored": same,
                      "errors": {"stage 1, LS": errors_ls[:, 16:].sum((1, 2)).tolist(),
                                 "stage 1, true channel": errors_stage1[:, 16:].sum((1, 2)).tolist(),
                                 "ours, true channel": errors_ours[:, 16:].sum((1, 2)).tolist()}}), flush=True)
