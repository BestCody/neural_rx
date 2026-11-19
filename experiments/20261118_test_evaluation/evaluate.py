"""Decode the TBLER test sequences with the two receivers trained in November 2026.

Receivers (results/<setting>/<condition>/<name>/episode_NNNNN.npz):
  pilot_memory           the proposed receiver (stage 2), with its memory: the checkpoint named by CHECKPOINT in
                         src/neural_rx/benchmark/tbler_sweep.py
  pilot_memory_emptied   the same weights with the memory emptied before every TB
  neural_rx_k2_trained   neural_rx with two iterations after stage 1, models/neural_rx_k2_20261117
  pilot_memory_current_tb  ablation, not decoded by default: the proposed receiver's weights, but the filter predicts
                         each pilot from the current TB only. Its memory holds just the TBs on which the filter's
                         weights are fitted, so no estimate of an earlier slot enters the prediction; the fitted
                         smoothing within the slot and the first-order model between the pilots remain.
  filter_kbest           comparison, not decoded by default: the proposed receiver's memory and channel filter in
                         front of the classical K-Best detector instead of the two CGNN iterations (filter_kbest.py)
  wiener_kbest           comparison, not decoded by default: the same memory and detector with a classical LMMSE
                         estimator over the current and the stored TBs, from known statistics (wiener.py)
  wiener_slot_kbest      the same estimator on the current TB only, as a check against the sweep's LMMSE + K-Best

The episodes are those every other receiver decoded: the sweep without power control (grids "sweep" and
"sweep_high" of experiments/20261104_tbler_sweep) and the power-controlled episodes of
experiments/20261112_early_retransmission ("power_control"). Each episode is regenerated once on the SNR grid of the
stored Large NRX (N_it=2) result, must have the same fingerprint, and is decoded by all three receivers.

    evaluate.py --gpu 0 --setting sweep --condition natural_umi --first-episode 0 --last-episode 16

A receiver whose result of a sequence exists is not decoded again; --receivers limits the run to some of them.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SWEEP = ROOT/"experiments/20261104_tbler_sweep/results"
PC = ROOT/"experiments/20261112_early_retransmission"
NEURAL_RX = ROOT/"models/neural_rx_k2_20261117/checkpoint_011000"
RECEIVERS = {"pilot_memory": "candidate_persistent", "pilot_memory_emptied": "candidate_reset",
             "neural_rx_k2_trained": "frozen_cold_k2", "pilot_memory_current_tb": "candidate_persistent",
             "filter_kbest": "candidate_persistent", "wiener_kbest": "candidate_persistent",
             "wiener_slot_kbest": "candidate_persistent"}
ESTIMATORS = {"filter_kbest": "filter", "wiener_kbest": "wiener", "wiener_slot_kbest": "wiener_slot"}
DEFAULT = ("pilot_memory", "pilot_memory_emptied", "neural_rx_k2_trained")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--setting", choices=("sweep", "sweep_high", "power_control"), required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--first-episode", type=int, default=0)
    parser.add_argument("--last-episode", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=HERE/"results")
    parser.add_argument("--receivers", nargs="+", choices=tuple(RECEIVERS), default=list(DEFAULT))
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import numpy as np
    import tensorflow as tf
    from neural_rx.backbone import E2E_Model, Parameters
    from neural_rx.backbone.training import load_weights
    from neural_rx.benchmark import tbler_sweep as sweep
    from neural_rx.benchmark.frozen import protocol_and_plan
    from neural_rx.benchmark.runtime import validate_episode
    from neural_rx.provenance.artifacts import file_sha256

    sweep.configure_runtime()
    MEMORY = sweep.CHECKPOINT
    chosen = {name: RECEIVERS[name] for name in args.receivers}
    if args.setting == "power_control":
        sys.path.insert(0, str(PC))
        from power_control import SETTINGS, PowerControlBank
        anchors = json.loads(sweep.ANCHORS.read_text())["conditions"]
        protocol, plan = protocol_and_plan("confirmation", sweep.PRBS, 32, anchors, protocol_path=sweep.PROTOCOL)
        parameters = Parameters(sweep.CONFIG, training=False, num_tx_eval=2, n_size_bwp_override=sweep.PRBS)
        bank = PowerControlBank(parameters, E2E_Model(parameters, training=False, mcs_arr_eval_idx=0), tf, SETTINGS)
        reference = PC/"results/receivers_pc"/args.condition/"frozen_cold_k2"
    else:
        protocol, plan, parameters, bank, anchors = sweep.sweep_context()
        reference = SWEEP/args.setting/args.condition/"frozen_cold_k2"
    base_job = sweep.sweep_job(plan, anchors, args.condition)

    receivers, identity = {}, {}
    for name, method in chosen.items():
        if name == "pilot_memory_current_tb":
            from neural_rx.receiver import model
            settings = model.architecture_settings
            model.architecture_settings = lambda s: dict(settings(s), history_tbs=settings(s)["fit_tbs"])
            try:
                receivers[name] = sweep.make_receiver(method, protocol)
            finally:
                model.architecture_settings = settings
            assert receivers[name].model.history_tbs == receivers[name].model.channel_filter.fit_tbs
        elif name in ESTIMATORS:
            from filter_kbest import FilterKBest
            receivers[name] = FilterKBest(sweep, protocol, tf, ESTIMATORS[name], args.condition, sweep.SCORING_START)
        else:
            receivers[name] = sweep.make_receiver(method, protocol)
        checkpoint = NEURAL_RX if method == "frozen_cold_k2" else MEMORY
        saved = json.loads((checkpoint/"manifest.json").read_text())
        weights = checkpoint/("nrx_weights.pkl" if method == "frozen_cold_k2" else "receiver.weights.h5")
        if file_sha256(weights) != saved["files"][weights.name]:
            raise ValueError(f"{name}: weights differ from the checkpoint manifest")
        if method == "frozen_cold_k2":
            load_weights(receivers[name].model, str(weights))
            receivers[name]._model_hash = receivers[name].model_checksum(receivers[name].model)
        identity[name] = (np.asarray(file_sha256(weights)), np.asarray(saved["update"]))

    for index in range(args.first_episode, args.last_episode):
        name = f"episode_{index:05d}.npz"
        outputs = {r: args.output_dir/args.setting/args.condition/r/name for r in chosen}
        outputs = {r: path for r, path in outputs.items() if not path.is_file()}
        if not outputs:
            continue
        with np.load(reference/name, allow_pickle=False) as data:
            stored = {key: data[key] for key in ("snr_db", "fingerprint")}
        job = dict(base_job, snr_db=[float(value) for value in stored["snr_db"]])
        started = time.perf_counter()
        episode = validate_episode(bank.generate(job, index), protocol, job)
        if not np.array_equal(np.asarray(episode.fingerprint), stored["fingerprint"]):
            raise RuntimeError(f"{args.setting}/{args.condition}/{name}: episode differs from the stored results")
        generated = time.perf_counter()-started
        report = {}
        for receiver, path in outputs.items():
            errors, disagreement = sweep.decode_episode(RECEIVERS[receiver], receivers[receiver], episode, None)
            if receiver in ESTIMATORS:      # like the sweep's stateless receivers: not detected before the scored TBs
                errors[:, :sweep.SCORING_START], disagreement[:, :sweep.SCORING_START] = False, False
            sweep._atomic_savez(path, errors=errors, crc_disagreement=disagreement, active=episode.active,
                                ue_ids=episode.ue_ids, snr_db=stored["snr_db"], fingerprint=stored["fingerprint"],
                                weights_sha256=identity[receiver][0], update=identity[receiver][1])
            report[receiver] = errors[:, sweep.SCORING_START:].sum((1, 2)).tolist()
        print(json.dumps({"setting": args.setting, "condition": args.condition, "episode": index,
                          "generate_s": round(generated, 1), "decode_s": round(time.perf_counter()-started-generated, 1),
                          "errors": report}), flush=True)
    for name, receiver in receivers.items():
        if not receiver.checksums_unchanged():
            raise RuntimeError(f"{name}: receiver weights changed during the run")


if __name__ == "__main__":
    main()
