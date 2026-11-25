"""Test TBLER of every receiver at another radio configuration (MCS), on the conditions and seeds of the 16-QAM tests.

Settings (results/<label>/<setting>/<condition>/<receiver>/episode_NNNNN.npz):
  sweep          no power control: the episodes of the 16-QAM sweep (seeds of experiments/20261104_tbler_sweep), on
                 the SNR grid anchor-6 ... anchor+8 dB in 1 dB steps (the 16-QAM grids "sweep" and "sweep_high" in one)
  power_control  closed-loop power control (experiments/20261112_early_retransmission/power_control.py) on the same
                 episodes, on the 16-QAM power-control grid shifted by this configuration's SNR shift (train.py)
Receivers:
  pilot_memory, pilot_memory_emptied   the proposed receiver (stage 2's best checkpoint), with and without its memory
  neural_rx_k2_trained                 the Large NRX retrained with two iterations (stage 1's best checkpoint)
  frozen_cold_k2, frozen_cold_k3, frozen_cold_k8   the released Large NRX of this MCS with 2, 3, and 8 iterations
  baseline_lslin_lmmse, baseline_lmmse_kbest, baseline_perf_csi_kbest   the classical receivers of the sweep
Each episode is generated once and decoded by every receiver whose result is missing, exactly as in the sweep.

    evaluate.py --config nrx_large_64qam.cfg --setting sweep --condition natural_umi --gpu 0 --first-episode 0 --last-episode 8
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from train import Run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PC = ROOT/"experiments/20261112_early_retransmission"
COVARIANCE = ROOT/"experiments/20261104_tbler_sweep/covariance"
SWEEP_OFFSETS_DB = tuple(float(offset) for offset in range(-6, 9))
PC_GRID_16QAM = (-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 10)  # run_receivers_pc.py
CLASSICAL = ("baseline_lslin_lmmse", "baseline_lmmse_kbest", "baseline_perf_csi_kbest")
RECEIVERS = {"pilot_memory": "candidate_persistent", "pilot_memory_emptied": "candidate_reset",
             "neural_rx_k2_trained": "frozen_cold_k2", "frozen_cold_k2": "frozen_cold_k2",
             "frozen_cold_k3": "frozen_cold_k3", "frozen_cold_k8": "frozen_cold_k8",
             "baseline_lslin_lmmse": "baseline_lslin_lmmse", "baseline_lmmse_kbest": "baseline_lmmse_kbest",
             "baseline_perf_csi_kbest": "baseline_perf_csi_kbest"}


def best_checkpoint(config, stage):
    run = Run(config, stage)
    if not (run.results/"finished.json").is_file():
        raise SystemExit(f"stage {stage} of {config} has not finished")
    return Path(json.loads((run.results/"best.json").read_text())["checkpoint"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--setting", choices=("sweep", "power_control"), required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--first-episode", type=int, default=0)
    parser.add_argument("--last-episode", type=int, default=32)
    parser.add_argument("--receivers", nargs="+", choices=tuple(RECEIVERS), default=list(RECEIVERS))
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if set(args.receivers) & set(CLASSICAL):
        # Two LMMSE + K-Best decoders do not fit into one GPU's memory at 64-QAM: a classical run waits until no other
        # classical run holds this GPU. The lock ends with the process, when its GPU memory is released.
        import fcntl
        (HERE/"logs").mkdir(exist_ok=True)
        lock = open(HERE/"logs"/f".classical_gpu{args.gpu}.lock", "w")
        fcntl.flock(lock, fcntl.LOCK_EX)

    import numpy as np
    import tensorflow as tf
    from neural_rx.backbone import E2E_Model, Parameters
    from neural_rx.backbone.training import load_weights
    from neural_rx.benchmark import tbler_sweep as sweep
    from neural_rx.benchmark.runtime import validate_episode
    from neural_rx.provenance.artifacts import file_sha256

    run = Run(args.config, 1)
    stage2 = best_checkpoint(args.config, 2) if {"pilot_memory", "pilot_memory_emptied"} & set(args.receivers) else None
    sweep.use_radio(args.config, run.anchors, stage2)
    sweep.configure_runtime()
    protocol, plan, parameters, bank, anchors = sweep.sweep_context()
    job = sweep.sweep_job(plan, anchors, args.condition)
    anchor = float(anchors[args.condition]["anchor_snr_db"])
    covariance = sweep.load_covariance(COVARIANCE, args.condition) if "baseline_lmmse_kbest" in args.receivers else None
    if args.setting == "sweep":
        job = dict(job, snr_db=[anchor+offset for offset in SWEEP_OFFSETS_DB])
    else:
        sys.path.insert(0, str(PC))
        from power_control import SETTINGS, PowerControlBank
        bank = PowerControlBank(parameters, E2E_Model(parameters, training=False, mcs_arr_eval_idx=0), tf, SETTINGS)
        job = dict(job, snr_db=[value+run.snr_shift() for value in PC_GRID_16QAM])
        if covariance is not None:       # received power steered to the target, as in run_receivers_pc.py
            covariance["space"] = covariance["space"]/covariance["power"]
            covariance["power"] = 1.

    receivers, identity = {}, {}
    for name in args.receivers:
        receivers[name] = sweep.make_receiver(RECEIVERS[name], protocol, covariance)
        if name == "neural_rx_k2_trained":
            weights = best_checkpoint(args.config, 1)/"nrx_weights.pkl"
            load_weights(receivers[name].model, str(weights))
            receivers[name]._model_hash = receivers[name].model_checksum(receivers[name].model)
            identity[name] = file_sha256(weights)
        elif name.startswith("pilot_memory"):
            identity[name] = file_sha256(stage2/"receiver.weights.h5")
    precoder = parameters.transmitters[0]._precoder._w
    output = HERE/"results"/run.label/args.setting/args.condition
    for index in range(args.first_episode, args.last_episode):
        name = f"episode_{index:05d}.npz"
        pending = [r for r in args.receivers if not (output/r/name).is_file()]
        if not pending:
            continue
        started = time.perf_counter()
        episode = validate_episode(bank.generate(job, index), protocol, job)
        channel = bank.last_channel
        generated = time.perf_counter()-started
        slots = np.asarray([int(tb["slot_number"]) for tb in episode.receiver_inputs[0]], np.int32)
        power = sweep.effective_power(channel, precoder)
        report = {}
        for receiver in pending:
            method = RECEIVERS[receiver]
            errors, disagreement = sweep.decode_episode(method, receivers[receiver], episode,
                                                        channel if method == "baseline_perf_csi_kbest" else None)
            sweep._atomic_savez(output/receiver/name, errors=errors, crc_disagreement=disagreement,
                                active=episode.active, ue_ids=episode.ue_ids, snr_db=np.asarray(job["snr_db"]),
                                slot_number=slots, channel_power=power, fingerprint=np.asarray(episode.fingerprint),
                                **({"weights_sha256": np.asarray(identity[receiver])} if receiver in identity else {}))
            report[receiver] = errors[:, sweep.SCORING_START:].sum((1, 2)).tolist()
        print(json.dumps({"config": args.config, "setting": args.setting, "condition": args.condition,
                          "episode": index, "generate_s": round(generated, 1),
                          "decode_s": round(time.perf_counter()-started-generated, 1), "errors": report}), flush=True)
    for name, receiver in receivers.items():
        if RECEIVERS[name] in sweep.NEURAL and not receiver.checksums_unchanged():
            raise RuntimeError(f"{name}: receiver weights changed during the run")


if __name__ == "__main__":
    main()
