"""Decode the power-controlled episodes with the comparison receivers, as in the TBLER sweep.

The episodes are those of log_evidence.py --power-control (power_control.py), on the same SNR grid; each episode's
fingerprint must equal the logged proposed NRX's. With power control, every UE's received power is steered to the
0 dB target, so the LMMSE estimator's spatial covariance is scaled to unit power (as in the Double-TDL validation
of experiments/20261104_tbler_sweep).

    run_receivers_pc.py --gpu 0 --condition natural_umi --methods frozen_cold_k8 baseline_lmmse_kbest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRID = (-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--first-episode", type=int, default=0)
    parser.add_argument("--last-episode", type=int, default=32)
    parser.add_argument("--evidence-dir", type=Path, default=HERE/"results/evidence_pc")
    parser.add_argument("--output-dir", type=Path, default=HERE/"results/receivers_pc")
    parser.add_argument("--covariance-dir", type=Path, default=HERE.parent/"20261104_tbler_sweep/covariance")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import numpy as np
    import tensorflow as tf
    sys.path.insert(0, str(HERE))
    from power_control import SETTINGS, PowerControlBank
    from neural_rx.backbone import E2E_Model, Parameters
    from neural_rx.benchmark import tbler_sweep as sweep
    from neural_rx.benchmark.frozen import protocol_and_plan
    from neural_rx.benchmark.runtime import validate_episode

    sweep.configure_runtime()
    anchors = json.loads(sweep.ANCHORS.read_text())["conditions"]
    protocol, plan = protocol_and_plan("confirmation", sweep.PRBS, 32, anchors, protocol_path=sweep.PROTOCOL)
    parameters = Parameters(sweep.CONFIG, training=False, num_tx_eval=2, n_size_bwp_override=sweep.PRBS)
    bank = PowerControlBank(parameters, E2E_Model(parameters, training=False, mcs_arr_eval_idx=0), tf, SETTINGS)
    job = dict(sweep.sweep_job(plan, anchors, args.condition), snr_db=[float(value) for value in GRID])
    covariance = None
    if "baseline_lmmse_kbest" in args.methods:
        covariance = sweep.load_covariance(args.covariance_dir, args.condition)
        covariance["space"] = covariance["space"]/covariance["power"]
        covariance["power"] = 1.
    receivers = {method: sweep.make_receiver(method, protocol, covariance) for method in args.methods}
    precoder = parameters.transmitters[0]._precoder._w
    for index in range(args.first_episode, args.last_episode):
        name = f"episode_{index:05d}.npz"
        pending = [m for m in args.methods if not (args.output_dir/args.condition/m/name).is_file()]
        if not pending:
            continue
        reference = args.evidence_dir/args.condition/name
        if not reference.is_file():
            raise RuntimeError(f"log the proposed NRX on {args.condition}/{name} first")
        started = time.perf_counter()
        episode = validate_episode(bank.generate(job, index), protocol, job)
        channel = bank.last_channel
        with np.load(reference, allow_pickle=False) as data:
            if not (np.array_equal(np.asarray(episode.fingerprint), data["fingerprint"])
                    and np.array_equal(np.asarray(job["snr_db"]), data["snr_db"])):
                raise RuntimeError(f"{args.condition}/{name}: episode differs from the logged proposed NRX")
        generated = time.perf_counter()-started
        power = sweep.effective_power(channel, precoder)
        slots = np.asarray([int(tb["slot_number"]) for tb in episode.receiver_inputs[0]], np.int32)
        for method in pending:
            started = time.perf_counter()
            errors, disagreement = sweep.decode_episode(
                method, receivers[method], episode, channel if method == "baseline_perf_csi_kbest" else None)
            sweep._atomic_savez(args.output_dir/args.condition/method/name, errors=errors,
                                crc_disagreement=disagreement, active=episode.active, ue_ids=episode.ue_ids,
                                snr_db=np.asarray(job["snr_db"]), slot_number=slots, channel_power=power,
                                power_gain_db=bank.last_power_gain_db, fingerprint=np.asarray(episode.fingerprint))
            print(json.dumps({"condition": args.condition, "episode": index, "method": method,
                              "generate_s": round(generated, 2), "decode_s": round(time.perf_counter()-started, 2),
                              "errors": errors[:, sweep.SCORING_START:].sum((1, 2)).tolist()}), flush=True)
    for method, receiver in receivers.items():
        if method in sweep.NEURAL and not receiver.checksums_unchanged():
            raise RuntimeError(f"{method} artifacts changed during the run")


if __name__ == "__main__":
    main()
