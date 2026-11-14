"""Decode the TBLER sweep's episodes again with the proposed NRX and log the evidence of every TB.

The episodes, SNR grid, and receiver are those of experiments/20261104_tbler_sweep, so each episode's fingerprint,
error flags, and CRC disagreements must equal the stored sweep; any difference stops the run.

    log_evidence.py --gpu 0 --condition natural_umi --reference-dir ../20261104_tbler_sweep/results/sweep
    log_evidence.py --gpu 1 --condition schedule_gap_reentry --snr-offsets 3 4 5 6 7 8 \
        --reference-dir ../20261104_tbler_sweep/results/sweep_high
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--reference-dir", type=Path,
                        help="stored sweep whose fingerprints and decisions must be reproduced exactly")
    parser.add_argument("--snr-offsets", type=float, nargs="+")
    parser.add_argument("--snr-db", type=float, nargs="+", help="absolute SNR grid instead of anchor offsets")
    parser.add_argument("--power-control", action="store_true", help="closed-loop power control (power_control.py)")
    parser.add_argument("--power-range-db", type=float, nargs=2, help="override the power-control range")
    parser.add_argument("--first-episode", type=int, default=0)
    parser.add_argument("--last-episode", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=HERE/"results/evidence")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import numpy as np
    import tensorflow as tf
    sys.path.insert(0, str(HERE))
    from evidence_adapter import FEATURES, EvidenceAdapter
    from neural_rx.benchmark import tbler_sweep as sweep
    from neural_rx.benchmark.frozen import make_manifest, resolve_manifest_artifacts
    from neural_rx.benchmark.runtime import validate_episode
    from neural_rx.benchmark.scoring import payload_error_flags

    # The same set-up order as tbler_sweep.run_sweep and tbler_sweep.sweep_context.
    sweep.configure_runtime()
    if args.power_control:
        from neural_rx.backbone import E2E_Model, Parameters
        from neural_rx.benchmark.frozen import protocol_and_plan
        from power_control import SETTINGS, PowerControlBank
        settings = dict(SETTINGS)
        if args.power_range_db:
            settings.update(minimum_db=args.power_range_db[0], maximum_db=args.power_range_db[1])
        anchors = json.loads(sweep.ANCHORS.read_text())["conditions"]
        protocol, plan = protocol_and_plan("confirmation", sweep.PRBS, 32, anchors, protocol_path=sweep.PROTOCOL)
        parameters = Parameters(sweep.CONFIG, training=False, num_tx_eval=2, n_size_bwp_override=sweep.PRBS)
        bank = PowerControlBank(parameters, E2E_Model(parameters, training=False, mcs_arr_eval_idx=0), tf, settings)
        print(json.dumps({"power_control": settings}), flush=True)
    else:
        protocol, plan, parameters, bank, anchors = sweep.sweep_context()
    job = sweep.sweep_job(plan, anchors, args.condition)
    if args.snr_offsets:
        anchor = float(anchors[args.condition]["anchor_snr_db"])
        job = dict(job, snr_db=[anchor+offset for offset in args.snr_offsets])
    if args.snr_db:
        job = dict(job, snr_db=[float(value) for value in args.snr_db])
    precoder = parameters.transmitters[0]._precoder._w
    manifest, mapping, seed = make_manifest(protocol, "candidate", sweep.CHECKPOINT)
    receiver = EvidenceAdapter(manifest=manifest, method="candidate_persistent", training_seed=seed,
                               artifacts=resolve_manifest_artifacts(manifest, mapping),
                               config_name=sweep.CONFIG, tf_module=tf)
    output = args.output_dir/args.condition
    output.mkdir(parents=True, exist_ok=True)
    for index in range(args.first_episode, args.last_episode):
        path = output/f"episode_{index:05d}.npz"
        if path.is_file():
            continue
        started = time.perf_counter()
        episode = validate_episode(bank.generate(job, index), protocol, job)
        stored = None
        if args.reference_dir:
            with np.load(args.reference_dir/args.condition/"candidate_persistent"/path.name, allow_pickle=False) as data:
                stored = {name: data[name] for name in data.files}
            if not (np.array_equal(np.asarray(episode.fingerprint), stored["fingerprint"])
                    and np.array_equal(np.asarray(job["snr_db"]), stored["snr_db"])):
                raise RuntimeError(f"{args.condition}/{path.name}: episode or SNR grid differs from the stored sweep")
        power = sweep.effective_power(bank.last_channel, precoder)
        gain = getattr(bank, "last_power_gain_db", np.zeros(episode.active.shape))
        generated = time.perf_counter()-started
        points = len(episode.receiver_inputs)
        tbs, users = episode.active.shape
        errors = np.zeros((points, tbs, users), bool)
        disagreement = np.zeros((points, tbs, users), bool)
        evidence = np.zeros((points, tbs, users, len(FEATURES)))
        noise = np.zeros((points, tbs))
        receiver.start_episode_batch(points)
        for tb in range(tbs):
            inputs = [episode.receiver_inputs[point][tb] for point in range(points)]
            result = receiver.decode_tb_batch(inputs)
            evidence[:, tb] = receiver.last_evidence
            noise[:, tb] = [float(value["noise_variance"]) for value in inputs]
            truth = np.broadcast_to(episode.bits[tb], (points, *episode.bits[tb].shape))
            active = np.broadcast_to(episode.active[tb], (points, users))
            errors[:, tb], disagreement[:, tb] = payload_error_flags(
                np.asarray(result["hard_bits"]), np.asarray(result["crc_status"]), truth, active)
        if stored is not None:
            if not (np.array_equal(errors, stored["errors"])
                    and np.array_equal(disagreement, stored["crc_disagreement"])
                    and np.allclose(power, stored["channel_power"], rtol=1e-6, atol=0)):
                changed = int((errors != stored["errors"]).sum())
                raise RuntimeError(f"{args.condition}/{path.name}: {changed} error flags differ from the stored sweep")
        if not receiver.checksums_unchanged():
            raise RuntimeError("receiver artifacts changed during the run")
        slots = np.asarray([int(tb["slot_number"]) for tb in episode.receiver_inputs[0]], np.int32)
        sweep._atomic_savez(path, errors=errors, crc_disagreement=disagreement, evidence=evidence,
                            features=np.asarray(FEATURES), noise_variance=noise, active=episode.active,
                            ue_ids=episode.ue_ids, channel_change_event=episode.channel_change_event,
                            snr_db=np.asarray(job["snr_db"]), slot_number=slots, channel_power=power,
                            power_gain_db=gain, fingerprint=np.asarray(episode.fingerprint))
        print(json.dumps({"condition": args.condition, "episode": index, "generate_s": round(generated, 2),
                          "decode_s": round(time.perf_counter()-started-generated, 2),
                          "matches_sweep": None if stored is None else True,
                          "errors": errors[:, sweep.SCORING_START:].sum((1, 2)).tolist()}), flush=True)


if __name__ == "__main__":
    main()
