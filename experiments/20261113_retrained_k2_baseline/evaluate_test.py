"""Decode the TBLER test sequences with the retrained two-iteration baseline.

The episodes are those every other receiver decoded: the sweep without power control (grids "sweep" and
"sweep_high" of experiments/20261104_tbler_sweep) and the power-controlled episodes of
experiments/20261112_early_retransmission ("power_control"). Each episode is regenerated on the SNR grid of the stored
Large NRX (N_it=2) result and must have the same fingerprint. The receiver is the benchmark's two-iteration Large NRX
adapter with the weights of a retrained checkpoint (nrx_weights.pkl); given the untrained checkpoint 0, whose weights
equal the shipped ones, it must reproduce the stored N_it=2 errors exactly (--compare).

    evaluate_test.py --gpu 0 --setting sweep --condition natural_umi --checkpoint models/.../checkpoint_0NNNNN
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--setting", choices=("sweep", "sweep_high", "power_control"), required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--first-episode", type=int, default=0)
    parser.add_argument("--last-episode", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=HERE/"results/test")
    parser.add_argument("--compare", action="store_true", help="Require the stored N_it=2 errors (checkpoint 0 only)")
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

    checkpoint = args.checkpoint.resolve()
    saved = json.loads((checkpoint/"manifest.json").read_text())
    weights = checkpoint/"nrx_weights.pkl"
    if file_sha256(weights) != saved["files"]["nrx_weights.pkl"]:
        raise ValueError("Retrained weights differ from the checkpoint manifest")
    if args.compare and saved["update"] != 0:
        raise ValueError("Only the untrained checkpoint reproduces the stored N_it=2 errors")
    receiver = sweep.make_receiver("frozen_cold_k2", protocol)
    load_weights(receiver.model, str(weights))
    receiver._model_hash = receiver.model_checksum(receiver.model)

    for index in range(args.first_episode, args.last_episode):
        name = f"episode_{index:05d}.npz"
        output = args.output_dir/args.setting/args.condition/"retrained_k2"/name
        if output.is_file():
            continue
        with np.load(reference/name, allow_pickle=False) as data:
            stored = {key: data[key] for key in ("errors", "snr_db", "fingerprint")}
        job = dict(base_job, snr_db=[float(value) for value in stored["snr_db"]])
        started = time.perf_counter()
        episode = validate_episode(bank.generate(job, index), protocol, job)
        if not np.array_equal(np.asarray(episode.fingerprint), stored["fingerprint"]):
            raise RuntimeError(f"{args.setting}/{args.condition}/{name}: episode differs from the stored results")
        generated = time.perf_counter()-started
        errors, disagreement = sweep.decode_episode("frozen_cold_k2", receiver, episode, None)
        if args.compare and not np.array_equal(errors, stored["errors"]):
            raise RuntimeError(f"{name}: {int((errors != stored['errors']).sum())} error flags differ from N_it=2")
        sweep._atomic_savez(output, errors=errors, crc_disagreement=disagreement, active=episode.active,
                            ue_ids=episode.ue_ids, snr_db=stored["snr_db"], fingerprint=stored["fingerprint"],
                            weights_sha256=np.asarray(file_sha256(weights)), update=np.asarray(saved["update"]))
        print(json.dumps({"setting": args.setting, "condition": args.condition, "episode": index,
                          "generate_s": round(generated, 2), "decode_s": round(time.perf_counter()-started-generated, 2),
                          "errors": errors[:, sweep.SCORING_START:].sum((1, 2)).tolist(),
                          "matches_stored_n_it_2": True if args.compare else None}), flush=True)
    if not receiver.checksums_unchanged():
        raise RuntimeError("receiver weights changed during the run")


if __name__ == "__main__":
    main()
