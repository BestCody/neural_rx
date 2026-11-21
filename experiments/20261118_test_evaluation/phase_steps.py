"""Phase steps per slot and pool UE of the power_control_phase sequences (evaluate.py --setting power_control_phase).

The decoded results store the phase steps only for the two scheduled positions of each slot, but with scheduling gaps
a UE's power step can fall in a slot in which it is not scheduled. This regenerates each sequence without decoding,
checks that it equals the decoded one, and saves the steps of all pool UEs, so that phase_jumps_summary.py can tell
for each scheduled TB whether its UE's phase changed since its previous TBs.

    phase_steps.py --gpu 0 --condition schedule_gap_reentry --first-episode 0 --last-episode 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PC = HERE.parent/"20261112_early_retransmission"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--first-episode", type=int, default=0)
    parser.add_argument("--last-episode", type=int, default=32)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import numpy as np
    import tensorflow as tf
    from neural_rx.backbone import E2E_Model, Parameters
    from neural_rx.benchmark import tbler_sweep as sweep
    from neural_rx.benchmark.frozen import protocol_and_plan
    from neural_rx.benchmark.runtime import validate_episode

    sweep.configure_runtime()
    sys.path.insert(0, str(PC))
    from power_control import SETTINGS
    from phase_jumps import PhaseJumpBank
    anchors = json.loads(sweep.ANCHORS.read_text())["conditions"]
    protocol, plan = protocol_and_plan("confirmation", sweep.PRBS, 32, anchors, protocol_path=sweep.PROTOCOL)
    parameters = Parameters(sweep.CONFIG, training=False, num_tx_eval=2, n_size_bwp_override=sweep.PRBS)
    bank = PhaseJumpBank(parameters, E2E_Model(parameters, training=False, mcs_arr_eval_idx=0), tf, SETTINGS)
    base_job = sweep.sweep_job(plan, anchors, args.condition)
    decoded = HERE/"results/power_control_phase"/args.condition/"pilot_memory"
    output = HERE/"results/power_control_phase_steps"/args.condition
    output.mkdir(parents=True, exist_ok=True)
    for index in range(args.first_episode, args.last_episode):
        name = f"episode_{index:05d}.npz"
        if (output/name).is_file():
            continue
        with np.load(decoded/name, allow_pickle=False) as data:
            stored = {key: data[key] for key in ("snr_db", "fingerprint", "ue_ids", "phase_steps")}
        job = dict(base_job, snr_db=[float(value) for value in stored["snr_db"]])
        episode = validate_episode(bank.generate(job, index), protocol, job)
        if not (np.array_equal(np.asarray(episode.fingerprint), stored["fingerprint"])
                and np.array_equal(bank.last_phase_steps, stored["phase_steps"])):
            raise RuntimeError(f"{args.condition}/{name}: sequence differs from the decoded one")
        ids = bank.last_generator._power_schedule[0]
        if not np.array_equal(ids, stored["ue_ids"]):
            raise RuntimeError(f"{args.condition}/{name}: schedule differs from the decoded one")
        sweep._atomic_savez(output/name, pool_phase_steps=bank.last_generator.last_phase_steps[0], ue_ids=ids,
                            fingerprint=stored["fingerprint"])
        print(json.dumps({"condition": args.condition, "episode": index}), flush=True)


if __name__ == "__main__":
    main()
