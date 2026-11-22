"""Screening for the fallback study: the proposed NRX, the same NRX with its memory emptied, and LS + LMMSE on
sequences with an impairment of impairments.py, with closed-loop power control, on a few sequences and SNR points.
Writes results/screen/<kind>_<strength>/<condition>/<receiver>/episode_NNNNN.npz and prints, per sequence and SNR
point, the errors of each receiver and the TBs that LS + LMMSE decodes but the proposed NRX does not.

    screen.py --kind doppler --strength 600 --condition natural_umi --first-episode 0 --last-episode 4
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PC_GRID = (-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 10)   # run_receivers_pc.py
POINTS = (4, 8, 12, 16, 18, 19)                                                          # 0, 2, 4, 6, 8, 10 dB
RECEIVERS = {"pilot_memory": "candidate_persistent", "pilot_memory_emptied": "candidate_reset",
             "baseline_lslin_lmmse": "baseline_lslin_lmmse"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True)
    parser.add_argument("--strength", type=float, default=0.)
    parser.add_argument("--condition", default="natural_umi")
    parser.add_argument("--first-episode", type=int, default=0)
    parser.add_argument("--last-episode", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=HERE/"results")
    args = parser.parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import numpy as np
    import tensorflow as tf
    from neural_rx.backbone import E2E_Model, Parameters
    from neural_rx.benchmark import tbler_sweep as sweep
    from neural_rx.benchmark.frozen import protocol_and_plan
    from neural_rx.benchmark.runtime import validate_episode
    sys.path.insert(0, str(HERE))
    from banks import SETTINGS_PC
    from impairments import ImpairedBank

    sweep.configure_runtime()
    anchors = json.loads(sweep.ANCHORS.read_text())["conditions"]
    protocol, plan = protocol_and_plan("confirmation", sweep.PRBS, 32, anchors, protocol_path=sweep.PROTOCOL)
    parameters = Parameters(sweep.CONFIG, training=False, num_tx_eval=2, n_size_bwp_override=sweep.PRBS)
    bank = ImpairedBank(parameters, E2E_Model(parameters, training=False, mcs_arr_eval_idx=0), tf, SETTINGS_PC,
                        args.kind, args.strength)
    grid = [float(PC_GRID[i]) for i in POINTS]
    job = dict(sweep.sweep_job(plan, anchors, args.condition), snr_db=grid)
    receivers = {name: sweep.make_receiver(method, protocol) for name, method in RECEIVERS.items()}
    output = args.output_dir/"screen"/f"{args.kind}_{args.strength:g}"/args.condition
    for index in range(args.first_episode, args.last_episode):
        started = time.perf_counter()
        episode = validate_episode(bank.generate(job, index), protocol, job)
        generated = time.perf_counter()-started
        errors = {}
        for name, receiver in receivers.items():
            path = output/name/f"episode_{index:05d}.npz"
            if path.is_file():
                with np.load(path) as data:
                    errors[name] = data["errors"]
                continue
            errors[name], disagreement = sweep.decode_episode(RECEIVERS[name], receiver, episode, None)
            sweep._atomic_savez(path, errors=errors[name], crc_disagreement=disagreement, active=episode.active,
                                ue_ids=episode.ue_ids, snr_db=np.asarray(grid), fingerprint=np.asarray(episode.fingerprint))
        scored = episode.active[sweep.SCORING_START:].astype(bool)
        count = {name: (e[:, sweep.SCORING_START:] & scored).sum((1, 2)).tolist() for name, e in errors.items()}
        rescued = ((errors["pilot_memory"] & ~errors["baseline_lslin_lmmse"])[:, sweep.SCORING_START:] & scored).sum((1, 2))
        print(json.dumps({"kind": args.kind, "strength": args.strength, "condition": args.condition, "episode": index,
                          "snr_db": grid, "scored": int(scored.sum()), "errors": count,
                          "lslin_decodes_ours_fails": rescued.tolist(), "generate_s": round(generated, 1),
                          "decode_s": round(time.perf_counter()-started-generated, 1)}), flush=True)


if __name__ == "__main__":
    main()
