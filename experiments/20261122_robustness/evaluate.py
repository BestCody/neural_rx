"""TBLER of the November 2026 16-QAM receivers on the sequences of banks.py, with closed-loop power control, on the
SNR grid of the power-control evaluation (experiments/20261112_early_retransmission, run_receivers_pc.py).

Tests:
  tdd_dddsu, tdd_dddddddsuu   uplink slots of a TDD pattern (banks.TDDBank)
  cfo                         a carrier frequency offset of +-0.1 ppm per UE (banks.FrequencyOffsetBank)
Receivers (results/<test>/<condition>/<receiver>/episode_NNNNN.npz):
  pilot_memory            the proposed receiver as published: the memory gets each TB's true slot index, so an entry's
                          age is the time since its slot, and entries older than eight slots (4 ms) are left out
  pilot_memory_uplink     the same weights, but the memory counts the UE's uplink slots, as if they were back to back,
                          so the eight-slot limit spans eight uplink slots; only differs from pilot_memory with TDD
  pilot_memory_emptied    the same weights with the memory emptied before every TB
  neural_rx_k2_trained    the Large NRX retrained with two iterations (stage 1), models/neural_rx_k2_20261117

    evaluate.py --test tdd_dddsu --condition natural_umi --first-episode 0 --last-episode 32 [--points 0 8 16]
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NEURAL_RX = ROOT/"models/neural_rx_k2_20261117/checkpoint_011000"
PC_GRID = (-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 10)   # run_receivers_pc.py
RECEIVERS = {"pilot_memory": "candidate_persistent", "pilot_memory_uplink": "candidate_persistent",
             "pilot_memory_emptied": "candidate_reset", "neural_rx_k2_trained": "frozen_cold_k2"}
TESTS = ("tdd_dddsu", "tdd_dddddddsuu", "cfo")


def with_slot_time(episode, slots):
    """The episode with each TB's true slot index as the memory's time (the adapter's time_index)."""
    inputs = tuple(tuple(dict(tb, time_index=int(slots[index])) for index, tb in enumerate(point))
                   for point in episode.receiver_inputs)
    return dataclasses.replace(episode, receiver_inputs=inputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=TESTS, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--first-episode", type=int, default=0)
    parser.add_argument("--last-episode", type=int, default=32)
    parser.add_argument("--points", type=int, nargs="+", help="indices into the SNR grid (default: all)")
    parser.add_argument("--receivers", nargs="+", choices=tuple(RECEIVERS))
    parser.add_argument("--output-dir", type=Path, default=HERE/"results")
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import numpy as np
    import tensorflow as tf
    from neural_rx.backbone import E2E_Model, Parameters
    from neural_rx.backbone.training import load_weights
    from neural_rx.benchmark import tbler_sweep as sweep
    from neural_rx.benchmark.frozen import protocol_and_plan
    from neural_rx.benchmark.runtime import validate_episode
    from neural_rx.provenance.artifacts import file_sha256
    sys.path.insert(0, str(HERE))
    from banks import SETTINGS_PC, FrequencyOffsetBank, TDDBank, uplink_slots

    sweep.configure_runtime()
    receivers_wanted = args.receivers or [r for r in RECEIVERS if args.test != "cfo" or r != "pilot_memory_uplink"]
    anchors = json.loads(sweep.ANCHORS.read_text())["conditions"]
    protocol, plan = protocol_and_plan("confirmation", sweep.PRBS, 32, anchors, protocol_path=sweep.PROTOCOL)
    parameters = Parameters(sweep.CONFIG, training=False, num_tx_eval=2, n_size_bwp_override=sweep.PRBS)
    e2e = E2E_Model(parameters, training=False, mcs_arr_eval_idx=0)
    if args.test == "cfo":
        bank = FrequencyOffsetBank(parameters, e2e, tf, SETTINGS_PC)
        pattern = None
    else:
        pattern = args.test.removeprefix("tdd_")
        bank = TDDBank(parameters, e2e, tf, SETTINGS_PC, pattern)
    grid = [float(PC_GRID[i]) for i in args.points] if args.points else [float(v) for v in PC_GRID]
    job = dict(sweep.sweep_job(plan, anchors, args.condition), snr_db=grid)

    receivers, identity = {}, {}
    for name in receivers_wanted:
        method = RECEIVERS[name]
        receivers[name] = sweep.make_receiver(method, protocol)
        checkpoint = NEURAL_RX if method == "frozen_cold_k2" else sweep.CHECKPOINT
        saved = json.loads((checkpoint/"manifest.json").read_text())
        weights = checkpoint/("nrx_weights.pkl" if method == "frozen_cold_k2" else "receiver.weights.h5")
        if file_sha256(weights) != saved["files"][weights.name]:
            raise ValueError(f"{name}: weights differ from the checkpoint manifest")
        if method == "frozen_cold_k2":
            load_weights(receivers[name].model, str(weights))
            receivers[name]._model_hash = receivers[name].model_checksum(receivers[name].model)
        identity[name] = np.asarray(file_sha256(weights))

    output = args.output_dir/args.test/args.condition
    for index in range(args.first_episode, args.last_episode):
        name = f"episode_{index:05d}.npz"
        pending = [r for r in receivers if not (output/r/name).is_file()]
        if not pending:
            continue
        started = time.perf_counter()
        episode = validate_episode(bank.generate(job, index), protocol, job)
        generator = bank.last_generator
        tbs = episode.active.shape[0]
        slots = uplink_slots(pattern, tbs) if pattern else np.arange(tbs)
        extra = {"uplink_slot": slots} if pattern else {"cfo_hz": generator.last_cfo_hz[0][generator._power_schedule[0]]}
        generated = time.perf_counter()-started
        report = {}
        for receiver in pending:
            method = RECEIVERS[receiver]
            decoded = with_slot_time(episode, slots) if receiver == "pilot_memory" else episode
            errors, disagreement = sweep.decode_episode(method, receivers[receiver], decoded, None)
            sweep._atomic_savez(output/receiver/name, errors=errors, crc_disagreement=disagreement,
                                active=episode.active, ue_ids=episode.ue_ids, snr_db=np.asarray(grid),
                                fingerprint=np.asarray(episode.fingerprint), power_gain_db=bank.last_power_gain_db,
                                weights_sha256=identity[receiver], **extra)
            report[receiver] = errors[:, sweep.SCORING_START:].sum((1, 2)).tolist()
        print(json.dumps({"test": args.test, "condition": args.condition, "episode": index,
                          "generate_s": round(generated, 1), "decode_s": round(time.perf_counter()-started-generated, 1),
                          "errors": report}), flush=True)
    for name, receiver in receivers.items():
        if not receiver.checksums_unchanged():
            raise RuntimeError(f"{name}: receiver weights changed during the run")


if __name__ == "__main__":
    main()
