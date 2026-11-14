"""Accuracy of cheaper IDD detectors on the UMi test sequences (proposed NRX with memory, same weights).

The exact detector evaluates all 256 symbol pairs per resource element with log-sum-exp. Variants (switches in
native_ops/pair_correction.cu.cc, read per call):
  maxlog       maxima instead of log-sum-exp
  k8_maxlog    only each UE's 8 most likely symbols by the NRX LLRs (64 pairs), max-log
  k4_maxlog    only each UE's 4 most likely symbols (16 pairs), max-log
In this test the full score table is still computed and then masked, so it measures accuracy, not speed. The fit
feature of the combining network still uses all pairs.

Episodes are regenerated as in experiments/20261114_no_idd_ablation and must match the stored fingerprints; "exact"
must reproduce the stored errors.

    evaluate.py --gpu 0 --first-episode 0 --last-episode 16
"""
import argparse
import json
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SWEEP = ROOT/"experiments/20261104_tbler_sweep/results"
VARIANTS = {"exact": (0, 0), "maxlog": (0, 1), "k8_maxlog": (8, 1), "k4_maxlog": (4, 1)}
METHOD, SETTING, CONDITION = "candidate_persistent", "sweep", "natural_umi"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--first-episode", type=int, default=0)
    parser.add_argument("--last-episode", type=int, default=32)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import numpy as np
    from neural_rx.benchmark import tbler_sweep as sweep
    from neural_rx.benchmark.runtime import validate_episode

    sweep.configure_runtime()
    protocol, plan, parameters, bank, anchors = sweep.sweep_context()
    base_job = sweep.sweep_job(plan, anchors, CONDITION)
    receiver = sweep.make_receiver(METHOD, protocol)
    for index in range(args.first_episode, args.last_episode):
        name = f"episode_{index:05d}.npz"
        outputs = {v: HERE/"results/test"/v/name for v in VARIANTS if v != "exact"}
        if all(path.is_file() for path in outputs.values()):
            continue
        with np.load(SWEEP/SETTING/CONDITION/METHOD/name, allow_pickle=False) as data:
            stored = {key: data[key] for key in data.files}
        job = dict(base_job, snr_db=[float(v) for v in stored["snr_db"]])
        episode = validate_episode(bank.generate(job, index), protocol, job)
        if not np.array_equal(np.asarray(episode.fingerprint), stored["fingerprint"]):
            raise RuntimeError(f"{name}: episode differs from the stored results")
        report = {"episode": index}
        for variant, (keep, maxlog) in VARIANTS.items():
            os.environ["NRX_IDD_KEEP"], os.environ["NRX_IDD_MAXLOG"] = str(keep), str(maxlog)
            errors, disagreement = sweep.decode_episode(METHOD, receiver, episode, None)
            report[variant] = errors[:, sweep.SCORING_START:].sum((1, 2)).tolist()
            if variant == "exact":
                scored = slice(sweep.SCORING_START, None)
                if (errors[:, scored] != stored["errors"][:, scored]).any():
                    raise RuntimeError(f"{name}: the exact detector differs from the stored run")
                continue
            keep_keys = {k: stored[k] for k in ("active", "ue_ids", "snr_db", "slot_number", "fingerprint", "channel_power")}
            sweep._atomic_savez(outputs[variant], errors=errors, crc_disagreement=disagreement, **keep_keys)
        os.environ["NRX_IDD_KEEP"], os.environ["NRX_IDD_MAXLOG"] = "0", "0"
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
