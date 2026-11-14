"""Receivers and CRC-triggered fallback with closed-loop power control, replayed exactly from per-TB results.

All receivers decoded identical power-controlled episodes (run_receivers_pc.py, log_evidence.py --power-control).
As in experiments/20261106_crc_fallback, the proposed NRX's memory is written from pilots only, so re-decoding a
TB with another receiver after a CRC failure does not change later TBs and can be evaluated from the error flags.
Scored on TBs 17-64 of active UEs, at the SNR point closest to the proposed NRX's 10% TBLER where a point is needed.
"""
import json
from pathlib import Path

import numpy as np

from controller import CONDITIONS, START, operating_points, required_snr

HERE = Path(__file__).parent
EVIDENCE = HERE/"results/evidence_pc"
RECEIVERS = HERE/"results/receivers_pc"
PRIMARY = "candidate_persistent"
REALISTIC = (PRIMARY, "candidate_reset", "frozen_cold_k2", "frozen_cold_k3", "frozen_cold_k8",
             "baseline_lslin_lmmse", "baseline_lmmse_kbest")
METHODS = (*REALISTIC, "baseline_perf_csi_kbest")
FALLBACKS = ("baseline_lmmse_kbest", "frozen_cold_k8", "baseline_lslin_lmmse")


def load(condition):
    """Per episode: errors [SNR, TB, UE] of every method, activity, UE IDs, and effective channel power."""
    episodes = []
    for index in range(32):
        name = f"episode_{index:05d}.npz"
        errors, reference = {}, None
        for method in METHODS:
            path = EVIDENCE/condition/name if method == PRIMARY else RECEIVERS/condition/method/name
            with np.load(path, allow_pickle=False) as data:
                errors[method] = data["errors"]
                fingerprint = data["fingerprint"].tobytes()
                if reference is None:
                    reference = {"fingerprint": fingerprint, "snr_db": data["snr_db"], "active": data["active"],
                                 "ue_ids": data["ue_ids"], "channel_power": data["channel_power"]}
                elif fingerprint != reference["fingerprint"] or not np.array_equal(data["snr_db"], reference["snr_db"]):
                    raise ValueError(f"{condition}/{method}/{name} was decoded on a different episode")
        episodes.append({**reference, "errors": errors})
    return episodes


def failures(episodes, rule):
    counts, blocks = 0, 0
    for episode in episodes:
        mask = episode["active"][START:][None]
        counts = counts+(rule(episode["errors"])[:, START:] & mask).sum((1, 2))
        blocks += int(mask.sum())
    return counts, blocks


def rounded(value):
    return None if value is None else round(value, 2)


def main():
    points = operating_points(EVIDENCE)
    report = {"operating_points_db": points}
    for condition in CONDITIONS:
        episodes = load(condition)
        snr = episodes[0]["snr_db"]
        point = int(np.flatnonzero(snr == points[condition])[0])
        required = {}
        for method in METHODS:
            count, blocks = failures(episodes, lambda e, m=method: e[m])
            required[method] = required_snr(snr, count, blocks)
        alone = required[PRIMARY]
        gains = {}
        for backup in FALLBACKS:
            count, blocks = failures(episodes, lambda e, b=backup: e[PRIMARY] & e[b])
            gains[backup] = round(round(alone, 2)-round(required_snr(snr, count, blocks), 2), 2)
        count, blocks = failures(episodes, lambda e: np.logical_and.reduce([e[m] for m in REALISTIC]))
        gains["upper"] = round(round(alone, 2)-round(required_snr(snr, count, blocks), 2), 2)
        failed = stuck = perfect = 0
        power_all, power_stuck = [], []
        for episode in episodes:
            mask = episode["active"][START:]
            own = episode["errors"][PRIMARY][point, START:] & mask
            every = np.logical_and.reduce([episode["errors"][m][point, START:] for m in REALISTIC]) & mask
            failed += int(own.sum())
            stuck += int(every.sum())
            perfect += int((every & episode["errors"]["baseline_perf_csi_kbest"][point, START:]).sum())
            power = 10*np.log10(episode["channel_power"][START:])
            power_all.append(power[mask])
            power_stuck.append(power[every])
        report[condition] = {
            "required_snr_db": {m: rounded(v) for m, v in required.items()},
            "fallback_gain_db": gains,
            "operating_point_failures": failed,
            "unfixable_share_of_failures": stuck/failed if failed else None,
            "unfixable_also_failed_by_perfect_csi": perfect/stuck if stuck else None,
            "median_power_unfixable_minus_all_db": (float(np.median(np.concatenate(power_stuck))
                                                         - np.median(np.concatenate(power_all))) if stuck else None)}
    (HERE/"results/fallback_pc.json").write_text(json.dumps(report, indent=1))
    print("operating points", points)
    for condition in CONDITIONS:
        r = report[condition]
        print(condition)
        print("  required SNR:", r["required_snr_db"])
        print("  fallback gains:", r["fallback_gain_db"])
        print(f"  unfixable {r['unfixable_share_of_failures']}, perfect CSI also fails {r['unfixable_also_failed_by_perfect_csi']},"
              f" power {r['median_power_unfixable_minus_all_db']}")


if __name__ == "__main__":
    main()
