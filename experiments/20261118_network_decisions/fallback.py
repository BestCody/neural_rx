"""CRC-triggered fallback around the proposed NRX, replayed exactly from saved per-TB results.

All receivers decoded identical channels, payloads, and noise. A memory entry of the proposed NRX is its own channel
estimate of the slot, computed before decoding, so it does not depend on which receiver decodes a TB afterwards, and
the other receivers are stateless. Decoding a TB again with another receiver after a CRC failure can therefore be
evaluated from the saved error flags without new simulations. Scored on TBs 17-64 of active UEs.

Without power control: experiments/20261118_test_evaluation (sweep, sweep_high) and experiments/20261104_tbler_sweep.
With power control: experiments/20261118_test_evaluation (power_control) and
experiments/20261112_early_retransmission/results/receivers_pc. Writes results/fallback.json.
"""
import json
from pathlib import Path

import numpy as np

from controller import CONDITIONS, START, required_snr

HERE = Path(__file__).parent
NEW = HERE.parent/"20261118_test_evaluation/results"
SWEEP = HERE.parent/"20261104_tbler_sweep/results"
PC = HERE.parent/"20261112_early_retransmission/results/receivers_pc"
PRIMARY = "pilot_memory"
OURS = (PRIMARY, "pilot_memory_emptied", "neural_rx_k2_trained")
STORED = ("frozen_cold_k2", "frozen_cold_k3", "frozen_cold_k8", "baseline_lslin_lmmse", "baseline_lmmse_kbest",
          "baseline_perf_csi_kbest")
REALISTIC = (*OURS, *STORED[:-1])                     # every receiver that does not need perfect CSI
FALLBACKS = ("baseline_lmmse_kbest", "frozen_cold_k8", "baseline_lslin_lmmse")
# fallbacks a real-time BS could run: LS + LMMSE and the stateless NRXs with two iterations (0.79 ms, Latency)
REAL_TIME = ("baseline_lslin_lmmse", "pilot_memory_emptied", "neural_rx_k2_trained", "frozen_cold_k2")


def load(condition, grids, stored):
    """Per sequence: errors [SNR, TB, UE] of every receiver over the merged SNR grid, activity, and UE IDs."""
    episodes = []
    for index in range(32):
        name = f"episode_{index:05d}.npz"
        merged = {}
        for position, grid in enumerate(grids):
            part = {}
            for method in (*OURS, *STORED):
                base = (NEW/grid if method in OURS else stored[position])/condition/method/name
                with np.load(base, allow_pickle=False) as data:
                    part[method] = data["errors"]
                    reference = part.setdefault("reference", {"fingerprint": data["fingerprint"].tobytes(),
                        "snr_db": data["snr_db"], "active": data["active"], "ue_ids": data["ue_ids"]})
                    if (data["fingerprint"].tobytes() != reference["fingerprint"]
                            or not np.array_equal(data["snr_db"], reference["snr_db"])):
                        raise ValueError(f"{condition}/{method}/{name} was decoded on a different sequence")
            if not merged:
                merged = {"snr_db": part["reference"]["snr_db"], "active": part["reference"]["active"],
                          "ue_ids": part["reference"]["ue_ids"], "errors": {m: part[m] for m in (*OURS, *STORED)}}
            else:      # the second grid starts at the last point of the first; the shared point must agree
                keep = ~np.isin(part["reference"]["snr_db"], merged["snr_db"])
                for method in (*OURS, *STORED):
                    shared = part[method][~keep]
                    if shared.size and not np.array_equal(shared, merged["errors"][method][-len(shared):]):
                        raise ValueError(f"{condition}/{method}/{name}: the shared SNR point differs")
                    merged["errors"][method] = np.concatenate([merged["errors"][method], part[method][keep]])
                merged["snr_db"] = np.concatenate([merged["snr_db"], part["reference"]["snr_db"][keep]])
        episodes.append(merged)
    return episodes


def failures(episodes, rule):
    counts, blocks = 0, 0
    for episode in episodes:
        mask = episode["active"][START:][None]
        counts = counts+(rule(episode["errors"])[:, START:] & mask).sum((1, 2))
        blocks += int(mask.sum())
    return counts, blocks


def after_failure(episode, point):
    """Per failed TB of the proposed NRX at the SNR point: whether the same UE's previous scheduled TB failed."""
    active, ids, failed = episode["active"].astype(bool), episode["ue_ids"], episode["errors"][PRIMARY][point]
    last, flags = {}, []
    for tb in range(active.shape[0]):
        for position in np.flatnonzero(active[tb]):
            ue = int(ids[tb, position])
            if tb >= START and failed[tb, position]:
                flags.append(last.get(ue, False))
            last[ue] = bool(failed[tb, position])
    return flags


def main():
    report = {}
    for setting, grids, stored in (("no_power_control", ("sweep", "sweep_high"), (SWEEP/"sweep", SWEEP/"sweep_high")),
                                   ("power_control", ("power_control",), (PC,))):
        report[setting] = {}
        for condition in CONDITIONS:
            episodes = load(condition, grids, stored)
            snr = episodes[0]["snr_db"]
            count, blocks = failures(episodes, lambda e: e[PRIMARY])
            alone = required_snr(snr, count, blocks)
            point = int(np.argmin(np.abs(snr-alone)))
            gains, rescued = {}, {}
            for backup in dict.fromkeys(FALLBACKS+REAL_TIME):
                both, _ = failures(episodes, lambda e, b=backup: e[PRIMARY] & e[b])
                gains[backup] = round(round(alone, 2)-round(required_snr(snr, both, blocks), 2), 2)
                rescued[backup] = int(count[point]-both[point])       # failed TBs this fallback decodes at the point
            every, _ = failures(episodes, lambda e: np.logical_and.reduce([e[m] for m in REALISTIC]))
            gains["upper"] = round(round(alone, 2)-round(required_snr(snr, every, blocks), 2), 2)
            failed = stuck = perfect = 0
            repeated = []
            for episode in episodes:
                mask = episode["active"][START:].astype(bool)
                own = episode["errors"][PRIMARY][point, START:] & mask
                none = np.logical_and.reduce([episode["errors"][m][point, START:] for m in REALISTIC]) & mask
                failed += int(own.sum())
                stuck += int(none.sum())
                perfect += int((none & episode["errors"]["baseline_perf_csi_kbest"][point, START:]).sum())
                repeated += after_failure(episode, point)
            report[setting][condition] = {
                "required_snr_db": round(alone, 2), "operating_point_db": float(snr[point]),
                "fallback_gain_db": gains, "fallback_rescued_tbs": rescued,
                "operating_point_failures": failed, "blocks": blocks,
                "unfixable_share_of_failures": stuck/failed if failed else None,
                "unfixable_also_failed_by_perfect_csi": perfect/stuck if stuck else None,
                "failures_after_failure_of_same_ue": float(np.mean(repeated)) if repeated else None}
    (HERE/"results").mkdir(exist_ok=True)
    (HERE/"results/fallback.json").write_text(json.dumps(report, indent=1))
    for setting, rows in report.items():
        print(setting)
        for condition, r in rows.items():
            print(f"  {condition:28s} alone {r['required_snr_db']:.2f} dB | gains {r['fallback_gain_db']} | at "
                  f"{r['operating_point_db']} dB: {r['operating_point_failures']} failures, "
                  f"{100*r['unfixable_share_of_failures']:.0f}% fail for every realistic receiver, "
                  f"{100*r['failures_after_failure_of_same_ue']:.0f}% follow a failure of the same UE")


if __name__ == "__main__":
    main()
