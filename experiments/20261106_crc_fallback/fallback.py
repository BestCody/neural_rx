"""CRC-triggered fallback around the proposed NRX, replayed exactly from the saved TBLER sweep.

All receivers in experiments/20261104_tbler_sweep decoded identical channels, payloads, and noise. The proposed
NRX's memory is written from pilots only, so its outputs do not depend on which receiver decoded earlier TBs, and the
baselines and the Large NRX are stateless. A policy that re-decodes a TB with another receiver after a CRC failure
can therefore be evaluated from the saved per-TB error flags without new simulations.

Policies (scored on TBs 17-64, as in the paper):
  alone      the proposed NRX only
  fallback   after a CRC failure, the same TB is decoded again with a second receiver
  flagged    as fallback, but a UE whose last two fallback attempts both failed is flagged for network-level
             action and not re-decoded until the proposed NRX passes the CRC for it again (fixed in advance)
  upper      the best any receiver-selection rule can do: a TB fails only if every realistic receiver fails
"""
import json
from pathlib import Path

import numpy as np

from neural_rx.benchmark.plan import decreasing_isotonic, snr_at_target

HERE = Path(__file__).parent
SWEEP = HERE.parent/"20261104_tbler_sweep/results"
CONDITIONS = ("natural_umi", "controlled_topology_change", "schedule_gap_reentry")
PRIMARY = "candidate_persistent"
FALLBACKS = ("baseline_lmmse_kbest", "frozen_cold_k8", "baseline_lslin_lmmse")
# Every receiver that does not need perfect CSI.
REALISTIC = ("candidate_persistent", "candidate_reset", "frozen_cold_k2", "frozen_cold_k3", "frozen_cold_k8",
             "baseline_lslin_lmmse", "baseline_lmmse_kbest")
START = 16
FLAG_AFTER = 2


def load(condition, method):
    """Per-episode errors [SNR, TB, UE] over the merged SNR grid, plus the shared episode data."""
    low = sorted((SWEEP/"sweep"/condition/method).glob("episode_*.npz"))
    high = sorted((SWEEP/"sweep_high"/condition/method).glob("episode_*.npz"))
    assert len(low) == len(high) == 32 and [p.name for p in low] == [p.name for p in high], (condition, method)
    episodes = []
    for a, b in zip(low, high, strict=True):
        with np.load(a) as x, np.load(b) as y:
            # Each sweep has its own fingerprint (it covers the SNR grid); receivers are compared within each sweep.
            assert np.array_equal(x["active"], y["active"]) and np.array_equal(x["ue_ids"], y["ue_ids"])
            keep =~np.isin(y["snr_db"], x["snr_db"])  # shared SNR points: keep the first sweep, as the summary does
            snr = np.concatenate([x["snr_db"], y["snr_db"][keep]])
            order = np.argsort(snr)
            episodes.append({"snr_db": snr[order],
                             "errors": np.concatenate([x["errors"], y["errors"][keep]])[order],
                             "crc_disagreement": int(x["crc_disagreement"][:, START:].sum()
                                                     + y["crc_disagreement"][keep][:, START:].sum()),
                             "active": x["active"], "ue_ids": x["ue_ids"], "channel_power": x["channel_power"],
                             "fingerprint": x["fingerprint"].tobytes()+y["fingerprint"].tobytes()})
    return episodes


def flagged(primary, backup, active, ue_ids):
    """Errors and re-decode flags [SNR, TB, UE] of the flagged-fallback policy, run causally from TB 16."""
    errors, redecoded = primary.copy(), np.zeros_like(primary)
    for point in range(primary.shape[0]):
        misses, flag = {}, set()
        for tb in range(START, primary.shape[1]):
            for position in np.flatnonzero(active[tb]):
                ue = int(ue_ids[tb, position])
                if not primary[point, tb, position]:
                    flag.discard(ue)
                    misses[ue] = 0
                    continue
                if ue in flag:
                    continue
                redecoded[point, tb, position] = True
                errors[point, tb, position] = backup[point, tb, position]
                misses[ue] = misses.get(ue, 0)+1 if backup[point, tb, position] else 0
                if misses[ue] >= FLAG_AFTER:
                    flag.add(ue)
    return errors, redecoded


def curve(snr, failures, blocks):
    rates = failures/blocks
    fitted = decreasing_isotonic(rates, np.full(len(rates), float(blocks)))
    return {"tbler": rates.tolist(), "snr_at_10pct_db": snr_at_target(snr, fitted, .1)}


def main():
    report = {"flag_after": FLAG_AFTER, "scored_tbs": [START+1, 64]}
    for condition in CONDITIONS:
        data = {method: load(condition, method) for method in REALISTIC}
        reference = data[PRIMARY]
        for method, episodes in data.items():
            assert [e["fingerprint"] for e in episodes] == [e["fingerprint"] for e in reference], method
            assert all(np.array_equal(e["snr_db"], r["snr_db"]) for e, r in zip(episodes, reference, strict=True))
        # The fallback is triggered by the CRC; with no CRC disagreements, a CRC failure is exactly a TB error.
        disagreements = {method: sum(e["crc_disagreement"] for e in episodes) for method, episodes in data.items()}
        assert not any(disagreements.values()), disagreements
        snr = reference[0]["snr_db"]
        policies = {"alone": lambda i: (data[PRIMARY][i]["errors"], np.zeros_like(data[PRIMARY][i]["errors"]))}
        for backup in FALLBACKS:
            policies[f"fallback_{backup}"] = (
                lambda i, b=backup: (data[PRIMARY][i]["errors"] & data[b][i]["errors"], data[PRIMARY][i]["errors"]))
            policies[f"flagged_{backup}"] = (
                lambda i, b=backup: flagged(data[PRIMARY][i]["errors"], data[b][i]["errors"], reference[i]["active"],
                                            reference[i]["ue_ids"]))
        policies["upper"] = lambda i: (np.logical_and.reduce([data[m][i]["errors"] for m in REALISTIC]),
                                       data[PRIMARY][i]["errors"])
        blocks = sum(int(e["active"][START:].sum()) for e in reference)
        result = {"snr_db": snr.tolist(), "blocks": blocks}
        for name, policy in policies.items():
            failures, redecoded = np.zeros(len(snr)), np.zeros(len(snr))
            for i, episode in enumerate(reference):
                errors, again = policy(i)
                mask = episode["active"][START:][None]
                failures += (errors[:, START:] & mask).sum((1, 2))
                redecoded += (again[:, START:] & mask).sum((1, 2))
            result[name] = {**curve(snr, failures, blocks), "redecoded_share": (redecoded/blocks).tolist()}
        # The replay must reproduce the sweep summary behind the paper for every single receiver.
        summary = json.loads((SWEEP/"summary.json").read_text())[condition]
        for method in REALISTIC:
            failures = sum((data[method][i]["errors"][:, START:] & e["active"][START:][None]).sum((1, 2))
                           for i, e in enumerate(reference))
            assert failures.tolist() == summary[method]["failures"], (condition, method)
        assert result["alone"]["snr_at_10pct_db"] == summary[PRIMARY]["snr_at_10pct_db"], condition
        # Operating point: the SNR point closest to the proposed NRX's 10% TBLER.
        alone = result["alone"]["snr_at_10pct_db"]
        point = int(np.argmin(np.abs(snr-alone)))
        result["operating_point_snr_db"] = float(snr[point])
        for name in policies:
            result[name]["gain_db"] = alone-result[name]["snr_at_10pct_db"]
            # The paper quotes differences of required SNRs rounded to 0.01 dB (scripts/nrx_paper_numbers.py).
            result[name]["gain_db_from_rounded"] = round(round(alone, 2)-round(result[name]["snr_at_10pct_db"], 2), 2)
            result[name]["redecoded_share_at_operating_point"] = result[name]["redecoded_share"][point]
        # Failures of the proposed NRX that no realistic receiver fixes, their channel power, and how many of them
        # even perfect CSI does not fix.
        perfect = load(condition, "baseline_perf_csi_kbest")
        assert [e["fingerprint"] for e in perfect] == [e["fingerprint"] for e in reference]
        stuck, failed, perfect_fails, power_stuck, power_all = 0, 0, 0, [], []
        for i, episode in enumerate(reference):
            mask = episode["active"][START:]
            primary = data[PRIMARY][i]["errors"][point, START:] & mask
            every = np.logical_and.reduce([data[m][i]["errors"][point, START:] for m in REALISTIC]) & mask
            failed += int(primary.sum())
            stuck += int(every.sum())
            perfect_fails += int((every & perfect[i]["errors"][point, START:]).sum())
            power = 10*np.log10(episode["channel_power"][START:])
            power_stuck.append(power[every])
            power_all.append(power[mask])
        result["unfixable_share_of_failures"] = stuck/failed
        result["unfixable_also_failed_by_perfect_csi"] = perfect_fails/stuck
        result["median_power_unfixable_minus_all_db"] = float(np.median(np.concatenate(power_stuck))
                                                              - np.median(np.concatenate(power_all)))
        report[condition] = result
    (HERE/"results").mkdir(exist_ok=True)
    (HERE/"results/fallback.json").write_text(json.dumps(report, indent=1))
    for condition in CONDITIONS:
        r = report[condition]
        print(f"{condition}: operating point {r['operating_point_snr_db']:.2f} dB, "
              f"unfixable {100*r['unfixable_share_of_failures']:.1f}% of failures "
              f"({100*r['unfixable_also_failed_by_perfect_csi']:.1f}% of them also with perfect CSI), "
              f"their median channel power {r['median_power_unfixable_minus_all_db']:+.2f} dB vs all TBs")
        for name in ("alone", *(f"{p}_{b}" for b in FALLBACKS for p in ("fallback", "flagged")), "upper"):
            v = r[name]
            print(f"  {name:36s} SNR@10% {v['snr_at_10pct_db']:6.2f} dB  gain {v['gain_db_from_rounded']:5.2f} dB  "
                  f"re-decoded {100*v['redecoded_share_at_operating_point']:5.1f}%")


if __name__ == "__main__":
    main()
