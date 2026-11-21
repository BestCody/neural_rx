"""Early retransmission with the proposed NRX versus the stateless Large NRX retrained with two iterations.

Both receivers decoded the same power-controlled test sequences; their per-TB evidence is in results/evidence_pc
(log_evidence.py) and results/evidence_pc_stateless (stateless_evidence.py). The controller is that of controller.py,
fitted on sequences 0-15 and tested on 16-31, at the same SNR points for both receivers (closest to the proposed
NRX's 10% TBLER), with the same inputs for both: the CRC history, the noise level, the LS and estimated channel
strength, then the LLR bit error rate, then the parity checks after 8 BP iterations; no memory features. Each
receiver's thresholds flag the same share of its correct fitting TBs (the share the CRC-history rule flags for the
proposed NRX).

Per 100 scheduled TBs of the test sequences: failures, failures caught before decoding finishes, and failures left
for a retransmission after the CRC fails (late), plus unneeded requests. Writes results/compare_receivers.json.
"""
import json
from pathlib import Path

import numpy as np

from controller import ANY_RECEIVER, FIT, HISTORY, TEST, load, operating_points, step_by_step

HERE = Path(__file__).resolve().parent
RECEIVERS = {"proposed": HERE/"results/evidence_pc", "stateless": HERE/"results/evidence_pc_stateless"}
CHECK_1 = HISTORY+ANY_RECEIVER+["estimate_strength"]
STEPS = {"check 1": CHECK_1, "check 2": CHECK_1+["llr_bit_error"],
         "check 3": CHECK_1+["llr_bit_error", "unsatisfied_checks", "failed_code_blocks", "posterior_bit_error"]}


def main():
    points = operating_points(RECEIVERS["proposed"])
    data = {name: (load(directory, FIT, points), load(directory, TEST, points)) for name, directory in RECEIVERS.items()}
    fit, _ = data["proposed"]
    budget = float(((fit["previous_failed"] == 1) & (fit["failed"] == 0)).sum()/(fit["failed"] == 0).sum())
    report = {"operating_points_db": points, "false_alarm_budget": budget, "inputs": STEPS, "receivers": {}}
    for name, (fit_data, test_data) in data.items():
        first = step_by_step(fit_data, test_data, STEPS, budget)
        failed, flagged = test_data["failed"] == 1, first >= 0
        tbs = len(failed)
        rule = test_data["previous_failed"] == 1
        report["receivers"][name] = {
            "test_ue_tbs": tbs, "failures_per_100": 100*float(failed.mean()),
            "caught_share": float(flagged[failed].mean()), "rule_caught_share": float(rule[failed].mean()),
            "early_per_100": 100*float((flagged & failed).sum()/tbs),
            "late_per_100": 100*float((~flagged & failed).sum()/tbs),
            "rule_late_per_100": 100*float((~rule & failed).sum()/tbs),
            "unneeded_per_100": 100*float((flagged & ~failed).sum()/tbs),
            "false_alarm_share": float(flagged[~failed].mean()),
            "caught_at": {step: float((first[failed] == position).mean()) for position, step in enumerate(STEPS)}}
        r = report["receivers"][name]
        print(f"{name:10s} per 100 TBs: {r['failures_per_100']:5.2f} failures, {r['early_per_100']:5.2f} caught early "
              f"({100*r['caught_share']:.1f}%), {r['late_per_100']:5.2f} late (CRC-history rule alone: "
              f"{r['rule_late_per_100']:5.2f}), {r['unneeded_per_100']:5.2f} unneeded requests "
              f"({100*r['false_alarm_share']:.2f}% of correct TBs)")
    (HERE/"results/compare_receivers.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
