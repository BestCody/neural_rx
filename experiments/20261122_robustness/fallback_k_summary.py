"""Summary of fallback_k.py: on which TBs the channel filter uses no stored TB (K = 0, or no fit), per UE and DMRS
symbol, for the scored TBs (17-64) of active UEs: the TB of the channel change and the two after it, the TBs right
after a phase jump of the UE, and all other TBs. Also checks that the errors equal the stored evaluation at the same
SNR point. Writes results/fallback_k_summary.json.
"""
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
START = 16
STORED = {"power_control": ROOT/"experiments/20261118_network_decisions/results/evidence_pc/{condition}/episode_{index:05d}.npz",
          "power_control_phase": ROOT/"experiments/20261118_test_evaluation/results/power_control_phase/{condition}/pilot_memory/episode_{index:05d}.npz"}


def main():
    report = {}
    for setting, condition in (("power_control", "controlled_topology_change"), ("power_control_phase", "natural_umi")):
        groups, mismatches, files = {}, 0, sorted(glob.glob(str(HERE/f"results/fallback_k/{setting}/{condition}/episode_*.npz")))
        for path in files:
            data = np.load(path)
            index, point = int(path[-9:-4]), float(data["snr_db"])
            with np.load(str(STORED[setting]).format(condition=condition, index=index)) as stored:
                errors = stored["errors"][int(np.flatnonzero(np.isclose(stored["snr_db"], point))[0])]
            scored = np.zeros(data["active"].shape, bool)
            scored[START:] = data["active"][START:].astype(bool)
            mismatches += int((scored & (data["errors"] != errors)).sum())
            unused = (data["k"] == 0) | (data["fitted"] == 0)
            label = np.full(scored.shape, "other", object)
            if setting == "power_control":
                change = np.flatnonzero(data["channel_change_event"].any(1))
                for offset, name in ((0, "change"), (1, "one_after_change"), (2, "two_after_change")):
                    if len(change) and change[0]+offset < scored.shape[0]:
                        label[change[0]+offset] = name
            else:
                label[data["phase_steps"].astype(bool)] = "phase_jump"
            for name in np.unique(label[scored]):
                mask = scored & (label == name)
                group = groups.setdefault(name, {"symbols": 0, "no_stored_tb": 0, "k": []})
                group["symbols"] += int(unused[mask].size)
                group["no_stored_tb"] += int(unused[mask].sum())
                group["k"] += data["k"][mask].ravel().tolist()
        report[setting] = {"condition": condition, "sequences": len(files), "error_mismatches": mismatches,
                           **{name: {"symbols": g["symbols"], "no_stored_tb_share": g["no_stored_tb"]/g["symbols"],
                                     "mean_k": float(np.mean(g["k"]))} for name, g in groups.items()}}
    (HERE/"results/fallback_k_summary.json").write_text(json.dumps(report, indent=1)+"\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
