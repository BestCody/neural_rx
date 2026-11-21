"""Power control with a random phase jump at every power step (evaluate.py --setting power_control_phase) against
power control without them: required SNR at 10% TBLER of the proposed NRX, the same with its memory emptied, and the
retrained Large NRX with two iterations, all decoded on the phase-jump sequences, as in summarize.py; the memory's
gain with and without phase jumps; how many scheduled UE-TBs in the scored slots follow a phase jump of their UE
since its previous scheduled TB; and each receiver's TBLER on the UE-TBs while a jump is in the filter's fit window
(the UE's first TB after the jump and its next two) and on the others, at the SNR point where the proposed NRX's
TBLER is closest to 10%. The jumps per slot and pool UE are in results/power_control_phase_steps (phase_steps.py).
Writes results/phase_jumps.json.
"""
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CORE = ("natural_umi", "controlled_topology_change", "schedule_gap_reentry")
RECEIVERS = ("pilot_memory", "pilot_memory_emptied", "neural_rx_k2_trained")
START = 16
FIT_TBS = 3


def jump_masks(steps, ids, active):
    """Per [slot, scheduled position]: whether the UE's phase jumped since its previous scheduled TB, and whether it
    did so for this TB or one of the UE's FIT_TBS-1 TBs before it."""
    jump, near = np.zeros(active.shape, bool), np.zeros(active.shape, bool)
    for ue in range(steps.shape[1]):
        slots, positions = np.nonzero((ids == ue) & active)
        order = np.argsort(slots)
        slots, positions = slots[order], positions[order]
        previous = np.concatenate([[-1], slots[:-1]])
        jumped = np.array([steps[a+1:b+1, ue].any() for a, b in zip(previous, slots)], bool)
        recent = jumped.copy()
        for lag in range(1, FIT_TBS):
            recent[lag:] |= jumped[:-lag]
        jump[slots, positions], near[slots, positions] = jumped, recent
    return jump, near


def main():
    from neural_rx.benchmark.tbler_sweep import summarize
    stored = json.loads((HERE/"results/summary.json").read_text())["power_control"]
    jumps = summarize(HERE/"results/power_control_phase")
    report = {}
    for condition in CORE:
        rows = {r: jumps[condition][r] for r in RECEIVERS}
        if len({(row["blocks"], row["episodes"]) for row in rows.values()}) != 1 or rows["pilot_memory"]["episodes"] != 32:
            raise ValueError(f"{condition}: incomplete or unpaired results")
        steps = scheduled = 0
        split = {}
        for receiver in RECEIVERS:
            counts = np.zeros((2, 2, len(rows[receiver]["failures"])))   # [near a jump or not, failures or TBs, SNR]
            for path in sorted(glob.glob(str(HERE/"results/power_control_phase"/condition/receiver/"episode_*.npz"))):
                steps_path = HERE/"results/power_control_phase_steps"/condition/Path(path).name
                with np.load(path) as data, np.load(steps_path) as pool:
                    if not np.array_equal(data["fingerprint"], pool["fingerprint"]):
                        raise ValueError(f"{steps_path}: steps of another sequence")
                    jump, near = jump_masks(pool["pool_phase_steps"], pool["ue_ids"], data["active"].astype(bool))
                    active, jump, near = data["active"][START:].astype(bool), jump[START:], near[START:]
                    errors, grid = data["errors"][:, START:], data["snr_db"]
                    for row, mask in enumerate((active & near, active & ~near)):
                        counts[row, 0] += (errors & mask[None]).sum((1, 2))
                        counts[row, 1] += mask.sum()
                    if receiver == "pilot_memory":
                        steps += int((jump & active).sum())
                        scheduled += int(active.sum())
            split[receiver] = counts
        ours = split["pilot_memory"]
        point = int(np.argmin(np.abs((ours[0, 0]+ours[1, 0])/(ours[0, 1]+ours[1, 1])-0.1)))
        snr = float(grid[point])
        required = {r: rows[r]["snr_at_10pct_db"] for r in RECEIVERS}
        before = {r: stored[condition]["required_snr_db"][r] for r in RECEIVERS}
        report[condition] = {
            "required_snr_db": required, "required_snr_db_without_jumps": before,
            "memory_gain_db": required["pilot_memory_emptied"]-required["pilot_memory"],
            "memory_gain_db_without_jumps": before["pilot_memory_emptied"]-before["pilot_memory"],
            "lead_over_retrained_db": required["neural_rx_k2_trained"]-required["pilot_memory"],
            "lead_over_retrained_db_without_jumps": before["neural_rx_k2_trained"]-before["pilot_memory"],
            "phase_jumps": steps, "scheduled_ue_tbs": scheduled, "phase_jump_share": steps/scheduled,
            "tbler_snr_db": snr, "ue_tbs_near_jumps": int(ours[0, 1, point]),
            "tbler_near_jumps": {r: float(c[0, 0, point]/c[0, 1, point]) for r, c in split.items()},
            "tbler_elsewhere": {r: float(c[1, 0, point]/c[1, 1, point]) for r, c in split.items()}}
        r = report[condition]
        print(f"{condition:27s} phase jumps in {100*r['phase_jump_share']:4.1f}% of scored UE-TBs | memory gain "
              f"{r['memory_gain_db']:.2f} dB (without jumps {r['memory_gain_db_without_jumps']:.2f}) | lead over the "
              f"retrained NRX {r['lead_over_retrained_db']:.2f} dB (without jumps {r['lead_over_retrained_db_without_jumps']:.2f})")
        print(f"{'':27s} TBLER at {snr:.1f} dB near jumps ({r['ue_tbs_near_jumps']} UE-TBs) / elsewhere: " + ", ".join(
            f"{name} {r['tbler_near_jumps'][name]:.3f}/{r['tbler_elsewhere'][name]:.3f}" for name in RECEIVERS))
    (HERE/"results/phase_jumps.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
