"""Ablation: how much of the memory's gain needs the estimates of earlier slots?

pilot_memory_current_tb (evaluate.py --receivers pilot_memory_current_tb) has the proposed receiver's weights, but its
filter predicts each pilot from the current TB only. Its weights are still fitted on the three newest entries, and the
first-order model between the pilots remains, so what it lacks is the estimates of earlier slots in the prediction.
Required SNR at 10% TBLER as in summarize.py. Writes results/current_tb.json.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORE = ("natural_umi", "controlled_topology_change", "schedule_gap_reentry")
RECEIVERS = ("pilot_memory_emptied", "pilot_memory_current_tb", "pilot_memory")


def main():
    from neural_rx.benchmark.tbler_sweep import summarize
    summaries = {"no_power_control": summarize([HERE/"results/sweep", HERE/"results/sweep_high"]),
                 "power_control": summarize(HERE/"results/power_control")}
    report = {}
    for setting, summary in summaries.items():
        report[setting] = {}
        for condition in CORE:
            rows = {receiver: summary[condition][receiver] for receiver in RECEIVERS}
            if len({(row["blocks"], row["episodes"]) for row in rows.values()}) != 1:
                raise ValueError(f"{setting}/{condition}: the receivers decoded different TBs")
            required = {receiver: row["snr_at_10pct_db"] for receiver, row in rows.items()}
            report[setting][condition] = {
                "required_snr_db": required,
                "memory_gain_db": required["pilot_memory_emptied"]-required["pilot_memory"],
                "gain_from_current_tb_only_db": required["pilot_memory_emptied"]-required["pilot_memory_current_tb"],
                "gain_from_earlier_slots_db": required["pilot_memory_current_tb"]-required["pilot_memory"]}
            row = report[setting][condition]
            print(f"{setting:17s} {condition:27s} emptied {required['pilot_memory_emptied']:5.2f}  current TB only "
                  f"{required['pilot_memory_current_tb']:5.2f}  full {required['pilot_memory']:5.2f} dB | gain "
                  f"{row['memory_gain_db']:.2f} = {row['gain_from_current_tb_only_db']:.2f} without + "
                  f"{row['gain_from_earlier_slots_db']:.2f} from earlier slots")
    (HERE/"results/current_tb.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
