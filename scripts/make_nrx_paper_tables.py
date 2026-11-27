"""The paper's tables, written from the result files as LaTeX tabulars into <paper>/tables/ (the floats and captions
stay in the section files):

    required_snr.tex      required SNR at 10% TBLER with 16-QAM, without and with power control, for all receivers,
                          the ablation with the current TB only, and the memory and filter in front of K-Best
    modulations.tex       required SNR with QPSK and 64-QAM, and with 16-QAM under power control with impairments
    further_datasets.tex  TBLER at a fixed SNR on the two further datasets
    latency.tex           median TensorRT latency on the A100
    network.tex           fallback and early retransmission (workshop version only)

    python scripts/make_nrx_paper_tables.py [--paper paper|paper_ieee]          # write the tables
    python scripts/make_nrx_paper_tables.py [--paper ...] --check               # exit 1 if a file is out of date

scripts/nrx_paper_recheck.py compares every cell with values it recomputes itself from the raw result files.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nrx_paper_numbers import (ROOT, CORE, classical, further, phase_jumps,  # noqa: E402
                               required_snr)

ROBUSTNESS = ROOT/"experiments/20261122_robustness/results"
MODULATIONS = ROOT/"experiments/20261121_modulations/results"
NETWORK = ROOT/"experiments/20261118_network_decisions/results"
IT = "$N_{\\mathrm{it}}=%d$"
NAMES = {"baseline_lslin_lmmse": "LS + LMMSE", "baseline_lmmse_kbest": "LMMSE + K-Best",
         "baseline_perf_csi_kbest": "Perf.-CSI + K-Best", "frozen_cold_k2": "Large NRX, "+IT % 2,
         "frozen_cold_k3": "Large NRX, "+IT % 3, "frozen_cold_k8": "Large NRX, "+IT % 8,
         "neural_rx_k2_trained": "Large NRX retrained, "+IT % 2, "pilot_memory": "Proposed NRX",
         "pilot_memory_emptied": "\\quad no memory", "pilot_memory_current_tb": "\\quad current TB only",
         "filter_kbest": "K-Best, memory and filter", "wiener_kbest": "K-Best, multi-slot LMMSE"}
SETTINGS = ("no_power_control", "power_control")
CONDITIONS = "UMi & Change & Gaps"


def number(value, digits=2):
    text = f"{value:.{digits}f}"
    return f"$-${text[1:]}" if text.startswith("-") else text


def row(label, values, digits=2):
    return " & ".join([label]+[number(v, digits) if v is not None else "--" for v in values])+" \\\\"


def required_snr_table():
    table = required_snr()[0]
    extra = classical()
    current = json.loads((ROOT/"experiments/20261118_test_evaluation/results/current_tb.json").read_text())
    value = {(s, c, m): table[s][c][m] for s in SETTINGS for c in CORE for m in table[s][c]}
    for s in SETTINGS:
        for c in CORE:
            value[s, c, "filter_kbest"] = extra[s][c]["filter_kbest_db"]
            value[s, c, "wiener_kbest"] = extra[s][c]["wiener_kbest_db"]
            value[s, c, "pilot_memory_current_tb"] = current[s][c]["required_snr_db"]["pilot_memory_current_tb"]
    groups = (("baseline_lslin_lmmse", "baseline_lmmse_kbest", "baseline_perf_csi_kbest"),
              ("frozen_cold_k2", "frozen_cold_k3", "frozen_cold_k8", "neural_rx_k2_trained"),
              ("pilot_memory", "pilot_memory_emptied", "pilot_memory_current_tb"), ("filter_kbest", "wiener_kbest"))
    lines = ["\\begin{tabular}{@{}l@{\\hspace{4pt}}rrr@{\\hspace{6pt}}rrr@{}}", "\\toprule",
             " & \\multicolumn{3}{c}{Without power control} & \\multicolumn{3}{c}{With power control} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}", f"Receiver & {CONDITIONS} & {CONDITIONS} \\\\"]
    for group in groups:
        lines.append("\\midrule")
        lines += [row(NAMES[m], [value[s, c, m] for s in SETTINGS for c in CORE]) for m in group]
    return lines+["\\bottomrule", "\\end{tabular}"]


def modulations_table():
    mods = {name: json.loads((MODULATIONS/label/"summary.json").read_text())
            for name, label in (("QPSK", "nrx_large_qpsk"), ("64-QAM", "nrx_large_64qam"))}
    robust = json.loads((ROBUSTNESS/"summary.json").read_text())
    jumps = json.loads((ROOT/"experiments/20261118_test_evaluation/results/phase_jumps.json").read_text())
    pc = required_snr()[0]["power_control"]
    assert phase_jumps()                                     # the same file nrx_paper_numbers.py reads
    receivers = ("frozen_cold_k3", "frozen_cold_k8", "neural_rx_k2_trained", "pilot_memory", "pilot_memory_emptied")
    spec = "@{}l" + "@{\\hspace{6pt}}rrr"*4 + "@{}"
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
             " & \\multicolumn{6}{c}{QPSK (MCS 9)} & \\multicolumn{6}{c}{64-QAM (MCS 19)} \\\\",
             "\\cmidrule(lr){2-7}\\cmidrule(l){8-13}",
             " & \\multicolumn{3}{c}{Without power control} & \\multicolumn{3}{c}{With power control}"
             " & \\multicolumn{3}{c}{Without power control} & \\multicolumn{3}{c}{With power control} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(l){11-13}",
             f"Receiver & {CONDITIONS} & {CONDITIONS} & {CONDITIONS} & {CONDITIONS} \\\\", "\\midrule"]
    for m in receivers:
        lines.append(row(NAMES[m], [mods[name][key][c]["required_snr_db"][m] for name in mods
                                    for key in ("sweep", "power_control") for c in CORE]))
    lines += ["\\midrule",
              " & \\multicolumn{12}{c}{16-QAM (MCS 14) with power control} \\\\", "\\cmidrule(l){2-13}",
              " & \\multicolumn{3}{c}{No impairment} & \\multicolumn{3}{c}{Phase jumps}"
              " & \\multicolumn{3}{c}{Frequency offset} & \\multicolumn{3}{c}{TDD} \\\\",
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(l){11-13}"]
    for m in ("neural_rx_k2_trained", "pilot_memory", "pilot_memory_emptied"):
        values = ([pc[c][m] for c in CORE] + [jumps[c]["required_snr_db"][m] for c in CORE]
                  + [robust["cfo"][c]["required_snr_db"][m] for c in CORE]
                  + [robust["tdd_dddsu"][c]["required_snr_db"][m] for c in CORE])
        lines.append(row(NAMES[m], values))
    return lines+["\\bottomrule", "\\end{tabular}"]


FURTHER_NAMES = {"historical_tdl_b100": "TDL-B100", "historical_tdl_c300": "TDL-C300", "natural_umi": "UMi",
                 "natural_uma": "UMa", "controlled_topology_change": "Channel change",
                 "schedule_gap_reentry": "Scheduling gaps", "close_los": "Close LOS UEs", "near_far": "Near-far UEs",
                 "ray_traced_munich": "Munich, ray-traced"}


def further_table():
    report = further()
    snr = json.loads((ROOT/"experiments/20261118_further_datasets/results/v2/protocol.json").read_text())
    import numpy as np
    points = np.load(ROOT/"experiments/20261118_further_datasets/results/v2/errors.npz")["snr_db"][:, 0]
    order = list(report["v2"])
    assert all(np.load(ROOT/"experiments/20261118_further_datasets/results/seed86/errors.npz")["snr_db"][:, 0]
               == points[:len(report["seed86"])]), "the datasets use different SNR points"
    assert snr["condition_order"][:len(order)] == order
    sub = "8 it. & Prop. & No mem."
    lines = ["\\begin{tabular}{@{}l@{\\hspace{4pt}}rrr@{\\hspace{8pt}}rrr@{}}", "\\toprule",
             " & \\multicolumn{3}{c}{Dataset 1} & \\multicolumn{3}{c}{Dataset 2} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}", f"Condition & {sub} & {sub} \\\\", "\\midrule"]
    for index, name in enumerate(order):
        values = []
        for dataset in ("v2", "seed86"):
            r = report[dataset].get(name)
            values += [100*r[k] if r else None for k in ("large_k8", "proposed", "no_memory")]
        lines.append(" & ".join([FURTHER_NAMES[name]] + [number(v, 1) if v is not None else "--" for v in values])
                     + " \\\\")
    return lines+["\\bottomrule", "\\end{tabular}"]


def network_table():
    fallback = json.loads((NETWORK/"fallback.json").read_text())
    probe = json.loads((ROBUSTNESS/"fallback_k_summary.json").read_text())
    controller = json.loads((NETWORK/"controller.json").read_text())
    compare = json.loads((NETWORK/"compare_receivers.json").read_text())["receivers"]
    harq = json.loads((ROBUSTNESS/"harq_summary.json").read_text())["all"]

    def pooled(setting, key=None, receiver=None):
        """Over the three conditions at the operating point: rescued failed TBs or a share of the failed TBs, in %."""
        rows = fallback[setting].values()
        failures = sum(r["operating_point_failures"] for r in rows)
        if receiver:
            return 100*sum(r["fallback_rescued_tbs"][receiver] for r in rows)/failures
        return 100*sum(r[key]*r["operating_point_failures"] for r in rows)/failures

    step = controller["step_by_step"]["CRC history and receiver evidence, budget of the rule"]
    p, s = compare["proposed"], compare["stateless"]
    other = max(probe["power_control"]["other"]["no_stored_tb_share"],
                probe["power_control_phase"]["other"]["no_stored_tb_share"])
    lines = ["\\begin{tabular}{@{}lrr@{}}", "\\toprule", " & No PC & PC \\\\", "\\midrule",
             "\\multicolumn{3}{@{}l}{\\emph{Fallback from the memory}: DMRS symbols on which} \\\\",
             "\\multicolumn{3}{@{}l}{the channel filter uses no stored TB [\\%]} \\\\",
             row("\\quad at a channel change", [None, 100*probe["power_control"]["change"]["no_stored_tb_share"]], 1),
             row("\\quad after a phase jump", [None, 100*probe["power_control_phase"]["phase_jump"]["no_stored_tb_share"]], 1),
             row("\\quad otherwise, at most", [None, 100*other], 1),
             "\\midrule",
             "\\multicolumn{3}{@{}l}{\\emph{Fallback after a CRC failure}: failed TBs rescued [\\%] by} \\\\"]
    for receiver in ("baseline_lslin_lmmse", "neural_rx_k2_trained", "baseline_lmmse_kbest", "frozen_cold_k8"):
        label = {"neural_rx_k2_trained": "Stateless NRX, "+IT % 2}.get(receiver, NAMES[receiver])
        lines.append(row("\\quad "+label.replace("\\quad ", "Proposed NRX, "),
                         [pooled(setting, receiver=receiver) for setting in SETTINGS], 1))
    lines += [row("Failed TBs after an error of the UE [\\%]",
                  [pooled(setting, "failures_after_failure_of_same_ue") for setting in SETTINGS], 1),
              "\\midrule",
              "\\emph{Early retransmission} & CRC hist. & Controller \\\\",
              "Errors caught early [\\%] & & \\\\",
              row("\\quad proposed NRX", [100*controller["rule"]["caught"], 100*step["caught"]], 1),
              row("\\quad proposed NRX, no memory state", [100*p["rule_caught_share"], 100*p["caught_share"]], 1),
              row("\\quad stateless NRX, no memory state", [100*s["rule_caught_share"], 100*s["caught_share"]], 1),
              row("Correct TBs flagged [\\%], proposed", [100*controller["rule"]["false_alarm"], 100*step["false_alarm"]], 1),
              "Errors after the CRC per 100 TBs & & \\\\",
              row("\\quad proposed NRX, no memory state", [p["rule_late_per_100"], p["late_per_100"]], 1),
              row("\\quad stateless NRX, no memory state", [s["rule_late_per_100"], s["late_per_100"]], 1),
              "\\midrule",
              "\\emph{Simulated HARQ} & After CRC & Early \\\\",
              row("Median time to decoding [ms]", [harq["regular"]["median_ms_to_decoding"],
                                                   harq["early"]["median_ms_to_decoding"]], 1),
              row("Unneeded transmissions per 100 TBs", [0., harq["early_extra_transmissions_per_100_tbs"]], 1),
              "\\bottomrule", "\\end{tabular}"]
    return lines


def latency_table():
    from nrx_paper_numbers import latency
    t = latency()
    values = [t[k] for k in ("large_k2_ms", "large_k3_ms", "large_k8_ms", "proposed_ms", "before_slot_ms",
                             "proposed_and_fit_ms", "proposed_overlapped_ms")]
    return ["\\begin{tabular}{@{}rrr@{\\hspace{10pt}}rrrr@{}}", "\\toprule",
            "\\multicolumn{3}{c}{Large NRX, $N_{\\mathrm{it}}$} & \\multicolumn{4}{c}{Proposed NRX} \\\\",
            "\\cmidrule(r){1-3}\\cmidrule(l){4-7}", "2 & 3 & 8 & Slot & Before & Seq. & Par. \\\\", "\\midrule",
            " & ".join(number(v) for v in values)+" \\\\", "\\bottomrule", "\\end{tabular}"]


TABLES = {"required_snr": required_snr_table, "modulations": modulations_table, "further_datasets": further_table,
          "latency": latency_table, "network": network_table}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", default="paper")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    folder = ROOT/args.paper/"tables"
    folder.mkdir(exist_ok=True)
    stale = []
    for name, make in TABLES.items():
        if args.paper != "paper" and name == "network":       # the IEEE version has no Network Decisions section
            continue
        content = "% written by scripts/make_nrx_paper_tables.py; do not edit\n" + "\n".join(make()) + "\n"
        path = folder/f"{name}.tex"
        if args.check:
            if not path.exists() or path.read_text() != content:
                stale.append(name)
        else:
            path.write_text(content)
    if args.check:
        print("tables out of date:", stale or "none")
        raise SystemExit(1 if stale else 0)


if __name__ == "__main__":
    main()
