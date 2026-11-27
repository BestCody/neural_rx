"""Figures of the IEEE version (paper_ieee/figures): a lower Fig. 2 and the modulation summary (Fig. 3).

modulations_summary: required SNR at 10% TBLER of the stateless NRXs relative to the proposed NRX, for QPSK (MCS 9),
16-QAM (MCS 14) and 64-QAM (MCS 19), without and with power control, one marker per test condition. 16-QAM from the
paper's evaluation (plot_nrx_paper_figures.load_summary), QPSK and 64-QAM from experiments/20261121_modulations.

    python scripts/plot_nrx_ieee_figures.py
"""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from plot_nrx_paper_figures import CURVES, PINK, ROOT, TRAINED, load_summary, memory_gain, tbler_curves  # noqa: E402

OUT = ROOT/"paper_ieee/figures"
MODULATIONS = ROOT/"experiments/20261121_modulations/results"
CONDITIONS = ("natural_umi", "controlled_topology_change", "schedule_gap_reentry")
# Each receiver keeps its color of Fig. 2 (the Large NRX family in orange) and gets its own marker shape.
MARKERS = {"pilot_memory_emptied": (CURVES["pilot_memory_emptied"][1]["color"], "v", "full", "Proposed NRX, no memory"),
           "neural_rx_k2_trained": (CURVES["neural_rx_k2_trained"][1]["color"], "^", "full",
                                    "Large NRX retrained, $N_{\\mathrm{it}}=2$"),
           "frozen_cold_k3": (CURVES["frozen_cold_k8"][1]["color"], "D", "none", "Large NRX, $N_{\\mathrm{it}}=3$"),
           "frozen_cold_k8": (CURVES["frozen_cold_k8"][1]["color"], "o", "full", "Large NRX, $N_{\\mathrm{it}}=8$")}


def required_snr():
    """[setting][modulation][condition][receiver] -> required SNR in dB."""
    summary = load_summary()
    table = {s: {"16-QAM": {c: {r: summary[s][c][r]["snr_at_10pct_db"] for r in (*MARKERS, "pilot_memory")}
                            for c in CONDITIONS}} for s in ("no_power_control", "power_control")}
    for label, name in (("nrx_large_qpsk", "QPSK"), ("nrx_large_64qam", "64-QAM")):
        rows = json.loads((MODULATIONS/label/"summary.json").read_text())
        for setting, key in (("no_power_control", "sweep"), ("power_control", "power_control")):
            table[setting][name] = {c: rows[key][c]["required_snr_db"] for c in CONDITIONS}
    return table


def modulations_summary():
    table = required_snr()
    order = ("QPSK", "16-QAM", "64-QAM")
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.45), sharey=True)
    for ax, (setting, title) in zip(axes, (("no_power_control", "(a) Without power control"),
                                           ("power_control", "(b) With power control"))):
        ax.axhline(0, color=PINK, linestyle="--", linewidth=1.05)
        for index, modulation in enumerate(order):
            for offset, (receiver, (color, marker, fill, _)) in zip((-.24, -.08, .08, .24), MARKERS.items()):
                values = [table[setting][modulation][c][receiver]-table[setting][modulation][c]["pilot_memory"]
                          for c in CONDITIONS]
                ax.plot([index+offset]*len(values), values, linestyle="none", marker=marker, markersize=3.6,
                        color=color, fillstyle=fill, markeredgewidth=.8)
        ax.set_xticks(range(len(order)), order)
        ax.set_xlim(-.55, len(order)-.45)
        ax.set_title(title, fontsize=7.5, pad=2)
        ax.grid(True, axis="y", linewidth=.3, color="#bbbbbb")
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7, length=2, pad=1.5)
    axes[0].set_ylabel("Required SNR above\nproposed NRX [dB]", fontsize=7.5, labelpad=2)
    handles = [plt.Line2D([], [], linestyle="none", marker=m, color=c, fillstyle=f, markersize=3.6, markeredgewidth=.8,
                          label=label) for c, m, f, label in MARKERS.values()]
    handles.append(plt.Line2D([], [], color=PINK, linestyle="--", linewidth=1.05, label="Proposed NRX"))
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=6.5, frameon=True, fancybox=False,
               edgecolor="black", borderpad=.3, columnspacing=1, handlelength=1.8, bbox_to_anchor=(.53, -.02))
    fig.subplots_adjust(wspace=.08)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/"modulations_summary.pdf", bbox_inches="tight", pad_inches=.01)
    fig.savefig(OUT/"modulations_summary.png", bbox_inches="tight", pad_inches=.01, dpi=220)
    return table


if __name__ == "__main__":
    # Fig. 2 of the workshop version, lower for the six-page format, and the Doppler figure; latency is a table.
    tbler_curves(load_summary(), size=(7, 2.35), out=OUT)
    required = json.loads((TRAINED/"doppler.json").read_text())["required_effective_snr_db"]
    memory_gain({g: required["pilot_memory_emptied"][g]-required["pilot_memory"][g] for g in ("0-50", "50-175", "175-400")}, out=OUT)
    table = modulations_summary()
    for setting, rows in table.items():
        for modulation, conditions in rows.items():
            print(setting, modulation, {c: {r: round(v-conditions[c]["pilot_memory"], 2) for r, v in values.items()
                                            if r != "pilot_memory"} for c, values in conditions.items()})
