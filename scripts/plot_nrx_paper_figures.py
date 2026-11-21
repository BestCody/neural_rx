"""Paper figures in the style of Wiesmayr et al. (arXiv:2409.02912)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/"experiments/20261104_tbler_sweep/results"        # released Large NRX and classical receivers
TRAINED = ROOT/"experiments/20261118_test_evaluation/results"     # proposed NRX and retrained Large NRX
TENSORRT = ROOT/"experiments/20261114_tensorrt_latency/results"
OUT = ROOT/"paper/figures"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                     "pdf.fonttype": 42, "axes.linewidth": .7, "font.size": 8})

GREEN, PURPLE, PINK = "#76b900", "#7b2d9e", "#d6006e"
CURVES = {"baseline_lslin_lmmse": ("LS + LMMSE", dict(color="#e41a1c", linestyle="-")),
          "baseline_lmmse_kbest": ("LMMSE + K-Best", dict(color="#1f78b4", linestyle="-")),
          "baseline_perf_csi_kbest": ("Perf.-CSI + K-Best", dict(color="#4daf4a", linestyle="-")),
          "frozen_cold_k2": ("Large NRX, $N_{\\mathrm{it}}=2$", dict(color="#ff7f00", linestyle=(0, (5, 1.5)))),
          "frozen_cold_k8": ("Large NRX, $N_{\\mathrm{it}}=8$", dict(color="#ff7f00", linestyle=":")),
          "neural_rx_k2_trained": ("Large NRX retrained, $N_{\\mathrm{it}}=2$",
                                   dict(color="#a65628", linestyle=(0, (4, 1, 1, 1, 1, 1)))),
          "pilot_memory_emptied": ("Proposed NRX, no memory", dict(color="#6a3d9a", linestyle="-.")),
          "pilot_memory": ("Proposed NRX", dict(color=PINK, linestyle="--"))}
PANELS = {"natural_umi": "(a) UMi", "controlled_topology_change": "(b) Channel change",
          "schedule_gap_reentry": "(c) Scheduling gaps"}


def save(fig, stem):
    fig.savefig(OUT/f"{stem}.pdf", bbox_inches="tight", pad_inches=.01)
    fig.savefig(OUT/f"{stem}.png", bbox_inches="tight", pad_inches=.01, dpi=220)
    plt.close(fig)


def architecture():
    fig = plt.figure(figsize=(7, 1.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, .228))
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((.075, .006), .83, .216, boxstyle="round,pad=0,rounding_size=.01", linewidth=.9,
                                edgecolor=GREEN, facecolor="none"))
    ax.text(.082, .209, "Neural Receiver", fontsize=7, color=GREEN, va="center")

    def block(x, y, width, label, color=GREEN, new=False):
        ax.add_patch(FancyBboxPatch((x, y), width, .05, boxstyle="round,pad=0,rounding_size=.008", linewidth=.8,
                                    edgecolor=color, facecolor="#fde6f1" if new else "white"))
        ax.text(x+width/2, y+.025, label, ha="center", va="center", fontsize=6.5, family="monospace")

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=6, color="#555555", linewidth=.8,
                                     shrinkA=0, shrinkB=0))

    def line(*points):
        ax.plot(*zip(*points, strict=True), color="#555555", linewidth=.8)

    top, bottom, bus = .118, .025, .19
    middle = top+.025
    chain = [(.2, .09, "StateInit", PURPLE, False), (.33, .1, "CGNNIt x2", PURPLE, False),
             (.47, .105, "ReadoutLLRs", PURPLE, False), (.615, .08, "LDPC", GREEN, False), (.735, .06, "CRC", GREEN, False)]
    for x, width, label, color, new in chain:
        block(x, top, width, label, color, new)
    for (x, width, *_), following in zip(chain, chain[1:], strict=False):
        arrow((x+width, middle), (following[0], middle))
    block(.095, top, .07, "LS Est.")
    block(.095, bottom, .1, "Pilot Memory", PINK, True)            # as "New in this work" in the legend
    block(.215, bottom, .115, "Channel Filter", PINK, True)

    # The received slot enters the LS estimator and StateInit.
    ax.text(.037, middle, "Received\nslot", ha="center", va="center", fontsize=6.5)
    arrow((.06, middle), (.095, middle))
    line((.083, middle), (.083, bus), (.245, bus))
    arrow((.245, bus), (.245, top+.05))
    ax.text(.86, middle, "Payload", ha="center", va="center", fontsize=6.5)
    arrow((.795, middle), (.83, middle))

    # The LS estimate of the slot reaches the filter and becomes the newest memory entry; the filter, whose weights
    # are fitted on the memory, feeds StateInit.
    arrow((.115, top), (.115, bottom+.05))
    line((.15, top), (.15, .097), (.235, .097))
    arrow((.235, .097), (.235, bottom+.05))
    arrow((.195, bottom+.025), (.215, bottom+.025))
    arrow((.27, bottom+.05), (.27, top))
    ax.legend(handles=[Patch(facecolor="white", edgecolor=PURPLE, label="Trainable"),
                       Patch(facecolor="#fde6f1", edgecolor=PINK, label="New in this work")],
              loc="lower right", bbox_to_anchor=(.895, .03), fontsize=6, frameon=False, handlelength=1.2, ncol=1,
              borderaxespad=0)
    save(fig, "receiver_architecture")


def tbler_curves(summary):
    """TBLER vs. SNR per condition: without power control (top row) and with it (bottom row)."""
    fig, grid = plt.subplots(2, 3, figsize=(7, 2.85), sharey=True)
    letters = iter("abcdef")
    for axes, setting, suffix in ((grid[0], "no_power_control", ""), (grid[1], "power_control", ", power control")):
        for ax, (condition, title) in zip(axes, PANELS.items(), strict=True):
            for method, (_, style) in CURVES.items():
                data = summary[setting][condition][method]
                ax.semilogy(data["snr_db"], [v if v > 0 else float("nan") for v in data["tbler"]], linewidth=1.05, **style)
            ax.set_ylim(3e-4, 1.2)
            ax.xaxis.set_major_locator(plt.MultipleLocator(5))
            ax.grid(True, which="both", linewidth=.3, color="#bbbbbb")
            ax.tick_params(labelsize=7, length=2, pad=1.5)
            ax.set_xlabel("SNR [dB]", fontsize=7.5, labelpad=1)
            ax.set_title(f"({next(letters)}) {title.split(') ')[1]}{suffix}", fontsize=7.5, y=-.47)
        axes[0].set_ylabel("TBLER", fontsize=7.5, labelpad=1)
    handles = [plt.Line2D([], [], linewidth=1.05, label=label, **style) for label, style in CURVES.values()]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=6.6, frameon=True, fancybox=False,
               edgecolor="black", borderpad=.3, columnspacing=1, handlelength=2.2, bbox_to_anchor=(.5, 1.13))
    fig.subplots_adjust(wspace=.09, hspace=.62, left=.06, right=.995, top=.98, bottom=.13)
    save(fig, "tbler_vs_snr")


def snr_and_latency(summary, latency):
    """Required SNR (left) and median TensorRT inference time on the A100 (right) vs. N_it for UMi, in the export of
    the code release of Wiesmayr et al."""
    umi = summary["no_power_control"]["natural_umi"]
    iterations = {2: "frozen_cold_k2", 3: "frozen_cold_k3", 8: "frozen_cold_k8"}
    engines = {2: "large_k2", 3: "large_k3", 8: "large_k8"}
    fig, left = plt.subplots(figsize=(3.3, 1.6))
    right = left.twinx()
    left.plot(list(iterations), [umi[m]["snr_at_10pct_db"] for m in iterations.values()], color="#ff7f00",
              linestyle=":", marker=".", linewidth=1.05, label="Large NRX (SNR)")
    # Same architecture as the Large NRX with N_it=2, so only its required SNR is shown.
    left.plot([2], [umi["neural_rx_k2_trained"]["snr_at_10pct_db"]], color="#a65628", marker="^", markersize=4.5,
              linestyle="none", label="Large NRX retrained, $N_{\\mathrm{it}}=2$ (SNR)")
    left.plot([2], [umi["pilot_memory"]["snr_at_10pct_db"]], color=PINK, marker="*", markersize=7,
              linestyle="none", label="Proposed NRX (SNR)")
    right.plot(list(engines), [latency[e] for e in engines.values()], color="#1f78b4", marker="s",
               markersize=3, linewidth=1.05, label="Large NRX latency")
    # Hollow and larger, so the Large NRX's marker at N_it = 2 stays visible next to it.
    right.plot([2], [latency["proposed"]], color=PINK, marker="s", markersize=5, markeredgewidth=.9,
               markerfacecolor="none", linestyle="none", label="Proposed NRX latency")
    left.set_xlabel("Number of NRX iterations $N_{\\mathrm{it}}$", fontsize=7.5, labelpad=1)
    left.set_ylabel("SNR@TBLER=10% [dB]", fontsize=7.5, labelpad=2)
    right.set_ylabel("Latency [ms]", fontsize=7.5, labelpad=2)
    left.set_xticks(range(1, 9))
    left.set_ylim(6.6, 8.2)
    right.set_ylim(.55, 2.6)  # keeps the latency markers at N_it = 2 below the SNR markers
    left.grid(True, linewidth=.3, color="#bbbbbb")
    for ax in (left, right):
        ax.tick_params(labelsize=7, length=2, pad=1.5)
    handles = left.get_legend_handles_labels()[0]+right.get_legend_handles_labels()[0]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=6.3, frameon=True, fancybox=False,
               edgecolor="black", borderpad=.3, columnspacing=1, bbox_to_anchor=(.5, 1.27))
    save(fig, "snr_latency_tradeoff")


def memory_gain(numbers):
    gains = numbers
    fig, ax = plt.subplots(figsize=(3.3, 1.45))
    bars = ax.bar(list(gains), list(gains.values()), width=.55, color=PINK)
    ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1.5)
    ax.set_xlabel("Doppler shift [Hz]", fontsize=7.5, labelpad=1)
    ax.set_ylabel("Memory gain [dB]", fontsize=7.5, labelpad=2)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", linewidth=.3, color="#bbbbbb")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7, length=2, pad=1.5)
    save(fig, "memory_gain_vs_doppler")


def load_summary():
    """TBLER curves and required SNR per setting, condition and receiver: the stored results of the released Large NRX
    and the classical receivers, plus the proposed NRX and the retrained Large NRX, decoded on the same sequences."""
    from neural_rx.benchmark.tbler_sweep import summarize
    summary = {"no_power_control": json.loads((RESULTS/"summary.json").read_text()),
               "power_control": summarize(ROOT/"experiments/20261112_early_retransmission/results/receivers_pc")}
    trained = json.loads((TRAINED/"summary.json").read_text())
    for setting, rows in trained.items():
        for condition, row in rows.items():
            for receiver, tbler in row["tbler"].items():
                summary[setting][condition][receiver] = {"snr_db": row["snr_db"], "tbler": tbler,
                                                         "snr_at_10pct_db": row["required_snr_db"][receiver]}
    return summary


def main():
    summary = load_summary()
    latency = json.loads((TENSORRT/"fused_a100.json").read_text())["median_ms"]
    required = json.loads((TRAINED/"doppler.json").read_text())["required_effective_snr_db"]
    architecture()
    tbler_curves(summary)
    snr_and_latency(summary, latency)
    memory_gain({group: required["pilot_memory_emptied"][group]-required["pilot_memory"][group]
                 for group in ("0-50", "50-175", "175-400")})


if __name__ == "__main__":
    main()
