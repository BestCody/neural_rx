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
OUT = ROOT/"paper/figures"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                     "pdf.fonttype": 42, "axes.linewidth": .7, "font.size": 8})

GREEN, PURPLE, PINK = "#76b900", "#7b2d9e", "#d6006e"
CURVES = {"baseline_lslin_lmmse": ("LS + LMMSE", dict(color="#e41a1c", linestyle="-")),
          "baseline_lmmse_kbest": ("LMMSE + K-Best", dict(color="#1f78b4", linestyle="-")),
          "baseline_perf_csi_kbest": ("Perf.-CSI + K-Best", dict(color="#4daf4a", linestyle="-")),
          "frozen_cold_k2": ("Large NRX, $N_{\\mathrm{it}}=2$", dict(color="#ff7f00", linestyle=(0, (5, 1.5)))),
          "frozen_cold_k3": ("Large NRX, $N_{\\mathrm{it}}=3$", dict(color="#ff7f00", linestyle=(0, (3, 1, 1, 1)))),
          "frozen_cold_k8": ("Large NRX, $N_{\\mathrm{it}}=8$", dict(color="#ff7f00", linestyle=":")),
          "neural_rx_k2_trained": ("Large NRX retrained, $N_{\\mathrm{it}}=2$",
                                   dict(color="#a65628", linestyle=(0, (4, 1, 1, 1, 1, 1)))),
          "pilot_memory_emptied": ("Proposed NRX, no memory", dict(color="#6a3d9a", linestyle="-.")),
          "pilot_memory": ("Proposed NRX", dict(color=PINK, linestyle="--"))}
PANELS = {"natural_umi": "(a) UMi", "controlled_topology_change": "(b) Channel change",
          "schedule_gap_reentry": "(c) Scheduling gaps"}


def save(fig, stem, out=None):
    out = OUT if out is None else out          # the IEEE version (plot_nrx_ieee_figures.py) writes elsewhere
    fig.savefig(out/f"{stem}.pdf", bbox_inches="tight", pad_inches=.01)
    fig.savefig(out/f"{stem}.png", bbox_inches="tight", pad_inches=.01, dpi=220)
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


def tbler_curves(summary, size=(7, 2.4), out=None):
    """TBLER vs. SNR per condition around TBLER = 10%, without power control (top row) and with it (bottom row). Each
    panel spans the receivers' required SNRs, and a dot marks each one (the crossing of 10% as in Table 1)."""
    fig, grid = plt.subplots(2, 3, figsize=size, sharey=True)
    letters = iter("abcdef")
    for axes, setting, suffix in ((grid[0], "no_power_control", ""), (grid[1], "power_control", ", power control")):
        for ax, (condition, title) in zip(axes, PANELS.items(), strict=True):
            rows = summary[setting][condition]
            required = [rows[method]["snr_at_10pct_db"] for method in CURVES]
            ax.axhline(.1, color="#555555", linewidth=.6, linestyle=(0, (1, 1.5)), zorder=1)
            for method, (_, style) in CURVES.items():
                data = rows[method]
                ax.semilogy(data["snr_db"], [v if v > 0 else float("nan") for v in data["tbler"]], linewidth=1.05,
                            zorder=2, **style)
                ax.plot([data["snr_at_10pct_db"]], [.1], marker="o", markersize=2.9, color=style["color"],
                        markeredgewidth=0, linestyle="none", zorder=3)
            ax.set_xlim(min(required)-.4, max(required)+.4)
            ax.set_ylim(.03, .4)
            ax.yaxis.set_major_locator(plt.FixedLocator([.05, .1, .2, .3]))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
            ax.yaxis.set_minor_locator(plt.NullLocator())
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
            ax.grid(True, which="major", linewidth=.3, color="#bbbbbb")
            ax.tick_params(labelsize=7, length=2, pad=1.5)
            ax.set_xlabel(f"SNR [dB]\n({next(letters)}) {title.split(') ')[1]}{suffix}", fontsize=7.5, labelpad=1, linespacing=1.6)
        axes[0].set_ylabel("TBLER", fontsize=7.5, labelpad=1)
    handles = [plt.Line2D([], [], linewidth=1.05, label=label, **style) for label, style in CURVES.values()]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=6.6, frameon=True, fancybox=False,
               edgecolor="black", borderpad=.3, columnspacing=1, handlelength=2.2, bbox_to_anchor=(.5, 1.13))
    fig.subplots_adjust(wspace=.09, hspace=.72, left=.06, right=.995, top=.98, bottom=.17)
    save(fig, "tbler_vs_snr", out)


def memory_gain(numbers):
    gains = numbers
    fig, ax = plt.subplots(figsize=(3.3, 1.3))
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
    required = json.loads((TRAINED/"doppler.json").read_text())["required_effective_snr_db"]
    architecture()
    tbler_curves(summary)
    memory_gain({group: required["pilot_memory_emptied"][group]-required["pilot_memory"][group]
                 for group in ("0-50", "50-175", "175-400")})


if __name__ == "__main__":
    main()
