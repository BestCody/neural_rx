"""Render the receiver-architecture and evaluation-setup diagrams; no receiver execution."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "misc"
FORMATS = ("png", "pdf")
INK = "#172c3d"
GRAY = "#526579"
TEAL = "#087f73"
BLUE = "#255e99"
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})


def canvas(title, subtitle, size=(13, 8)):
    fig = plt.figure(figsize=size, facecolor="white")
    ax = fig.add_axes([0.02, 0.055, 0.96, 0.78])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    fig.text(0.055, 0.935, title, fontsize=22, weight="bold", color=INK)
    fig.text(0.055, 0.885, subtitle, fontsize=12, color=GRAY)
    return fig, ax


def box(ax, x, y, w, h, title, detail="", color=BLUE, size=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.007,rounding_size=0.014",
                                linewidth=1.3, edgecolor=color, facecolor="#f4f8fb" if color == BLUE else "#eef8f5"))
    ax.text(x+w/2, y+h*(0.66 if detail else 0.5), title,
            ha="center", va="center", fontsize=size, weight="bold", color=color)
    if detail:
        ax.text(x+w/2, y+h*0.28, detail, ha="center", va="center", fontsize=size-1.5, color=INK)


def arrow(ax, points, color=GRAY, style="-"):
    for first, second in zip(points[:-2], points[1:-1], strict=True):
        ax.plot([first[0], second[0]], [first[1], second[1]], color=color, lw=1.5, linestyle=style)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=13,
                                 color=color, linewidth=1.5, linestyle=style))


def save(fig, stem):
    for ext in FORMATS:
        fig.savefig(OUT / f"{stem}.{ext}", dpi=180, facecolor="white")
    plt.close(fig)


def architecture():
    fig, ax = canvas("Pilot-history neural receiver",
                     "K=2 CGNN • learned channel filter over per-UE pilot history • 8 + 10 LDPC iterations")
    x, w, h = 0.28, 0.39, 0.105
    stages = [
        (0.835, "Current received grid", "Pilots + overlapping users’ data"),
        (0.675, "CGNN detector: K=2", "Initializer → round 1 → round 2 → bit scores"),
        (0.515, "LDPC: first 8 iterations", "Code constraints provide additional evidence"),
        (0.355, "Symbol-pair correction + blend", "Physical pair scores, learned per-bit blend"),
        (0.195, "LDPC: 10 further iterations", "Continue from the retained decoder state"),
        (0.035, "Decoded payload + CRC", "Final bits and error-detection result"),
    ]
    for y, title, detail in stages:
        box(ax, x, y, w, h, title, detail)
    for (y1, _, _), (y2, _, _) in zip(stages, stages[1:], strict=False):
        arrow(ax, [(x+w/2, y1-0.01), (x+w/2, y2+h+0.009)])
    box(ax, 0.765, 0.655, 0.215, 0.15, "Per-UE pilot history",
        "Current LS + 7 previous TBs", color=TEAL, size=11)
    arrow(ax, [(0.68, 0.888), (0.872, 0.888), (0.872, 0.816)], TEAL)
    ax.text(0.735, 0.91, "Extract current pilots", color=TEAL, fontsize=10)
    arrow(ax, [(0.754, 0.729), (0.681, 0.729)], TEAL)
    ax.text(0.73, 0.79, "Filtered\nchannel", ha="center", va="bottom", fontsize=10, color=TEAL)
    arrow(ax, [(0.872, 0.643), (0.872, 0.407), (0.681, 0.407)], TEAL)
    ax.text(0.865, 0.49, "Learned channel filter\n+ correlated uncertainty", ha="center", fontsize=10, color=TEAL,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 4})
    ax.text(0.876, 0.27, "Stored by UE identity (4 slots)\nExpires 8 slots after last update\nUpdated each block",
            ha="center", va="center", fontsize=10, linespacing=1.6, color=TEAL)
    arrow(ax, [(0.27, 0.888), (0.115, 0.888), (0.115, 0.407), (0.268, 0.407)])
    ax.text(0.118, 0.62, "Current received\nsignal", ha="center", va="center", fontsize=11,
            color=GRAY, bbox={"facecolor": "white", "edgecolor": "none", "pad": 5})
    ax.text(0.13, 0.18, "Only pilot history\npersists across blocks.\n"
            "Detector and decoder states\nrestart for each payload.",
            ha="center", va="center", fontsize=10, color=GRAY, linespacing=1.55)
    fig.text(0.055, 0.02, "K counts CGNN processing rounds. The decoder split uses the declared total of 18 LDPC "
             "iterations (config/memory_receiver.json).", fontsize=10, color=GRAY)
    save(fig, "receiver_architecture")


def evaluation():
    fig, ax = canvas("Evaluation: paired evolving-channel episodes",
                     "Sionna link-level simulation • 2 active UEs • 4 receive antennas • 16-QAM, MCS 14 • 132 PRBs",
                     (13, 7.4))
    box(ax, 0.035, 0.71, 0.22, 0.14, "UE transmitters", "New payload each TB + LDPC")
    box(ax, 0.32, 0.71, 0.20, 0.14, "Evolving channel", "8 conditions, see below")
    arrow(ax, [(0.267, 0.78), (0.31, 0.78)])
    box(ax, 0.58, 0.71, 0.16, 0.14, "Sum + noise", "Received mixture")
    arrow(ax, [(0.53, 0.78), (0.57, 0.78)])
    box(ax, 0.805, 0.80, 0.175, 0.10, "Pilot-history receiver", "persistent memory", color=TEAL, size=10.5)
    box(ax, 0.805, 0.665, 0.175, 0.10, "Same weights, reset", "memory cleared every TB", size=10.5)
    box(ax, 0.805, 0.53, 0.175, 0.10, "Cold K8 / Cold K3", "frozen; accuracy / latency", size=10.5)
    arrow(ax, [(0.752, 0.78), (0.775, 0.78), (0.775, 0.85), (0.795, 0.85)], TEAL)
    arrow(ax, [(0.775, 0.78), (0.775, 0.715), (0.795, 0.715)])
    arrow(ax, [(0.775, 0.715), (0.775, 0.58), (0.795, 0.58)])
    ax.text(0.98, 0.47, "Same received observations\nfor every receiver",
            ha="right", va="top", fontsize=10, color=GRAY)
    ax.text(0.035, 0.56, "Conditions", fontsize=13, weight="bold", color=INK)
    ax.text(0.035, 0.50, "Core (pooled headline): UMi • UMa • topology change • schedule-gap re-entry",
            fontsize=11, color=INK)
    ax.text(0.035, 0.455, "Reported: TDL-B100 • TDL-C300 • close LoS • near-far", fontsize=11, color=GRAY)
    ax.text(0.035, 0.33, "Each of 32 episodes per condition", fontsize=13, weight="bold", color=INK)
    box(ax, 0.035, 0.105, 0.25, 0.15, "TBs 1–16", "Process; build pilot history", color=TEAL)
    box(ax, 0.35, 0.105, 0.62, 0.15, "TBs 17–64: score every receiver",
        "TBLER over scored TBs of both UEs; no training during evaluation")
    arrow(ax, [(0.298, 0.18), (0.339, 0.18)])
    fig.text(0.055, 0.026, "Development bank selects the checkpoint; "
             "an independent confirmation bank reports accuracy. "
             "Latency: p95, prepared LS grid to decoded payload.", fontsize=10, color=GRAY)
    save(fig, "evaluation_setup")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render explanatory diagrams; no receiver execution.")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=FORMATS)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    FORMATS = tuple(args.formats)
    OUT.mkdir(parents=True, exist_ok=True)
    architecture()
    evaluation()
