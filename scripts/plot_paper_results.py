"""Plot the selected receiver's sealed confirmation, latency, and development results; no receiver execution."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neural_rx.benchmark.generalization_protocol import verify_payload

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/20261019_learned_blend_joint_training_6000/results"
OUT = ROOT / "misc"
FORMATS = ("png", "pdf")

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
# Color follows the receiver in every figure (validated categorical slots 1-4).
METHODS = {
    "candidate_persistent": ("Pilot history (persistent)", "#2a78d6"),
    "candidate_reset": ("Same weights, reset each TB", "#eb6834"),
    "frozen_cold_k8": ("Cold K8", "#1baf7a"),
    "frozen_cold_k3": ("Cold K3", "#eda100"),
}
CONDITIONS = {
    "natural_umi": "UMi", "natural_uma": "UMa", "controlled_topology_change": "Topology\nchange",
    "schedule_gap_reentry": "Schedule-gap\nre-entry", "historical_tdl_b100": "TDL-B100",
    "historical_tdl_c300": "TDL-C300", "close_los": "Close LoS", "near_far": "Near-far",
}
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})


def load_result(directory):
    result = json.loads((directory/"results.json").read_text())
    verify_payload(result, field="result_sha256")
    protocol = json.loads((directory/"protocol.json").read_text())
    verify_payload(protocol, field="protocol_sha256")
    if result["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError(f"{directory.name}: result and protocol differ")
    return result, protocol


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, stem, output):
    for ext in FORMATS:
        fig.savefig(output/f"{stem}.{ext}", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def confirmation_figure(result, protocol, output, rows):
    methods = ["candidate_persistent", "candidate_reset", "frozen_cold_k8"]
    summary = result["summary"]
    order = protocol["core_conditions"]+[c for c in protocol["condition_order"] if c not in protocol["core_conditions"]]
    fig, (pooled, per) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True,
                                      gridspec_kw={"width_ratios": [1.1, 8], "wspace": 0.04})
    width = 0.26
    for i, m in enumerate(methods):
        label, color = METHODS[m]
        core = summary[m]["core_tbler"]*100
        pooled.bar(i*width, core, width, color=color, edgecolor=SURFACE, linewidth=2)
        pooled.text(i*width, core+0.4, f"{core:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        values = [summary[m]["per_condition_tbler"][c]*100 for c in order]
        per.bar([k+(i-1)*width for k in range(len(order))], values, width, color=color, edgecolor=SURFACE,
                linewidth=2, label=label)
        rows += [("confirmation_tbler_pct", m, "core_pooled", f"{core:.4f}")]
        rows += [("confirmation_tbler_pct", m, c, f"{v:.4f}") for c, v in zip(order, values, strict=True)]
    pooled.set_xticks([width])
    pooled.set_xticklabels(["Core\npooled"])
    per.set_xticks(range(len(order)))
    per.set_xticklabels([CONDITIONS.get(c, c) for c in order], fontsize=9.5)
    per.axvline(len(protocol["core_conditions"])-0.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    per.text(len(protocol["core_conditions"])-0.42, per.get_ylim()[1]*0.97, "reported only →",
             fontsize=9, color=MUTED, va="top")
    for ax in (pooled, per):
        style(ax)
    per.spines["left"].set_visible(False)
    per.tick_params(axis="y", length=0)
    pooled.set_ylabel("TBLER over TBs 17–64 (%) — lower is better")
    per.legend(frameon=False, loc="upper right", fontsize=9.5, labelcolor=INK)
    comparison = result["comparisons"]["candidate_reset"]
    fig.suptitle("Confirmation TBLER by condition", x=0.07, ha="left", fontsize=15, weight="bold", color=INK, y=1.03)
    update = Path(result['checkpoint']).name.split('_')[1]
    fig.text(0.07, 0.955, f"Checkpoint {update} • pooled memory benefit vs reset "
             f"{comparison['difference_percentage_points']:+.2f} pp "
             f"(95% CI {comparison['ci95'][0]*100:+.2f} to {comparison['ci95'][1]*100:+.2f})",
             fontsize=10.5, color=MUTED)
    save(fig, "confirmation_tbler", output)


def latency_figure(result, output, rows):
    receivers = ["candidate_persistent", "frozen_cold_k3"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    width = 0.36
    for i, m in enumerate(receivers):
        label, color = METHODS[m]
        values = [result["receivers"][m]["p50_ms"], result["receivers"][m]["p95_ms"]]
        positions = [k+(i-0.5)*width for k in range(2)]
        ax.bar(positions, values, width, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        for x, v in zip(positions, values, strict=True):
            ax.text(x, v+0.4, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
        rows += [("latency_ms", m, stat, f"{v:.4f}") for stat, v in zip(("p50", "p95"), values, strict=True)]
    style(ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["p50", "p95 (primary)"])
    ax.set_ylabel("Latency per TB (ms) — lower is better")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9.5, labelcolor=INK)
    ax.set_title("Trained-weight latency", loc="left", fontsize=14, weight="bold", color=INK, pad=22)
    ax.text(0, 1.03, "Prepared LS grid → decoded payload, incl. memory, detection, LDPC, CRC, transfers",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    save(fig, "latency", output)


def development_figure(results, selected, output, rows):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    updates = sorted(results)
    for m in [m for m in METHODS if all(m in results[u]["summary"] for u in updates)]:
        label, color = METHODS[m]
        values = [results[u]["summary"][m]["core_tbler"]*100 for u in updates]
        ax.plot(updates, values, color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        rows += [("development_core_tbler_pct", m, str(u), f"{v:.4f}") for u, v in zip(updates, values, strict=True)]
    ax.axvline(selected, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(selected, ax.get_ylim()[1], f" selected: {selected}", fontsize=9, color=MUTED, va="top")
    style(ax)
    ax.set_xticks(updates)
    ax.set_xlabel("Training update")
    ax.set_ylabel("Core pooled TBLER (%) — lower is better")
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK)
    ax.set_title("Development TBLER during training", loc="left", fontsize=14, weight="bold", color=INK, pad=22)
    ax.text(0, 1.03, "Fixed development bank; selection = lowest persistent TBLER", transform=ax.transAxes,
            fontsize=8.5, color=MUTED)
    save(fig, "development_tbler", output)


def main():
    parser = argparse.ArgumentParser(description="Plot sealed paper results; no receiver execution.")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection = json.loads((args.results/"selection.json").read_text())
    selected = int(Path(selection["checkpoint"]).name.split("_")[1])
    confirmation, protocol = load_result(args.results/f"confirmation_{selected:06d}")
    latency, _ = load_result(args.results/f"latency_{selected:06d}")
    development = {int(d.name.split("_")[1]): load_result(d)[0]
                   for d in sorted(args.results.glob("development_*")) if d.is_dir()}
    rows = []
    confirmation_figure(confirmation, protocol, output, rows)
    latency_figure(latency, output, rows)
    development_figure(development, selected, output, rows)
    with (output/"paper_results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("measure", "receiver", "condition_or_statistic", "value"))
        writer.writerows(rows)
    inputs = {"confirmation": confirmation, "latency": latency, **{f"development_{u:06d}": r
                                                                  for u, r in development.items()}}
    manifest = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "inputs_result_sha256": {k: r["result_sha256"] for k, r in inputs.items()},
                "selected_checkpoint_update": selected}
    (output/"paper_results_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"core_tbler_pct": {m: round(confirmation["summary"][m]["core_tbler"]*100, 3)
                                         for m in confirmation["summary"]},
                      "p95_ms": {m: round(v["p95_ms"], 3) for m, v in latency["receivers"].items()}}))


if __name__ == "__main__":
    main()
