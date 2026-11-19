"""TBLER so far, on the test sequences that all listed receivers have decoded (paired; not the final summary)."""
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STORED = {"sweep": ROOT/"experiments/20261104_tbler_sweep/results/sweep", "sweep_high": ROOT/"experiments/20261104_tbler_sweep/results/sweep_high",
          "power_control": ROOT/"experiments/20261112_early_retransmission/results/receivers_pc"}
NEW = ("pilot_memory", "pilot_memory_emptied", "neural_rx_k2_trained")
for setting in ("sweep", "sweep_high", "power_control"):
    for condition in ("natural_umi", "controlled_topology_change", "schedule_gap_reentry"):
        done = sorted(p.name for p in (HERE/"results"/setting/condition/"neural_rx_k2_trained").glob("episode_*.npz"))
        done = [n for n in done if all((HERE/"results"/setting/condition/r/n).is_file() for r in NEW)]
        if not done:
            continue
        rows = {}
        for label, base, method in [(r, HERE/"results"/setting, r) for r in NEW]+[("released 2 it", STORED[setting], "frozen_cold_k2"),
                                                                                  ("released 8 it", STORED[setting], "frozen_cold_k8")]:
            errors, blocks, snr = 0, 0, None
            for name in done:
                d = np.load(base/condition/method/name)
                active = d["active"][16:].astype(bool)
                errors = errors+(d["errors"][:, 16:] & active[None]).sum((1, 2)); blocks += int(active.sum()); snr = d["snr_db"]
            rows[label] = 100*errors/blocks
        print(f"{setting} / {condition}: {len(done)} sequences, {blocks} blocks per SNR point")
        print("   SNR dB".ljust(24), " ".join(f"{s:6.1f}" for s in snr))
        for label, values in rows.items():
            print(f"   {label:21s}", " ".join(f"{v:6.2f}" for v in values))
