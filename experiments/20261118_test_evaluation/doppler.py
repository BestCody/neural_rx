"""Required effective SNR at 10% TBLER on the UMi test sequences, per Doppler group and receiver.

Effective SNR = SNR + the UE's effective channel power in dB (stored with the sweep of
experiments/20261104_tbler_sweep; every receiver decoded the same sequences). The Doppler group of a UMi test
sequence is its index modulo 3: 0-50 Hz, 50-175 Hz, 175-400 Hz. Uses the sequences that all receivers have decoded;
writes results/doppler.json when all 32 are there.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STORED = ROOT/"experiments/20261104_tbler_sweep/results"
START = 16
GROUPS = ("0-50", "50-175", "175-400", "all")
RECEIVERS = {"pilot_memory": HERE/"results", "pilot_memory_emptied": HERE/"results", "neural_rx_k2_trained": HERE/"results",
             "frozen_cold_k2": STORED, "frozen_cold_k3": STORED, "frozen_cold_k8": STORED,
             "baseline_lmmse_kbest": STORED, "baseline_perf_csi_kbest": STORED}


def main(condition="natural_umi"):
    from neural_rx.benchmark.plan import decreasing_isotonic, snr_at_target
    edges = np.arange(-6., 14.)
    centers = edges[:-1]+.5
    names = sorted(p.name for p in (STORED/"sweep"/condition/"candidate_persistent").glob("episode_*.npz"))
    done = [n for n in names if all((base/s/condition/r/n).is_file() for r, base in RECEIVERS.items()
                                    for s in ("sweep", "sweep_high"))]
    table = {}
    for receiver, base in RECEIVERS.items():
        table[receiver] = {}
        for group in GROUPS:
            failures, blocks = np.zeros(len(centers)), np.zeros(len(centers))
            for directory, first in (("sweep", 0), ("sweep_high", 1)):       # +3 dB appears in both sweeps
                for name in done:
                    if group != "all" and int(name.split("_")[1][:5]) % 3 != GROUPS.index(group):
                        continue
                    with np.load(STORED/directory/condition/"candidate_persistent"/name) as reference, \
                            np.load(base/directory/condition/receiver/name) as data:
                        if not np.array_equal(reference["fingerprint"], data["fingerprint"]):
                            raise ValueError(f"{receiver}/{name}: decoded on a different sequence")
                        active = reference["active"][START:].astype(bool)
                        power = 10*np.log10(reference["channel_power"][START:])
                        for point in range(first, len(data["snr_db"])):
                            index = np.floor(data["snr_db"][point]+power-edges[0]).astype(int)
                            valid = (index >= 0) & (index < len(centers)) & active
                            np.add.at(failures, index[valid], data["errors"][point, START:][valid].astype(float))
                            np.add.at(blocks, index[valid], 1)
            rates = failures/np.maximum(blocks, 1)
            table[receiver][group] = float(snr_at_target(centers, decreasing_isotonic(rates, np.maximum(blocks, 1)), .1))
    print(f"{condition}: {len(done)} of {len(names)} sequences; required effective SNR at 10% TBLER (dB)")
    print(f"  {'':24s}" + "".join(f"{g:>10s}" for g in GROUPS))
    for receiver, row in table.items():
        print(f"  {receiver:24s}" + "".join(f"{row[g]:10.2f}" for g in GROUPS))
    if len(done) == len(names):
        (HERE/"results/doppler.json").write_text(json.dumps({"condition": condition, "sequences": len(done),
                                                             "required_effective_snr_db": table}, indent=1))


if __name__ == "__main__":
    main()
