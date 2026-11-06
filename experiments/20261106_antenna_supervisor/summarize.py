"""Check setting 0 against the sweep, then TBLER, SNR@TBLER=10%, and channel power per run."""
import json
from pathlib import Path

import numpy as np

from neural_rx.benchmark.plan import snr_at_target

HERE = Path(__file__).parent
SWEEP = HERE.parent/"20261104_tbler_sweep/results"
START = 16
MODES = ("fixed0", "fixed1", "fixed2", "fixed3", "oracle", "supervised", "nomem_fixed0", "nomem_oracle",
         "nomem_supervised")
SAVED = {"fixed0": "candidate_persistent", "nomem_fixed0": "candidate_reset"}


def saved_errors(condition, name, method):
    """The sweep's results at offsets -6..+8 dB, scored from TB 16 like the sweep."""
    low = np.load(SWEEP/"sweep"/condition/method/name)["errors"]
    high = np.load(SWEEP/"sweep_high"/condition/method/name)["errors"]
    return np.concatenate([low, high[1:]])[:, START:]


def main():
    report = {}
    for folder in sorted(p for p in (HERE/"results").iterdir() if p.is_dir()):
        names = sorted(set.intersection(*({p.name for p in (folder/mode).glob("episode_*.npz")} for mode in MODES)))
        if not names:
            continue
        result = {"episodes": len(names), "sweep_mismatches": 0}
        for mode in MODES:
            failures, blocks, switches, gain, share = 0, 0, 0, [], np.zeros(4)
            for name in names:
                data = np.load(folder/mode/name)
                active = data["active"][START:]
                failures = failures+(data["errors"][:, START:] & active[None]).sum((1, 2))
                blocks += int(active.sum())
                switches += int(data["switches"].sum())
                setting = data["setting"][:, START:]
                share += np.bincount(setting[:, active].ravel(), minlength=4)
                power = data["channel_power"][START:]
                chosen = np.take_along_axis(power[None], setting[..., None].astype(int), axis=-1)[..., 0]
                gain.append(10*np.log10(chosen[:, active]/power[active][None, :, 0]).ravel())
                if mode in SAVED:
                    saved = saved_errors(folder.name, name, SAVED[mode])
                    result["sweep_mismatches"] += int((data["errors"][-len(saved):, START:] != saved).sum())
            snr = data["snr_db"]
            tbler = failures/blocks
            result[mode] = {"snr_db": snr.tolist(), "tbler": tbler.tolist(),
                            "snr_at_10pct_db": snr_at_target(snr, tbler, .1),
                            "switches_per_sequence_and_snr": switches/len(names)/len(snr),
                            "setting_share": (share/share.sum()).round(3).tolist(),
                            "median_power_gain_db": float(np.median(np.concatenate(gain)))}
        report[folder.name] = result
    (HERE/"summary.json").write_text(json.dumps(report, indent=1))
    for condition, result in report.items():
        print(f"{condition}: {result['episodes']} sequences, mismatches with the sweep {result['sweep_mismatches']}")
        for mode in MODES:
            value = result[mode]
            required = value["snr_at_10pct_db"]
            print(f"  {mode:16s} SNR@10% {'n/a' if required is None else f'{required:6.2f}'} dB  "
                  f"power gain {value['median_power_gain_db']:6.2f} dB  "
                  f"switches {value['switches_per_sequence_and_snr']:.2f}  share {value['setting_share']}")


if __name__ == "__main__":
    main()
