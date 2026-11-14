"""Retrain the Large NRX with two iterations on the proposed receiver's training data and budget.

The proposed receiver's selected checkpoint has 22,000 updates from the shipped Large NRX: 17,000 parent updates
(initializer_parent_updates in its recipe) plus 5,000 in its final run. The baseline gets the same 22,000 updates with
the same recipe, seed, optimizer, learning-rate policy, and development checks every 1,000 updates, and the
checkpoint with the lowest development core TBLER is selected (earlier one on ties), as for the proposed receiver.
Only the receiver differs: Large NRX iterations 1-2 on the LS estimate and plain BP (receiver_variant
cgnn_k2_baseline), without memory, filter, or IDD.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = ROOT/"models/retrained_k2_baseline_20261113"
STOP = 17000+5000
POINTS = list(range(1000, STOP+1, 1000))
EXECUTION = ["--gpu", "0", "--data-gpu", "1", "--replica-gpu", "1", "--detector-batch-tbs", "8",
             "--descriptor-chunk-tbs", "32", "--render-batch-tbs", "16", "--prefetch-examples", "8",
             "--prefetch-memory-mib", "4096", "--channel-workers", "1"]


def prune(models, current):
    """Keep checkpoint 0, every development checkpoint, and anything newer; the disk is nearly full."""
    for path in models.glob("checkpoint_*"):
        update = int(path.name.split("_")[1])
        if 0 < update < current and update % 1000:
            shutil.rmtree(path)


def evaluate(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    update = int(checkpoint.name.split("_")[1])
    output = HERE/"results"/f"development_{update:06d}"
    if not (output/"results.json").is_file():
        with (HERE/"logs"/f"development_{update:06d}.log").open("w") as log:
            subprocess.run([sys.executable, str(HERE/"evaluate_dev.py"), "--checkpoint", str(checkpoint),
                            "--gpu", "1", "--anchors", str(HERE/"inputs/anchors.json"),
                            "--episode-cache", str(HERE/"episodes/development"), "--output-dir", str(output)],
                           check=True, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="2"))
    prune(checkpoint.parent, update)
    return json.loads((output/"results.json").read_text())


def select():
    results = [json.loads((HERE/"results"/f"development_{u:06d}"/"results.json").read_text()) for u in POINTS]
    best = min(results, key=lambda r: (r["summary"]["candidate_persistent"]["core_tbler"], r["update"]))
    report = {"checkpoint": best["checkpoint"], "update": best["update"],
              "core_tbler": best["summary"]["candidate_persistent"]["core_tbler"],
              "development": {r["update"]: r["summary"]["candidate_persistent"]["core_tbler"] for r in results}}
    (HERE/"results/best.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


def main():
    from neural_rx.cli.train import parser, run_training
    (HERE/"logs").mkdir(exist_ok=True)
    argv = ["--recipe", str(HERE/"inputs/training_recipe.json"), "--steps", str(STOP),
            "--output-dir", str(MODELS), *EXECUTION]
    latest = MODELS/"latest.json"
    if latest.is_file():
        argv += ["--resume", str(MODELS/json.loads(latest.read_text())["checkpoint"])]
    run_training(parser().parse_args(argv), evaluation_points=POINTS, evaluate=evaluate)
    select()


if __name__ == "__main__":
    main()
