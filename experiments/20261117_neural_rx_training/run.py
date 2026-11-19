"""Stage 1: train neural_rx (the Large NRX) with two iterations on our sequences.

Starts from the shipped Large NRX and runs until it is stopped (./run.sh stop), with a development check every 1,000
updates; results/best.json always names the checkpoint with the lowest development core TBLER so far (earlier one on
ties). Loss: the binary cross-entropy of the LLRs against the transmitted bits and against the soft bits of the shipped
eight-iteration network, plus the original NRX's channel-estimate term.

Stage 2 (experiments/20261118_no_teacher_training) adds the pilot memory on top of this stage's best checkpoint,
with the same sequences and without the term for the eight-iteration network.

The gradients are computed on both GPUs (the learner on GPU 0, a second gradient process on GPU 1), data generation
runs on GPU 1, and the development checks run on GPU 0 next to the learner (GPU 1 is fully reserved). Training
continues during a check; its result reaches the learning-rate schedule 300 updates after its checkpoint.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = ROOT/"models/neural_rx_k2_20261117"
INTERVAL = 1000              # updates between development checks
DELAY = 300                  # a check runs alongside training; its result is used this many updates later
STOP = 1000*INTERVAL         # far beyond any real run: the training ends with ./run.sh stop
POINTS = range(INTERVAL, STOP+1, INTERVAL)
LEARNER_GPU, OTHER_GPU = "0", "1"
EXECUTION = ["--gpu", LEARNER_GPU, "--data-gpu", OTHER_GPU, "--replica-gpu", OTHER_GPU, "--detector-batch-tbs", "8",
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
                            "--gpu", LEARNER_GPU, "--anchors", str(HERE/"inputs/anchors.json"),
                            "--episode-cache", str(ROOT/"experiments/20261113_retrained_k2_baseline/episodes/development"), "--output-dir", str(output)],
                           check=True, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="2"))
    prune(checkpoint.parent, update)
    record_best()
    return json.loads((output/"results.json").read_text())


def record_best():
    """Write the best development checkpoint so far: lowest core TBLER, the earlier one on ties."""
    results = [json.loads(path.read_text()) for path in sorted((HERE/"results").glob("development_*/results.json"))]
    best = min(results, key=lambda r: (r["summary"]["candidate_persistent"]["core_tbler"], r["update"]))
    report = {"checkpoint": best["checkpoint"], "update": best["update"],
              "core_tbler": best["summary"]["candidate_persistent"]["core_tbler"],
              "development": {r["update"]: r["summary"]["candidate_persistent"]["core_tbler"] for r in results}}
    temporary = HERE/"results/best.json.tmp"
    temporary.write_text(json.dumps(report, indent=1))
    temporary.replace(HERE/"results/best.json")


def main():
    from neural_rx.cli.train import parser, run_training
    (HERE/"logs").mkdir(exist_ok=True)
    argv = ["--recipe", str(HERE/"inputs/training_recipe.json"), "--steps", str(STOP),
            "--output-dir", str(MODELS), *EXECUTION]
    latest = MODELS/"latest.json"
    if latest.is_file():
        argv += ["--resume", str(MODELS/json.loads(latest.read_text())["checkpoint"])]
    run_training(parser().parse_args(argv), evaluation_points=POINTS, evaluate=evaluate, evaluation_delay=DELAY)


if __name__ == "__main__":
    main()
