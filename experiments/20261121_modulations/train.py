"""Both training stages of the 16-QAM receivers at another radio configuration (MCS), e.g. 64-QAM or QPSK.

    train.py --config nrx_large_64qam.cfg --stage 1      the Large NRX retrained with two iterations
    train.py --config nrx_large_64qam.cfg --stage 2      the proposed receiver, starting from stage 1's best checkpoint

Each stage repeats its 16-QAM run (experiments/20261117_neural_rx_training, experiments/20261118_no_teacher_training)
with the same recipe, data, schedule, and development checks, except:
- radio_config is the given configuration, so data, LDPC code, readout, and the stage-1 teacher (the released
  eight-iteration NRX) are those of its MCS;
- stage 1 starts from the released weights of that MCS, stage 2 from stage 1's best checkpoint;
- the training SNR ranges (noise_schedule core_range_db and range_db) are shifted by the median difference between
  this configuration's calibration anchors and the 16-QAM ones, so the training covers the same TBLER region;
- the development set (the same 8 conditions x 32 sequences, at this configuration's anchors) is cached in /dev/shm.
A stage ends by itself when PATIENCE development checks in a row did not improve on the best one (the 16-QAM stage 1
ended three checks after its best checkpoint), or at MAX_UPDATES; results/<label>/stage<N>/finished.json records it.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_RECIPES = {1: ROOT/"experiments/20261117_neural_rx_training/inputs/training_recipe.json",
                2: ROOT/"experiments/20261118_no_teacher_training/inputs/training_recipe.json"}
ANCHORS_16QAM = ROOT/"experiments/20261117_neural_rx_training/inputs/anchors.json"
CACHE = Path("/dev/shm/nrx_modulations")     # approved by the user for these caches (2026-11-21)
INTERVAL = 1000              # updates between development checks
DELAY = 300                  # a check runs alongside training; its result is used this many updates later
PATIENCE = 3                 # checks in a row without a new best before the stage ends
MAX_UPDATES = 20000         # lowered from 30000 on 2026-11-22: 64-QAM stage 1 gained 0.08 points of development TBLER after 20000
POINTS = range(INTERVAL, MAX_UPDATES+1, INTERVAL)
LEARNER_GPU, OTHER_GPU = "0", "1"
EXECUTION = ["--gpu", LEARNER_GPU, "--data-gpu", OTHER_GPU, "--replica-gpu", OTHER_GPU, "--detector-batch-tbs", "8",
             "--descriptor-chunk-tbs", "32", "--render-batch-tbs", "16", "--prefetch-examples", "8",
             "--prefetch-memory-mib", "4096", "--channel-workers", "1"]


class Run:
    def __init__(self, config, stage):
        self.config, self.stage, self.label = config, stage, Path(config).stem
        self.results = HERE/"results"/self.label/f"stage{stage}"
        self.logs = HERE/"logs"/self.label
        self.models = ROOT/"models"/(f"{self.label}_k2_20261121" if stage == 1 else f"{self.label}_pilot_memory_20261121")
        self.anchors = HERE/"results"/self.label/"calibration/anchors.json"
        self.recipe = self.results/"recipe.json"
        self.cache = CACHE/self.label/"development"

    def snr_shift(self):
        old = json.loads(ANCHORS_16QAM.read_text())["conditions"]
        new = json.loads(self.anchors.read_text())["conditions"]
        return float(np.median([new[c]["anchor_snr_db"]-old[c]["anchor_snr_db"] for c in old]))

    def write_recipe(self):
        """At the first start: the 16-QAM recipe of this stage for this configuration. Later starts reuse it."""
        if self.recipe.is_file():
            return
        from neural_rx.provenance.artifacts import file_sha256
        recipe = json.loads(BASE_RECIPES[self.stage].read_text())
        if self.stage == 1:
            initializer = ROOT/"models/pretrained"/f"{self.label}_weights"
            recipe.update(initializer_checkpoint=str(initializer.relative_to(ROOT)),
                          initializer_sha256=file_sha256(initializer), initializer_parent_updates=0)
        else:
            best = Run(self.config, 1).results/"best.json"
            if not (Run(self.config, 1).results/"finished.json").is_file():
                raise SystemExit("Stage 1 of this configuration has not finished")
            checkpoint = Path(json.loads(best.read_text())["checkpoint"])
            manifest = json.loads((checkpoint/"manifest.json").read_text())
            recipe.update(initializer_checkpoint=str((checkpoint/"receiver.weights.h5").relative_to(ROOT)),
                          initializer_sha256=manifest["files"]["receiver.weights.h5"],
                          initializer_parent_updates=manifest["update"])
        shift = self.snr_shift()
        noise = recipe["noise_schedule"]
        noise["core_range_db"] = [value+shift for value in noise["core_range_db"]]
        noise["range_db"] = [value+shift for value in noise["range_db"]]
        recipe["radio_config"] = self.config
        self.results.mkdir(parents=True, exist_ok=True)
        self.recipe.write_text(json.dumps(recipe, indent=1)+"\n")
        print(json.dumps({"recipe": str(self.recipe), "snr_shift_db": shift,
                          "initializer": recipe["initializer_checkpoint"]}), flush=True)

    def prune(self, current):
        """Keep checkpoint 0, every development checkpoint, and anything newer; the disk is nearly full."""
        for path in self.models.glob("checkpoint_*"):
            update = int(path.name.split("_")[1])
            if 0 < update < current and update % INTERVAL:
                shutil.rmtree(path)

    def evaluate(self, checkpoint):
        checkpoint = Path(checkpoint).resolve()
        update = int(checkpoint.name.split("_")[1])
        output = self.results/f"development_{update:06d}"
        if not (output/"results.json").is_file():
            with (self.logs/f"stage{self.stage}_development_{update:06d}.log").open("w") as log:
                subprocess.run([sys.executable, str(HERE/"evaluate_dev.py"), "--stage", str(self.stage),
                                "--config", self.config, "--checkpoint", str(checkpoint), "--gpu", LEARNER_GPU,
                                "--anchors", str(self.anchors), "--episode-cache", str(self.cache),
                                "--output-dir", str(output)],
                               check=True, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="2"))
        self.prune(update)
        best = self.record_best()
        if len([u for u in best["development"] if int(u) > best["update"]]) >= PATIENCE or update >= MAX_UPDATES:
            (self.results/"finished.json").write_text(json.dumps(best, indent=1))
            print(json.dumps({"finished": True, "best": best["update"], "last": update}), flush=True)
            os.killpg(os.getpgid(0), signal.SIGINT)        # as ./run.sh stop: Ctrl-C to the training's processes
        return json.loads((output/"results.json").read_text())

    def record_best(self):
        """The best development checkpoint so far: lowest core TBLER, the earlier one on ties."""
        results = [json.loads(path.read_text()) for path in sorted(self.results.glob("development_*/results.json"))]
        best = min(results, key=lambda r: (r["summary"]["candidate_persistent"]["core_tbler"], r["update"]))
        report = {"checkpoint": best["checkpoint"], "update": best["update"],
                  "core_tbler": best["summary"]["candidate_persistent"]["core_tbler"],
                  "development": {r["update"]: r["summary"]["candidate_persistent"]["core_tbler"] for r in results}}
        temporary = self.results/"best.json.tmp"
        temporary.write_text(json.dumps(report, indent=1))
        temporary.replace(self.results/"best.json")
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    run = Run(args.config, args.stage)
    if (run.results/"finished.json").is_file():
        print(f"{run.label} stage {run.stage} has finished", flush=True)
        return
    run.logs.mkdir(parents=True, exist_ok=True)
    run.write_recipe()
    from neural_rx.cli.train import parser as train_parser, run_training
    argv = ["--recipe", str(run.recipe), "--steps", str(MAX_UPDATES), "--output-dir", str(run.models), *EXECUTION]
    latest = run.models/"latest.json"
    if latest.is_file():
        argv += ["--resume", str(run.models/json.loads(latest.read_text())["checkpoint"])]
    run_training(train_parser().parse_args(argv), evaluation_points=POINTS, evaluate=run.evaluate,
                 evaluation_delay=DELAY)


if __name__ == "__main__":
    main()
