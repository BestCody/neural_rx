#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path


from neural_rx.paths import ROOT


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate receivers against pretrained neural_rx references.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--experiment-checkpoint", type=Path)
    parser.add_argument("--experiment-phase", default="smoke",
                        choices=("smoke", "latency", "calibration", "development", "confirmation"))
    parser.add_argument("--experiment-anchors", type=Path)
    parser.add_argument("--experiment-prbs", type=int, default=132)
    parser.add_argument("--experiment-episodes", type=int)
    parser.add_argument("--benchmark-protocol", type=Path, default=ROOT / "config/temporal_benchmark_v2.json",
                        help="Declared scenarios and scoring; v2 adds the ray-traced Munich report condition")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Evaluation artifacts under experiments/YYYYMMDD_descriptive_name/results/")
    parser.add_argument("--episode-cache", type=Path, default=ROOT / "datasets/evaluation",
                        help="Episode directory; a dataset.json manifest selects a fixed read-only dataset")
    parser.add_argument("--episode-cache-readonly", action="store_true")
    parser.add_argument("--rt-episode-cache", type=Path, default=ROOT / "datasets/rt_munich/episodes",
                        help="Episode directory for ray-traced conditions")
    parser.add_argument("--reference-cache", type=Path, default=ROOT / "results/historical/reference_cache",
                        help="Directory for reusable frozen Cold receiver results")
    return parser


def main():
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from neural_rx.benchmark.frozen import run_experiment

    run_experiment(args)


if __name__ == "__main__":
    main()
