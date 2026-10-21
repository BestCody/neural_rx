#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate receivers against pretrained neural_rx references.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--experiment-checkpoint", type=Path)
    parser.add_argument("--experiment-phase", choices=("smoke", "latency", "calibration", "development", "confirmation"), default="smoke")
    parser.add_argument("--experiment-anchors", type=Path)
    parser.add_argument("--experiment-prbs", type=int, default=132)
    parser.add_argument("--experiment-episodes", type=int)
    parser.add_argument("--rt-channel-bank", type=Path,
                        help="Exported Sionna RT path bank; evaluate it as a separate scenario")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Evaluation artifacts under experiments/YYYYMMDD_descriptive_name/results/")
    parser.add_argument("--episode-cache", type=Path, default=ROOT / "datasets/evaluation",
                        help="Episode directory; a dataset.json manifest selects a fixed read-only dataset")
    parser.add_argument("--episode-cache-readonly", action="store_true")
    parser.add_argument("--reference-cache", type=Path, default=ROOT / "results/historical/reference_cache",
                        help="Directory for reusable frozen Cold receiver results")
    return parser


def main():
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from temporal_benchmark_frozen import run_experiment

    run_experiment(args)


if __name__ == "__main__":
    main()
