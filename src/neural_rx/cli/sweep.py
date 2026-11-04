#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description="TBLER-vs-SNR sweep with classical baselines.")
    parser.add_argument("--gpu", type=int, default=0)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--condition", required=True)
    run.add_argument("--methods", nargs="+", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--covariance-dir", type=Path, required=True)
    run.add_argument("--first-episode", type=int, default=0)
    run.add_argument("--last-episode", type=int, default=32)
    run.add_argument("--snr-offsets", type=float, nargs="+", help="Override the SNR grid around the operating point")

    covariance = commands.add_parser("covariance")
    covariance.add_argument("--conditions", nargs="+", required=True)
    covariance.add_argument("--covariance-dir", type=Path, required=True)
    covariance.add_argument("--draws", type=int, default=64)
    covariance.add_argument("--batch-size", type=int, default=32)

    reproduce = commands.add_parser("reproduce")
    reproduce.add_argument("--sealed", type=Path, required=True)
    reproduce.add_argument("--conditions", nargs="+", required=True)
    reproduce.add_argument("--methods", nargs="+", required=True)
    reproduce.add_argument("--episodes", type=int, default=2)

    double = commands.add_parser("doubletdl")
    double.add_argument("--systems", nargs="+", required=True)
    double.add_argument("--snr-db", type=float, nargs="+", required=True)
    double.add_argument("--batch-size", type=int, default=7)
    double.add_argument("--batches", type=int, default=30)
    double.add_argument("--covariance-dir", type=Path, required=True)
    double.add_argument("--output", type=Path, required=True)

    latency = commands.add_parser("latency")
    latency.add_argument("--methods", nargs="+", required=True)
    latency.add_argument("--output", type=Path, required=True)

    summary = commands.add_parser("summarize")
    summary.add_argument("--results-dir", type=Path, nargs="+", required=True)
    summary.add_argument("--output", type=Path, required=True)
    return parser


def main():
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.command == "summarize" else str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from neural_rx.benchmark import tbler_sweep

    if args.command == "summarize":
        args.output.write_text(json.dumps(tbler_sweep.summarize(args.results_dir), indent=1))
        return
    {"run": tbler_sweep.run_sweep, "covariance": tbler_sweep.run_covariance,
     "reproduce": tbler_sweep.run_reproduce, "doubletdl": tbler_sweep.run_doubletdl,
     "latency": tbler_sweep.run_latency}[args.command](args)


if __name__ == "__main__":
    main()
