"""Generate the development episodes of a radio configuration into its cache before training, so that the first
development check does not hold up the training (the 16-QAM runs found theirs already cached).

    pregenerate_dev.py --config nrx_large_64qam.cfg --gpu 0 [--part 0 --parts 2]

Uses the same protocol, plan, and episode bank as evaluate_dev.py, so the checks read exactly these files. With
--parts, each process generates every parts-th job (condition).
"""
import argparse
import json
import os
import time

from train import Run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import sionna as sn
    import tensorflow as tf
    from evaluate_dev import development_bank
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(20260930)
    sn.config.seed = 20260930
    run = Run(args.config, 1)
    _, plan, provider, _ = development_bank(args.config, run.anchors, run.cache, None, tf)
    for number, job in enumerate(plan["jobs"]):
        if number % args.parts != args.part:
            continue
        for index in range(int(job["episode_count"])):
            started = time.perf_counter()
            provider.generate(job, index)
            print(json.dumps({"condition": job["condition"], "episode": index,
                              "seconds": round(time.perf_counter()-started, 1)}), flush=True)


if __name__ == "__main__":
    main()
