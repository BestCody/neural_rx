#!/usr/bin/python3

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import argparse
import fcntl
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent


def training_paths(label, output_dir, initial_weights=None):
    if Path(label).name != label:
        raise ValueError("Training label must be a filename")
    directory = Path(output_dir).expanduser().resolve()
    destination = directory / f"{label}_weights"
    if destination.resolve().is_relative_to((ROOT / "models/pretrained").resolve()):
        raise ValueError("Training output must be outside the pretrained weights directory")
    if destination.exists():
        raise FileExistsError("Training output already exists; select a new output directory")
    initial = (ROOT / "models/pretrained" / f"{label}_weights" if initial_weights is None
               else Path(initial_weights).expanduser().resolve())
    if not initial.is_file():
        raise FileNotFoundError(initial)
    return initial, destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_name", help="config filename", type=str, required=True)
    parser.add_argument("-gpu", help="GPU to use", type=int, default=0)
    parser.add_argument("-debug", help="Set debugging configuration", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    sys.path.insert(0, str(ROOT))
    from utils import E2E_Model, training_loop, Parameters, load_weights
    parameters = Parameters(args.config_name, training=True)
    label = str(parameters.label)
    initial, destination = training_paths(label, args.output_dir, args.initial_weights)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial, destination = training_paths(label, args.output_dir, args.initial_weights)
        if args.debug:
            tf.config.run_functions_eagerly(True)
        model = E2E_Model(parameters, training=True)
        model(1, 1.)
        model.summary()
        load_weights(model, str(initial))
        training_loop(model,
            label=label, filename=str(destination),
            training_logdir=str(destination.parent / 'logs'), training_seed=42,
            training_schedule=parameters.training_schedule,
            eval_ebno_db_arr=parameters.eval_ebno_db_arr,
            min_num_tx=parameters.min_num_tx, max_num_tx=parameters.max_num_tx,
            sys_parameters=parameters,
            mcs_arr_training_idx=list(range(len(parameters.mcs_index))),
            xla=parameters.xla)


if __name__ == "__main__":
    main()
