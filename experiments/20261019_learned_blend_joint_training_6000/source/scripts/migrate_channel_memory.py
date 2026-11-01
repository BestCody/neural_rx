from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from cnn_autoencoder.training_options import configure_gpu
    tf = configure_gpu(args.gpu)
    from artifact_io import atomic_json, file_sha256
    from temporal_receiver_core import build_backbone, prepare_cgnn_inputs
    from cnn_autoencoder.migration import load_complete_receiver
    from cnn_autoencoder.model import TemporalUEMemoryCGNN, architecture_settings
    from cnn_autoencoder.pilot_layout import PilotLayout
    tf.keras.utils.set_random_seed(args.seed)
    recipe = json.loads((ROOT/"config/cnn_memory_training.json").read_text())
    settings = json.loads((ROOT/"config/cnn_autoencoder_finalist.json").read_text())
    source_settings = json.loads((args.source.parent/"cnn_autoencoder_finalist.json").read_text())
    source_architecture = dict(source_settings["architecture"])
    if source_architecture.pop("memory") != "pilot_filter_calibrated_feedback":
        raise ValueError("Expected the complete preceding calibrated-feedback receiver")
    target_architecture = dict(settings["architecture"])
    target_architecture.pop("memory")
    target_architecture.pop("blend_features")
    target_architecture.pop("blend_hidden")
    if source_architecture != target_architecture:
        raise ValueError("Migration must preserve the inherited detector, memory, and decoder dimensions")
    p, e2e = build_backbone(recipe["radio_config"], 2, False, n_size_bwp_override=min(recipe["prbs"]))
    receiver, tx = e2e._receiver, p.transmitters[0]
    base = receiver._neural_rx._cgnn
    dimensions = architecture_settings(settings)
    layout = PilotLayout(receiver._ls_est, tx.resource_grid, dimensions["neighbor_radius"])
    model = TemporalUEMemoryCGNN(base, layout, tx._mapper, **dimensions, name="channel_memory_k2")
    y = tf.zeros([1, 1, p.num_rx_antennas, layout.symbols, layout.frequency], tf.complex64)
    ls = tf.zeros([1, layout.users, layout.frequency, layout.symbols, 2*p.num_rx_antennas])
    model(prepare_cgnn_inputs(receiver, y, ls, tf.ones([1, layout.users])), tf.ones([1]),
          ue_ids=tf.range(layout.users)[None])
    mapping = load_complete_receiver(model, args.source, args.source_sha256)
    args.output.mkdir(parents=True)
    path = args.output/"initial_receiver.weights.h5"
    model.save_weights(str(path))
    atomic_json(args.output/"cnn_autoencoder_finalist.json", settings)
    atomic_json(args.output/"initialization.json", dict(source=str(args.source.resolve()),
        source_sha256=args.source_sha256, seed=args.seed, weights_sha256=file_sha256(path),
        retained="Every preceding receiver tensor, including channel estimates, uncertainty, calibration, and correction gain",
        initialized="Glorot blend kernel, zero hidden bias and output layer; initial blend coefficient 0.5",
        optimizer="new Adam state required", training_updates=0,
        parent_updates=json.loads((args.source.parent/"manifest.json").read_text())["update"],
        tensor_mapping=mapping))
    print(json.dumps(dict(weights=str(path), sha256=file_sha256(path)), indent=2))


if __name__ == "__main__":
    main()
