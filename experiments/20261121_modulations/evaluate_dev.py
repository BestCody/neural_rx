"""Development TBLER of one training checkpoint at any radio configuration (MCS), scored as for 16-QAM.

Stage 1 (--stage 1): the retrained two-iteration Large NRX, i.e., the benchmark's two-iteration adapter loading the
checkpoint's nrx_weights.pkl, as experiments/20261117_neural_rx_training/evaluate_dev.py does for 16-QAM. Stage 2
(--stage 2): the proposed receiver, i.e., the benchmark's candidate adapter loading the checkpoint, as
experiments/20261118_no_teacher_training/evaluate_dev.py does. Same development protocol, seeds and core-condition
scoring; the anchors and the released teacher are those of the radio configuration. The result is stored under
summary["candidate_persistent"], which the trainer reads.
"""
import argparse
import json
import os
from pathlib import Path


def development_bank(config, anchors, episode_cache, episodes, tf):
    """The development protocol, plan, and cached episode bank of a radio configuration, as in the 16-QAM checks."""
    from neural_rx.backbone import E2E_Model, Parameters
    from neural_rx.benchmark.candidate import pretrained_checkpoint
    from neural_rx.benchmark.episodes import CachedScenarioBank, scenario_identity
    from neural_rx.benchmark.frozen import load_ray_traced_banks, protocol_and_plan
    from neural_rx.benchmark.generalization_protocol import seal_payload, verify_payload
    from neural_rx.benchmark.scenarios import TemporalBenchmarkScenarioBank
    from neural_rx.paths import ROOT
    from neural_rx.provenance.artifacts import file_sha256

    artifact = json.loads(anchors.read_text())
    verify_payload(artifact, field="manifest_sha256")
    if artifact["teacher_sha256"] != file_sha256(pretrained_checkpoint(config)) or artifact["prbs"] != 132:
        raise ValueError("Calibration reference differs")
    protocol, plan = protocol_and_plan("development", 132, episodes, artifact["conditions"],
                                       config_name=config)
    # As in the 16-QAM development checks of both stages, so that both stages score the same cached episodes.
    protocol["methods"] = protocol["methods"]+["frozen_cold_k2"]
    protocol["execution"]["accuracy_reference_methods"] = ["frozen_cold_k2"]
    rt_banks = load_ray_traced_banks(protocol, plan)
    if rt_banks:
        raise ValueError("The development conditions are not expected to include ray-traced channels")
    identity = scenario_identity(ROOT/"config"/config, protocol)
    parameters = Parameters(config, training=False, num_tx_eval=2, n_size_bwp_override=132)
    e2e = E2E_Model(parameters, training=False, mcs_arr_eval_idx=0)
    provider = CachedScenarioBank(TemporalBenchmarkScenarioBank(parameters, e2e, tf, rt_banks=rt_banks),
                                  episode_cache, protocol, identity)
    identity = provider.identity
    if artifact["source_identity"]["radio"] != identity["radio"]:
        raise ValueError("Calibration radio configuration differs")
    protocol["calibration"]["source_manifest_sha256"] = artifact["manifest_sha256"]
    protocol = seal_payload(protocol, field="protocol_sha256")
    plan["protocol_sha256"] = protocol["protocol_sha256"]
    provider.ensure_capacity(plan["jobs"])

    return protocol, plan, provider, identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--config", required=True, help="Radio configuration, e.g. nrx_large_64qam.cfg")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--episode-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, help="Fewer episodes per condition, for smoke tests only")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import numpy as np
    import sionna as sn
    import tensorflow as tf
    from neural_rx.backbone.training import load_weights
    from neural_rx.benchmark.frozen import make_benchmark_adapter, make_manifest
    from neural_rx.benchmark.generalization_protocol import seal_payload
    from neural_rx.benchmark.runtime import evaluate_method_set
    from neural_rx.benchmark.scoring import method_summary
    from neural_rx.provenance.artifacts import atomic_json, file_sha256

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(20260930)
    sn.config.seed = 20260930

    checkpoint = args.checkpoint.resolve()
    saved = json.loads((checkpoint/"manifest.json").read_text())
    name = "nrx_weights.pkl" if args.stage == 1 else "receiver.weights.h5"
    weights = checkpoint/name
    if file_sha256(weights) != saved["files"][name]:
        raise ValueError("Weights differ from the checkpoint manifest")
    variant = saved["binding"]["recipe"].get("receiver_variant", "pilot_memory")
    if variant != ("cgnn_k2_baseline" if args.stage == 1 else "pilot_memory"):
        raise ValueError(f"Checkpoint of {variant} is not a stage-{args.stage} checkpoint")
    if Path(saved["binding"]["recipe"]["radio_config"]).name != args.config:
        raise ValueError("Checkpoint was trained for another radio configuration")

    protocol, plan, provider, identity = development_bank(args.config, args.anchors, args.episode_cache, args.episodes, tf)

    if args.stage == 1:
        method = "frozen_cold_k2"
        manifest, mapping, seed = make_manifest(protocol, method)
        adapter = make_benchmark_adapter(manifest=manifest, method=method, training_seed=seed, artifact_map=mapping,
                                         config_name=args.config, tf_module=tf)
        load_weights(adapter.model, str(weights))
        adapter._model_hash = adapter.model_checksum(adapter.model)
        adapters = {method: adapter}
        description = "retrained two-iteration Large NRX (benchmark frozen_cold_k2 adapter, retrained weights)"
    else:
        method = "candidate_persistent"
        manifest, mapping, seed = make_manifest(protocol, "candidate", checkpoint)
        adapters = {name: make_benchmark_adapter(manifest=manifest, method=name, training_seed=seed,
                                                 artifact_map=mapping, config_name=args.config, tf_module=tf)
                    for name in protocol["execution"]["candidate_methods"]}
        description = "proposed receiver (candidate adapter)"
    arrays, preflight, _ = evaluate_method_set(protocol, plan, provider, adapters, tuple(adapters))
    if not all(preflight.values()):
        raise RuntimeError(f"Receiver preflight failed: {preflight}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/"errors.npz", **arrays)
    summary = method_summary([arrays], method, protocol, "development")
    bank = seal_payload({"protocol": protocol["protocol_sha256"], "episode_identity": identity,
                         "fingerprints": arrays["episode_fingerprint"].tolist()}, field="bank_sha256")["bank_sha256"]
    result = {"phase": "development", "stage": args.stage, "config": args.config, "checkpoint": str(checkpoint),
              "update": saved["update"], "method": description, "weights_sha256": file_sha256(weights),
              "bank_sha256": bank, "preflight": preflight, "protocol_sha256": protocol["protocol_sha256"],
              "episodes_per_condition": plan["episodes_per_condition"],
              "summary": {"candidate_persistent": summary}, "raw_sha256": file_sha256(output/"errors.npz")}
    atomic_json(output/"results.json", seal_payload(result, field="result_sha256"))
    print(json.dumps({"update": saved["update"], "core_tbler": summary["core_tbler"],
                      "core_errors": summary["core_errors"], "core_blocks": summary["core_blocks"]}), flush=True)


if __name__ == "__main__":
    main()
