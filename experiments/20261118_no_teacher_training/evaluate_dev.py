"""Development TBLER of one checkpoint of the proposed receiver, scored like the baseline's checkpoints.

Follows the development evaluation of the retrained two-iteration baseline (same protocol, anchors, seeds, cached
episodes, and core-condition scoring), with the benchmark's adapter of the proposed receiver loading the checkpoint.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
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
    from neural_rx.backbone import E2E_Model, Parameters
    from neural_rx.benchmark.candidate import pretrained_checkpoint
    from neural_rx.benchmark.episodes import CachedScenarioBank, scenario_identity
    from neural_rx.benchmark.frozen import (load_ray_traced_banks, make_benchmark_adapter, make_manifest,
                                            protocol_and_plan)
    from neural_rx.benchmark.generalization_protocol import seal_payload, verify_payload
    from neural_rx.benchmark.runtime import evaluate_method_set
    from neural_rx.benchmark.scenarios import TemporalBenchmarkScenarioBank
    from neural_rx.benchmark.scoring import method_summary
    from neural_rx.paths import ROOT
    from neural_rx.provenance.artifacts import atomic_json, file_sha256

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(20260930)
    sn.config.seed = 20260930

    checkpoint = args.checkpoint.resolve()
    saved = json.loads((checkpoint/"manifest.json").read_text())
    weights = checkpoint/"receiver.weights.h5"
    if file_sha256(weights) != saved["files"]["receiver.weights.h5"]:
        raise ValueError("Receiver weights differ from the checkpoint manifest")
    if saved["binding"]["recipe"].get("receiver_variant", "pilot_memory") != "pilot_memory":
        raise ValueError("Not a checkpoint of the proposed receiver")

    artifact = json.loads(args.anchors.read_text())
    verify_payload(artifact, field="manifest_sha256")
    if artifact["teacher_sha256"] != file_sha256(pretrained_checkpoint()) or artifact["prbs"] != 132:
        raise ValueError("Calibration reference differs")
    protocol, plan = protocol_and_plan("development", 132, args.episodes, artifact["conditions"])
    # As in the baseline's development evaluation, so that the cached episodes have the same identity.
    protocol["methods"] = protocol["methods"]+["frozen_cold_k2"]
    protocol["execution"]["accuracy_reference_methods"] = ["frozen_cold_k2"]
    rt_banks = load_ray_traced_banks(protocol, plan)
    if rt_banks:
        raise ValueError("The development conditions are not expected to include ray-traced channels")
    identity = scenario_identity(ROOT/"config/nrx_large.cfg", protocol)
    parameters = Parameters("nrx_large.cfg", training=False, num_tx_eval=2, n_size_bwp_override=132)
    e2e = E2E_Model(parameters, training=False, mcs_arr_eval_idx=0)
    provider = CachedScenarioBank(TemporalBenchmarkScenarioBank(parameters, e2e, tf, rt_banks=rt_banks),
                                  args.episode_cache, protocol, identity)
    identity = provider.identity
    if artifact["source_identity"]["radio"] != identity["radio"]:
        raise ValueError("Calibration radio configuration differs")
    protocol["calibration"]["source_manifest_sha256"] = artifact["manifest_sha256"]
    protocol = seal_payload(protocol, field="protocol_sha256")
    plan["protocol_sha256"] = protocol["protocol_sha256"]
    provider.ensure_capacity(plan["jobs"])

    methods = tuple(protocol["execution"]["candidate_methods"])
    manifest, mapping, seed = make_manifest(protocol, "candidate", checkpoint)
    adapters = {name: make_benchmark_adapter(
        manifest=manifest, method=name, training_seed=seed, artifact_map=mapping, config_name="nrx_large.cfg",
        tf_module=tf) for name in methods}
    arrays, preflight, _ = evaluate_method_set(protocol, plan, provider, adapters, methods)
    if not all(preflight.values()):
        raise RuntimeError(f"Receiver preflight failed: {preflight}")
    method = "candidate_persistent"

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/"errors.npz", **arrays)
    summaries = {name: method_summary([arrays], name, protocol, "development") for name in methods}
    summary = summaries[method]
    bank = seal_payload({"protocol": protocol["protocol_sha256"], "episode_identity": identity,
                         "fingerprints": arrays["episode_fingerprint"].tolist()}, field="bank_sha256")["bank_sha256"]
    result = {"phase": "development", "checkpoint": str(checkpoint), "update": saved["update"],
              "method": "proposed receiver (candidate adapter)",
              "weights_sha256": file_sha256(weights), "bank_sha256": bank, "preflight": preflight,
              "protocol_sha256": protocol["protocol_sha256"], "episodes_per_condition": plan["episodes_per_condition"],
              "summary": summaries, "raw_sha256": file_sha256(output/"errors.npz")}
    atomic_json(output/"results.json", seal_payload(result, field="result_sha256"))
    print(json.dumps({"update": saved["update"], "core_tbler": summary["core_tbler"],
                      "core_errors": summary["core_errors"], "core_blocks": summary["core_blocks"],
                      "all": {name: value["core_tbler"] for name, value in summaries.items()}}), flush=True)


if __name__ == "__main__":
    main()
