#!/usr/bin/env python3
"""Run calibration, candidate, matched, or static references."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from temporal_benchmark_cache import (
    write_candidate_output,
    write_reference_cache,
    write_static_reference_cache,
)
from temporal_benchmark_candidate import (
    load_adapter,
    load_candidate_manifest,
    resolve_manifest_artifacts,
)
from temporal_benchmark_plan import (
    load_calibration_artifact,
    load_stage_plan,
)
from temporal_benchmark_protocol import load_benchmark_protocol
from temporal_benchmark_runtime import (
    evaluate_calibration_counts,
    evaluate_method_set,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("calibration", "candidate", "reference", "static-reference"),
        required=True,
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", default="nrx_large.cfg")
    parser.add_argument("--artifact-map", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--matched-cold-k3", type=Path)
    parser.add_argument("--matched-cold-k8", type=Path)
    parser.add_argument("--historical-cold-k8", type=Path)
    parser.add_argument("--perfect-csi-kbest", type=Path)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--stage-plan", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "config" / "temporal_benchmark_v1.json",
    )
    return parser


def _require_path(value: Path | None, label: str) -> Path:
    """Require one mode-specific file."""
    if value is None:
        raise ValueError(f"{label} is required")
    return value


def _load_plan(args, protocol):
    """Load the calibration and stage plan together."""
    calibration_path = _require_path(args.calibration, "--calibration")
    plan_path = _require_path(args.stage_plan, "--stage-plan")
    calibration = load_calibration_artifact(calibration_path, protocol)
    return load_stage_plan(plan_path, protocol, calibration)


def _make_adapter(manifest, method, training_seed, artifact_map, context):
    """Construct one adapter from verified registered files."""
    artifacts = resolve_manifest_artifacts(manifest, artifact_map)
    return load_adapter(
        manifest["adapter_factory"],
        manifest=manifest,
        method=method,
        training_seed=training_seed,
        artifacts=artifacts,
        **context,
    )


def _load_matched_baselines(args, protocol):
    """Load training-seed-matched reference manifests."""
    return {
        "matched_cold_k3": load_candidate_manifest(
            _require_path(args.matched_cold_k3, "--matched-cold-k3"), protocol
        ),
        "matched_cold_k8": load_candidate_manifest(
            _require_path(args.matched_cold_k8, "--matched-cold-k8"), protocol
        ),
    }


def _load_static_baselines(args, protocol):
    """Load seed-independent report manifests."""
    return {
        "historical_cold_k8": load_candidate_manifest(
            _require_path(args.historical_cold_k8, "--historical-cold-k8"),
            protocol,
        ),
        "perfect_csi_kbest": load_candidate_manifest(
            _require_path(args.perfect_csi_kbest, "--perfect-csi-kbest"),
            protocol,
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    protocol = load_benchmark_protocol(args.protocol)
    artifact_map = json.loads(args.artifact_map.read_text(encoding="utf-8"))
    if not isinstance(artifact_map, dict):
        raise ValueError("artifact map must be a JSON object")

    import sionna as sn
    import tensorflow as tf
    from temporal_benchmark_scenarios import TemporalBenchmarkScenarioBank
    from utils import E2E_Model, Parameters

    tf.config.experimental.enable_op_determinism()
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.random.set_seed(20_262_100)
    np_seed = 20_262_100
    import numpy as np
    np.random.seed(np_seed)
    sn.config.seed = np_seed
    parameters = Parameters(
        args.config,
        training=False,
        num_tx_eval=int(protocol["radio"]["maximum_active_ues"]),
        system="nrx",
        n_size_bwp_override=int(protocol["radio"]["n_size_bwp"]),
    )
    e2e_model = E2E_Model(
        parameters, training=False, mcs_arr_eval_idx=0
    )
    provider = TemporalBenchmarkScenarioBank(parameters, e2e_model, tf)
    context = {
        "config_name": args.config,
        "parameters": parameters,
        "e2e_model": e2e_model,
        "tf_module": tf,
    }
    destination = args.output_dir.expanduser().resolve()
    if args.mode == "calibration":
        manifest = load_candidate_manifest(
            _require_path(args.matched_cold_k8, "--matched-cold-k8"), protocol
        )
        seed = int(protocol["calibration"]["training_recipe_seed"])
        adapter = _make_adapter(
            manifest, "matched_cold_k8", seed, artifact_map, context
        )
        result = evaluate_calibration_counts(protocol, provider, adapter)
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / "calibration_counts.json"
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")
        return
    plan = _load_plan(args, protocol)
    seed = None if args.training_seed is None else int(args.training_seed)
    if args.mode == "candidate":
        if seed is None:
            raise ValueError("--training-seed is required for candidate mode")
        manifest = load_candidate_manifest(
            _require_path(args.candidate, "--candidate"), protocol
        )
        methods = tuple(protocol["execution"]["candidate_methods"])
        adapters = {
            method: _make_adapter(
                manifest, method, seed, artifact_map, context
            )
            for method in methods
        }
        arrays, preflight, elapsed = evaluate_method_set(
            protocol,
            plan,
            provider,
            adapters,
            methods,
            latency_methods=tuple(
                protocol["execution"]["latency"]["candidate_methods"]
            ),
        )
        output = write_candidate_output(
            destination,
            protocol=protocol,
            plan=plan,
            candidate=manifest,
            training_seed=seed,
            arrays=arrays,
            preflight=preflight,
            elapsed_seconds=elapsed,
        )
        print(f"wrote {output}")
        return
    if args.mode == "static-reference":
        baselines = _load_static_baselines(args, protocol)
        methods = tuple(protocol["execution"]["static_reference_methods"])
        adapters = {
            method: _make_adapter(
                baselines[method], method, None, artifact_map, context
            )
            for method in methods
        }
        arrays, preflight, elapsed = evaluate_method_set(
            protocol,
            plan,
            provider,
            adapters,
            methods,
            oracle_method="perfect_csi_kbest",
        )
        output = write_static_reference_cache(
            destination,
            protocol=protocol,
            plan=plan,
            baselines=baselines,
            arrays=arrays,
            preflight=preflight,
            elapsed_seconds=elapsed,
        )
        print(f"wrote {output}")
        return
    baselines = _load_matched_baselines(args, protocol)
    if seed is None:
        raise ValueError("--training-seed is required for reference mode")
    methods = tuple(protocol["execution"]["accuracy_reference_methods"])
    latency_methods = tuple(
        protocol["execution"]["latency"]["reference_methods"]
    )
    adapter_methods = tuple(dict.fromkeys(methods + latency_methods))
    adapters = {
        method: _make_adapter(
            baselines[method], method, seed, artifact_map, context
        )
        for method in adapter_methods
    }
    arrays, preflight, elapsed = evaluate_method_set(
        protocol,
        plan,
        provider,
        adapters,
        methods,
        latency_methods=latency_methods,
    )
    output = write_reference_cache(
        destination,
        protocol=protocol,
        plan=plan,
        baselines=baselines,
        training_seed=seed,
        arrays=arrays,
        preflight=preflight,
        elapsed_seconds=elapsed,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
