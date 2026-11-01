from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from temporal_benchmark_candidate import (artifact_record,
                                         pretrained_checkpoint, resolve_manifest_artifacts, validate_manifest)
from temporal_benchmark_protocol import load_benchmark_protocol, resolved_condition_settings
from temporal_benchmark_io import DECODED_LATENCY_DEFINITION
from temporal_benchmark_scoring import crc_diagnostics, pooled_tbler, TBLER_DEFINITION
from temporal_generalization_protocol import seal_payload, verify_payload
from artifact_provenance import evaluation_source_records, validate_evaluation_sources
from artifact_io import atomic_json, file_sha256, validate_checkpoint_files

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ("frozen_cold_k8", "frozen_cold_k2")



def protocol_and_plan(phase, width, episodes=None, anchors=None, rt_bank=None):
    if width != 132 and phase != "smoke":
        raise ValueError("Full experiments require the declared 132-PRB deployment")
    protocol = copy.deepcopy(load_benchmark_protocol(ROOT/"config/temporal_benchmark_v1.json"))
    protocol["protocol"] = "pretrained_nrx_experiment_v1"
    protocol["radio"]["n_size_bwp"] = width
    phase_settings = protocol["evaluation_phases"][phase]
    if rt_bank is not None:
        if phase in ("calibration", "development", "confirmation") and rt_bank.config["split"] != phase:
            raise ValueError("The RT channel split does not match the evaluation phase")
        name = rt_bank.config["name"]
        protocol["conditions"] = {name: dict(provider="ray_traced", receiver_inputs=[],
            channel_bank_sha256=rt_bank.sha256, ue_pool_size=len(rt_bank.config["users"]))}
        for key in ("condition_order", "core_conditions", "stationary_conditions", "development_conditions"):
            protocol[key] = [name]
        protocol["report_conditions"], protocol["stress_conditions"] = [], []
        protocol["execution"]["latency"]["condition"] = name
        if len(rt_bank.config["users"]) < protocol["radio"]["maximum_active_ues"]:
            raise ValueError("The channel bank has fewer UEs than the declared active receiver configuration")
        protocol["rt_channel_bank"] = rt_bank.manifest
    if phase in ("smoke", "latency"):
        conditions = [protocol["execution"]["latency"]["condition"]]
    elif phase == "development":
        conditions = list(protocol["development_conditions"])
        protocol["stationary_conditions"] = [name for name in protocol["stationary_conditions"] if name in conditions]
    else:
        conditions = list(protocol["condition_order"])
    seed_order = list(protocol["condition_order"])
    tbs = phase_settings["num_tbs"]
    count = episodes if episodes is not None else phase_settings["episodes"]
    if count < 1:
        raise ValueError("Episode count must be positive")
    if rt_bank is not None and count > len(rt_bank.records):
        raise ValueError("Export more RT episodes or request fewer evaluation episodes")
    refs = ["frozen_cold_k8"]
    candidates = ["candidate_persistent"] if phase == "development" else ["candidate_persistent", "candidate_reset"]
    execution = protocol["execution"]
    execution.update(candidate_methods=candidates, accuracy_reference_methods=refs,
                     latency_only_reference_methods=["frozen_cold_k3"],
                     persistent_methods=["candidate_persistent"],
                     accuracy_episode_batch_size=1 if phase in ("smoke", "latency") else execution["accuracy_episode_batch_size"])
    protocol["methods"] = candidates + refs + ["frozen_cold_k3"]
    protocol["training"] = {"reference_policy": "frozen_no_additional_training", "selection_data": "development_only"}
    first = phase_settings["scoring_start_tb"]
    protocol["stages"] = {"development": dict(num_tbs=tbs, episodes_per_condition=count,
        condition_names=conditions, scoring_window=[first,tbs],
        snr_points=[f"point_{i}" for i in range(len(phase_settings["snr_db"]))] if phase == "calibration" else ["anchor"])}
    protocol["calibration"] = dict(baseline="frozen_cold_k8", additional_training_updates=0, target_tbler=.1)
    protocol["experiment"] = {"phase": phase, "scoring_start_tb": first,
                               "latency_definition": DECODED_LATENCY_DEFINITION}
    protocol["runtime_sources"] = evaluation_source_records()
    protocol = seal_payload(protocol, field="protocol_sha256")
    jobs = []
    base = phase_settings["seed"]
    for name in conditions:
        i = seed_order.index(name)
        settings = resolved_condition_settings(protocol, name, tbs)
        if phase == "calibration":
            snrs = list(phase_settings["snr_db"])
        elif phase in ("development", "confirmation"):
            if anchors is None or name not in anchors:
                raise ValueError("Accuracy comparison requires frozen K8 calibration anchors")
            snrs = [float(anchors[name]["anchor_snr_db"])]
        else:
            snrs = list(phase_settings["snr_db"])
        jobs.append({"condition": name, "provider": settings["provider"], "provider_settings": settings,
                     "episode_count": count, "num_tbs": tbs, "snr_db": snrs,
                     "seed_streams": {k: base+i*1000+offset for k,offset in protocol["seed_domains"].items()}})
    names = [f"point_{i}" for i in range(len(jobs[0]["snr_db"]))] if phase == "calibration" else ["anchor"]
    return protocol, {"stage": "development", "jobs": jobs, "episodes_per_condition": count,
                      "snr_point_names": names, "protocol_sha256": protocol["protocol_sha256"]}


def load_receiver_checkpoint_export(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve()
    directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
    weights = directory / "receiver.weights.h5"
    if checkpoint != directory and checkpoint != weights:
        raise ValueError("Expected a checkpoint export directory or its receiver.weights.h5")
    saved = json.loads((directory / "manifest.json").read_text())
    required = {"receiver.weights.h5", "cnn_autoencoder_finalist.json"}
    if not required.issubset(saved["files"]):
        raise ValueError("Checkpoint export must bind weights and receiver configuration")
    validate_checkpoint_files(directory, saved["files"])
    settings = json.loads((directory / "cnn_autoencoder_finalist.json").read_text())
    architecture = settings["architecture"]
    if architecture != saved["binding"].get("architecture"):
        raise ValueError("Checkpoint architecture differs from its training binding")
    if "training_updates" in settings and settings["training_updates"] != saved["update"]:
        raise ValueError("Checkpoint training status differs from its saved update")
    return directory, saved


def make_manifest(protocol, role, checkpoint=None):
    support = [ROOT/"config/nrx_large.cfg"]
    if role == "candidate":
        if checkpoint is None:
            raise ValueError("Candidate checkpoint directory is required")
        directory, saved = load_receiver_checkpoint_export(checkpoint)
        weights = directory/"receiver.weights.h5"
        support += [directory/"cnn_autoencoder_finalist.json", directory/"manifest.json"]
        seed = saved["binding"]["seed"]
        training = {"additional_updates": saved["update"], "binding": saved["binding"]}
    else:
        weights = pretrained_checkpoint()
        seed, training = 0, {"additional_updates": 0, "provenance": "shipped_neural_rx"}
    paths = [weights] + support
    records = [artifact_record(p) for p in paths]
    inference = {"iterations": 2 if role == "candidate" else int(role[-1]),
                 "persistent_memory": role == "candidate"}
    manifest = seal_payload({"manifest": "temporal_benchmark_frozen_v2", "schema_version": 2,
        "role": role, "radio": protocol["radio"], "training": training, "inference": inference,
        "benchmark_protocol_sha256": protocol["protocol_sha256"],
        "checkpoints": [dict(records[0], training_seed=seed)], "support_artifacts": records[1:]}, field="manifest_sha256")
    validate_manifest(manifest, protocol)
    return manifest, {r["sha256"]: str(p) for r,p in zip(records,paths)}, seed


def verified_result(directory):
    directory = Path(directory)
    result = json.loads((directory/"results.json").read_text())
    verify_payload(result, field="result_sha256")
    protocol = json.loads((directory/"protocol.json").read_text())
    verify_payload(protocol, field="protocol_sha256")
    if result["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError("Result protocol differs")
    validate_evaluation_sources(protocol["runtime_sources"])
    raw = "samples.npz" if result["phase"] == "latency" else "errors.npz"
    if file_sha256(directory/raw) != result["raw_sha256"]:
        raise ValueError("Result samples changed")
    for name, expected in result.get("calibration_extension_sha256", {}).items():
        if file_sha256(directory/name) != expected:
            raise ValueError("Calibration extension samples changed")
    for role, manifest in result["manifests"].items():
        validate_manifest(manifest, protocol)
        current, _, _ = make_manifest(protocol, role, result.get("checkpoint"))
        if manifest != current:
            raise ValueError("Evaluated receiver artifacts or settings changed; rerun the evaluation")
    return result


def make_benchmark_adapter(*, manifest, artifact_map, **kwargs):
    from temporal_benchmark_adapters import ColdNeuralRxBenchmarkAdapter

    if manifest["role"] == "candidate":
        from pilot_history_receiver.benchmark_adapter import PilotHistoryBenchmarkAdapter

        adapter = PilotHistoryBenchmarkAdapter
    else:
        adapter = ColdNeuralRxBenchmarkAdapter
    return adapter(manifest=manifest,
                   artifacts=resolve_manifest_artifacts(manifest, artifact_map), **kwargs)


def rates(arrays, methods, first):
    return {method: pooled_tbler(arrays[f"errors_{method}"], arrays["active"], slice(first-1, None)).tolist()
            for method in methods}



def development_report(arrays, methods, protocol):
    from temporal_benchmark_scoring import paired_summary, method_summary, late_drift_summary, adaptation_summary
    runs = [arrays]
    rng = np.random.RandomState(protocol["statistics"]["bootstrap_seed"])
    draws = int(protocol["statistics"]["bootstrap_samples"])
    summary = {m:method_summary(runs,m,protocol,"development") for m in methods}
    comparisons = {m:paired_summary(runs,"candidate_persistent",m,protocol,"development",draws,rng)
                   for m in methods if m != "candidate_persistent"}
    windows = {}
    bounds = dict(protocol["windows"],tb1=[1,1],whole_episode=[1,protocol["stages"]["development"]["num_tbs"]])
    for name,(start,end) in bounds.items():
        windows[name] = {m:pooled_tbler(arrays[f"errors_{m}"],arrays["active"],slice(start-1,end)).tolist()
                         for m in methods}
    beats_k8 = comparisons["frozen_cold_k8"]["ci95"][1] < 0
    memory_benefit = bool(comparisons["candidate_reset"]["ci95"][1] < 0) if "candidate_reset" in comparisons else None
    return dict(tbler_definition=TBLER_DEFINITION, summary=summary, comparisons=comparisons,
        windows=windows, late_drift=late_drift_summary(runs,protocol,"development",draws,rng,
            reference="candidate_reset" if "candidate_reset" in comparisons else "frozen_cold_k8"),
        adaptation=adaptation_summary(runs,protocol,"development",reference="frozen_cold_k8"),
        cold_k8_accuracy_target_met=bool(beats_k8), memory_benefit_demonstrated=memory_benefit,
        accuracy_criteria_met=bool(beats_k8 and memory_benefit) if memory_benefit is not None else None,
        eligibility_scope=f"{protocol['experiment']['phase']} evidence at one training seed; final accuracy requires improvement over frozen Cold K8 and the same weights reset; trained-weight latency remains required")


def calibrate_reference(arrays, protocol, plan, provider, output, reference_cache, tf):
    from temporal_benchmark_reference import evaluate_frozen_references
    from temporal_benchmark_plan import calibrate_condition

    first = protocol["experiment"]["scoring_start_tb"] - 1
    lower, upper = protocol["evaluation_phases"]["calibration"]["snr_bounds_db"]
    conditions, artifacts = {}, {}

    def points_for(values, job, index):
        active = values["active"][index, :, first:]
        return [{"snr_db": snr,
                 "errors": int((values["errors_frozen_cold_k8"][index, si, :, first:] & active).sum()),
                 "blocks": int(active.sum())} for si, snr in enumerate(job["snr_db"])]

    for ci, job in enumerate(plan["jobs"]):
        points = points_for(arrays, job, ci)
        target = protocol["calibration"]["target_tbler"]
        fitted = calibrate_condition(points, target)
        while not fitted["target_bracketed"]:
            grid = sorted(point["snr_db"] for point in points)
            spacing = float(np.min(np.diff(grid)))
            span = grid[-1] - grid[0]
            if fitted["isotonic_tbler"][-1] > target:
                extra = np.arange(grid[-1] + spacing, min(upper, grid[-1] + span) + spacing/2, spacing)
                extra = extra[extra <= upper]
            else:
                extra = np.arange(grid[0] - spacing, max(lower, grid[0] - span) - spacing/2, -spacing)[::-1]
                extra = extra[extra >= lower]
            if not len(extra):
                break
            name = job["condition"]
            print(json.dumps(dict(phase="calibration_extension", condition=name, snr_db=extra.tolist())), flush=True)
            extension_protocol = copy.deepcopy(protocol)
            stage = extension_protocol["stages"]["development"]
            stage.update(condition_names=[name], snr_points=[f"point_{i}" for i in range(len(extra))])
            extension_protocol["core_conditions"] = [name]
            extension_protocol["stationary_conditions"] = []
            extension_protocol = seal_payload(extension_protocol, field="protocol_sha256")
            extension_plan = dict(plan, jobs=[dict(job, snr_db=extra.tolist())],
                                  snr_point_names=stage["snr_points"],
                                  protocol_sha256=extension_protocol["protocol_sha256"])
            provider.ensure_capacity(extension_plan["jobs"])
            registrations = {"frozen_cold_k8": make_manifest(extension_protocol, "frozen_cold_k8")}
            values, _, _ = evaluate_frozen_references(reference_cache, extension_protocol, extension_plan,
                                                     provider, registrations, tf)
            directory = output/"extensions"/f"{name}_{len(points):03d}"
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory/"errors.npz", **values)
            atomic_json(directory/"protocol.json", extension_protocol)
            atomic_json(directory/"plan.json", extension_plan)
            for filename in ("errors.npz", "protocol.json", "plan.json"):
                path = directory/filename
                artifacts[str(path.relative_to(output))] = file_sha256(path)
            points.extend(points_for(values, extension_plan["jobs"][0], 0))
            fitted = calibrate_condition(points, target)
        conditions[job["condition"]] = fitted
    return conditions, artifacts


def run_experiment(args):
    import tensorflow as tf
    import sionna as sn
    from temporal_benchmark_episodes import CachedScenarioBank, scenario_identity
    from temporal_benchmark_scenarios import TemporalBenchmarkScenarioBank
    from temporal_benchmark_runtime import evaluate_method_set, _measure_latency
    from temporal_benchmark_reference import evaluate_frozen_references
    from utils import E2E_Model, Parameters
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(20260930)
    sn.config.seed = 20260930
    phase, width = args.experiment_phase, args.experiment_prbs
    rt_bank = None
    if getattr(args, "rt_channel_bank", None) is not None:
        from rt_channel_bank import RTChannelBank
        rt_bank = RTChannelBank(args.rt_channel_bank)
    anchors = None
    if phase in ("development", "confirmation"):
        if args.experiment_anchors is None:
            raise ValueError("--experiment-anchors is required")
        artifact = json.loads(args.experiment_anchors.read_text())
        verify_payload(artifact, field="manifest_sha256")
        if artifact["teacher_sha256"] != file_sha256(pretrained_checkpoint()) or artifact["prbs"] != width:
            raise ValueError("Calibration reference differs")
        anchors = artifact["conditions"]
    protocol, plan = protocol_and_plan(phase, width, args.experiment_episodes, anchors,
                                     rt_bank=rt_bank)
    if anchors is not None:
        unbracketed = [job["condition"] for job in plan["jobs"]
                       if not anchors[job["condition"]].get("target_bracketed", False)]
        if unbracketed:
            raise ValueError(f"Calibration did not bracket 10% TBLER for {phase}: {', '.join(unbracketed)}")
    identity = scenario_identity(ROOT/"config/nrx_large.cfg", protocol)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    p = Parameters("nrx_large.cfg", training=False, num_tx_eval=2, n_size_bwp_override=width)
    e2e = E2E_Model(p, training=False, mcs_arr_eval_idx=0)
    provider = CachedScenarioBank(TemporalBenchmarkScenarioBank(p, e2e, tf, rt_bank=rt_bank), args.episode_cache, protocol,
                                  identity, readonly=args.episode_cache_readonly)
    identity = provider.identity
    if phase in ("development", "confirmation"):
        if artifact["source_identity"]["radio"] != identity["radio"]:
            raise ValueError("Calibration radio configuration differs")
        if artifact["source_identity"].get("rt_scenario_sha256") != identity.get("rt_scenario_sha256"):
            raise ValueError("Calibration ray-traced scenario differs")
        if rt_bank is not None and artifact["source_identity"]["rt_channel_seed"] == identity["rt_channel_seed"]:
            raise ValueError("Calibration and accuracy require independently seeded RT channel banks")
        protocol["calibration"]["source_manifest_sha256"] = artifact["manifest_sha256"]
        protocol = seal_payload(protocol, field="protocol_sha256")
        plan["protocol_sha256"] = protocol["protocol_sha256"]
    provider.ensure_capacity(plan["jobs"])
    manifests = {}
    def make(method):
        role = "candidate" if method.startswith("candidate_") else method
        manifest, mapping, seed = make_manifest(protocol, role, args.experiment_checkpoint)
        manifests[role] = manifest
        return make_benchmark_adapter(manifest=manifest, method=method, training_seed=seed,
                                      artifact_map=mapping, config_name="nrx_large.cfg", tf_module=tf)
    atomic_json(output/"protocol.json", protocol)
    atomic_json(output/"plan.json", plan)
    if phase == "latency":
        methods = ("candidate_persistent", "frozen_cold_k3")
        adapters = {m:make(m) for m in methods}
        timings = _measure_latency(protocol, plan, provider, adapters, methods)
        if not all(adapter.checksums_unchanged() for adapter in adapters.values()):
            raise RuntimeError("Latency receiver artifact changed")
        results = {m: {"p50_ms": float(np.median(timings[f"latency_ms_{m}"])),
                       "p95_ms": float(np.percentile(timings[f"latency_ms_{m}"],95)),
                       "crc_diagnostics": crc_diagnostics(timings[f"latency_errors_{m}"],
                                                          timings[f"latency_crc_disagreement_{m}"])} for m in methods}
        passed = results["candidate_persistent"]["p95_ms"] < results["frozen_cold_k3"]["p95_ms"]
        np.savez_compressed(output/"samples.npz", **timings)
        atomic_json(output/"results.json", seal_payload({"phase": phase, "gate_passed": passed,
                    "checkpoint": str(args.experiment_checkpoint.resolve()) if args.experiment_checkpoint else None,
                    "receiver": "pilot-history-memory",
                    "receivers": results, "latency_definition": DECODED_LATENCY_DEFINITION, "manifests": manifests,
                    "protocol_sha256": protocol["protocol_sha256"], "episode_identity": identity,
                    "raw_sha256": file_sha256(output/"samples.npz")}, field="result_sha256"))
        print(json.dumps(results), flush=True)
        return
    all_arrays, checks = {}, {}
    groups = [] if phase == "calibration" else [tuple(protocol["execution"]["candidate_methods"])]
    groups.append(tuple(protocol["execution"]["accuracy_reference_methods"]))
    for methods in groups:
        if all(m in FROZEN for m in methods):
            registrations = {m: make_manifest(protocol, m) for m in methods}
            manifests.update({m: record[0] for m, record in registrations.items()})
            arrays, preflight, _ = evaluate_frozen_references(
                args.reference_cache, protocol, plan, provider, registrations, tf)
        else:
            adapters = {m:make(m) for m in methods}
            arrays, preflight, _ = evaluate_method_set(protocol, plan, provider, adapters, methods)
            del adapters
        if not all(preflight.values()):
            raise RuntimeError(f"Receiver preflight failed: {preflight}")
        if all_arrays and not np.array_equal(all_arrays["episode_fingerprint"], arrays["episode_fingerprint"]):
            raise RuntimeError("Receiver groups did not use identical episodes")
        all_arrays.update(arrays)
        checks.update(preflight)
    np.savez_compressed(output/"errors.npz", **all_arrays)
    scored = rates(all_arrays, tuple(m for group in groups for m in group), protocol["experiment"]["scoring_start_tb"])
    result = {"phase": phase, "rates": scored, "tbler_definition": TBLER_DEFINITION, "preflight": checks, "manifests": manifests,
              "protocol_sha256": protocol["protocol_sha256"], "episode_identity": identity,
              "checkpoint": str(args.experiment_checkpoint.resolve()) if args.experiment_checkpoint else None,
              "receiver": "pilot-history-memory", "scoring_start_tb": protocol["experiment"]["scoring_start_tb"]}
    first = result["scoring_start_tb"] - 1
    result["crc_diagnostics"] = {
        m: crc_diagnostics(all_arrays[f"errors_{m}"][:, :, :, first:],
                           all_arrays[f"crc_disagreement_{m}"][:, :, :, first:])
        for m in scored
    }
    if phase == "calibration":
        conditions, extension_hashes = calibrate_reference(all_arrays, protocol, plan, provider, output,
                                                          args.reference_cache, tf)
        result["calibration_extension_sha256"] = extension_hashes
        result["unbracketed_conditions"] = [name for name, values in conditions.items()
                                             if not values["target_bracketed"]]
        artifact = seal_payload({"conditions": conditions, "tbler_definition": TBLER_DEFINITION, "prbs": width, "source_identity": identity,
                                 "teacher_sha256": manifests["frozen_cold_k8"]["checkpoints"][0]["sha256"],
                                 "runtime_sources": protocol["runtime_sources"]}, field="manifest_sha256")
        atomic_json(output/"anchors.json", artifact)
    result["bank_sha256"] = seal_payload({"protocol": protocol["protocol_sha256"],
        "episode_identity": identity, "fingerprints": all_arrays["episode_fingerprint"].tolist()},
        field="bank_sha256")["bank_sha256"]
    if phase in ("development", "confirmation"):
        result.update(development_report(all_arrays,tuple(scored),protocol))
    result["raw_sha256"] = file_sha256(output/"errors.npz")
    atomic_json(output/"results.json", seal_payload(result,field="result_sha256"))
    print(json.dumps({"rates": scored, "preflight": checks}), flush=True)
