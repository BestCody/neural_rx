from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from pilot_history_receiver.training_options import add_execution_arguments, add_preparation_arguments, requested_updates, resolve_preparation
from pilot_history_receiver.checkpoints import recover_latest
from artifact_io import atomic_json, file_sha256
from artifact_provenance import normalize_training_recipe
from temporal_benchmark_frozen import verified_result
from temporal_benchmark_protocol import load_benchmark_protocol

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def selection_key(result):
    return (result["summary"]["candidate_persistent"]["core_tbler"],
            result["manifests"]["candidate"]["training"]["additional_updates"])


def evaluation_updates(recipe, stop, previous=()):
    interval = recipe["development_interval"]
    if type(interval) is not int or interval < 1:
        raise ValueError("Development interval must be a positive integer")
    if any(type(point) is not int or point < 0 for point in previous):
        raise ValueError("Evaluation updates must be nonnegative integers")
    return sorted({p for p in (stop, *range(interval, stop+1, interval), *previous) if 0 < p <= stop})


def continued_registration(previous, current):
    mutable = {"requested_updates", "checkpoints", "execution"}
    old_contract = {k:v for k,v in previous.items() if k not in mutable}
    new_contract = {k:v for k,v in current.items() if k not in mutable}
    old_contract["recipe"] = normalize_training_recipe(old_contract["recipe"])
    new_contract["recipe"] = normalize_training_recipe(new_contract["recipe"])
    if old_contract != new_contract:
        raise ValueError("Existing training run has a different registration")
    return dict(current, checkpoints=evaluation_updates(current["recipe"], current["requested_updates"],
                                                        previous["checkpoints"]))


def parser():
    p = argparse.ArgumentParser(description="Jointly train K2 and channel memory with development-TBLER scheduling")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Run artifacts under experiments/YYYYMMDD_descriptive_name/results/")
    p.add_argument("--model-dir", type=Path,
                   help="Checkpoint directory. Defaults to models/<output directory name>")
    p.add_argument("--recipe", type=Path, default=ROOT/"config/memory_training.json")
    p.add_argument("--episode-cache", type=Path, default=ROOT/"datasets/evaluation")
    p.add_argument("--development-cache", type=Path,
                   help="Read development episodes from a retained read-only bank; other phases use --episode-cache")
    p.add_argument("--anchors", type=Path, help="Reuse retained Cold K8 calibration anchors without recalibrating")
    p.add_argument("--reference-cache", type=Path, help="Reuse frozen-reference results from this directory")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--evaluation-gpu", type=int,
                   help="Defaults to the preparation or replica GPU, whose workers stop during evaluation")
    p.add_argument("--steps", type=int, required=True,
                   help="Total updates, including restored updates; no training-length ceiling")
    add_execution_arguments(p)
    add_preparation_arguments(p)
    p.add_argument("--dry-run", action="store_true")
    return p


def main():
    args = parser().parse_args()
    recipe = normalize_training_recipe(read(args.recipe))
    total_updates = requested_updates(args.steps)
    resolve_preparation(args, recipe["effective_examples"])
    if args.evaluation_gpu is None:
        args.evaluation_gpu = next(gpu for gpu in (args.data_gpu, args.replica_gpu, args.gpu) if gpu is not None)
    output = args.output_dir.expanduser().resolve()
    models = (args.model_dir or ROOT/"models"/output.name).expanduser().resolve()
    benchmark = load_benchmark_protocol(ROOT/"config/temporal_benchmark_v1.json")
    from temporal_generalization_protocol import verify_payload
    development_cache = None
    if args.development_cache is not None:
        dataset = read(args.development_cache/"dataset.json")
        verify_payload(dataset, field="manifest_sha256")
        if dataset["identity"]["radio"] != benchmark["radio"]:
            raise ValueError("Retained development radio configuration differs")
        development_cache = dict(path=str(args.development_cache.resolve()), manifest_sha256=dataset["manifest_sha256"])
    retained_anchors = None
    if args.anchors is not None:
        from temporal_benchmark_candidate import pretrained_checkpoint
        anchors = read(args.anchors)
        verify_payload(anchors, field="manifest_sha256")
        if anchors["teacher_sha256"] != file_sha256(pretrained_checkpoint()) or anchors["prbs"] != benchmark["radio"]["n_size_bwp"]:
            raise ValueError("Retained calibration reference differs")
        retained_anchors = dict(path=str(args.anchors.resolve()), sha256=file_sha256(args.anchors))
    registration = dict(protocol="channel_memory_joint_training_run_v1", recipe=recipe,
        architecture=read(ROOT/"config/memory_receiver.json")["architecture"],
        model_directory=str(models), benchmark_protocol_sha256=benchmark["protocol_sha256"],
        calibration_anchors=retained_anchors,
        development_cache=development_cache,
        execution=dict(detector_batch_tbs=args.detector_batch_tbs, descriptor_chunk_tbs=args.descriptor_chunk_tbs,
                       render_batch_tbs=args.render_batch_tbs,
                       replica_gpu=args.replica_gpu, replica_memory_mib=args.replica_memory_mib,
                       preparation_gpu_memory_mib=args.preparation_gpu_memory_mib,
                       data_gpu=args.data_gpu, prefetch_examples=args.prefetch_examples,
                       evaluation_gpu=args.evaluation_gpu,
                       channel_workers=args.channel_workers, prefetch_memory_mib=args.prefetch_memory_mib),
        requested_updates=total_updates, checkpoints=evaluation_updates(recipe, total_updates),
        reference_policy="frozen_k8_accuracy; frozen_k3_latency; reset_ablation_at_confirmation_only",
        evaluation_policy="one learner; development TBLER controls learning rate and selection, never duration; automatic independent confirmation",
        selection="lowest pooled persistent TBLER over the declared primary conditions on the fixed development bank; retain earlier checkpoint on ties")
    if (output/"registration.json").exists():
        previous = read(output/"registration.json")
        registration = continued_registration(previous, registration)
    if args.dry_run:
        print(json.dumps(registration, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output/"pilot.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/"registration.json").exists():
            previous = read(output/"registration.json")
            registration = continued_registration(previous, registration)
        latest = recover_latest(models)
        restored = 0 if latest is None else latest["update"]
        if restored > total_updates:
            raise ValueError("Requested stop precedes a completed training update")
        atomic_json(output/"registration.json", registration)

        def run(arguments, log):
            command = [sys.executable]+[str(x) for x in arguments]
            print(json.dumps(dict(command=command, log=str(log))), flush=True)
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", buffering=1) as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)

        anchors = args.anchors.resolve() if args.anchors is not None else output/"calibration"/"anchors.json"
        reference_cache = args.reference_cache.resolve() if args.reference_cache is not None else output/"reference_cache"

        def evaluate(phase, checkpoint, directory):
            if retained_anchors is not None and file_sha256(anchors) != retained_anchors["sha256"]:
                raise ValueError("Retained calibration anchors changed during training")
            if phase == "development" and development_cache is not None:
                dataset = read(args.development_cache/"dataset.json")
                verify_payload(dataset, field="manifest_sha256")
                if dataset["manifest_sha256"] != development_cache["manifest_sha256"]:
                    raise ValueError("Retained development bank changed during training")
            if not (directory/"results.json").exists():
                command = [ROOT/"scripts/run_temporal_benchmark.py",
                    "--experiment-phase", phase, "--experiment-checkpoint", checkpoint,
                    "--experiment-prbs", benchmark["radio"]["n_size_bwp"],
                    "--gpu", args.evaluation_gpu, "--episode-cache",
                    (args.development_cache if phase == "development" and args.development_cache is not None else args.episode_cache).resolve(),
                    "--reference-cache", reference_cache, "--output-dir", directory]
                if phase == "development" and args.development_cache is not None:
                    command += ["--episode-cache-readonly"]
                if phase in ("development", "confirmation"):
                    command += ["--experiment-anchors", anchors]
                run(command, directory/"run.log")
            result = verified_result(directory)
            if result["phase"] != phase or Path(result["checkpoint"]).resolve() != checkpoint.resolve():
                raise ValueError("Saved evaluation belongs to a different phase or checkpoint")
            return result

        try:
            candidates = {}

            def development(checkpoint):
                step = int(read(checkpoint/"manifest.json")["update"])
                atomic_json(output/"status.json", dict(status="evaluating", completed_updates=step,
                    requested_updates=total_updates, checkpoint=str(checkpoint)))
                if retained_anchors is None:
                    evaluate("calibration", models/"checkpoint_000000", output/"calibration")
                directory = output/f"development_{step:06d}"
                result = evaluate("development", checkpoint, directory)
                if any(previous["bank_sha256"] != result["bank_sha256"] for previous, _ in candidates.values()):
                    raise ValueError("Development checkpoint comparisons must use the same bank")
                candidates[step] = (result, directory)
                best, best_directory = min(candidates.values(), key=lambda item: selection_key(item[0]))
                atomic_json(output/"best.json", dict(checkpoint=best["checkpoint"],
                    development_result=str(best_directory/"results.json"),
                    core_tbler=best["summary"]["candidate_persistent"]["core_tbler"]))
                atomic_json(output/"status.json", dict(status="training", requested_updates=total_updates,
                    completed_updates=max(restored, step), evaluated_checkpoints=len(candidates)))
                return result

            for step in registration["checkpoints"]:
                if step < restored:
                    development(models/f"checkpoint_{step:06d}")
            from train_memory_receiver import run_training
            training_args = argparse.Namespace(**vars(args))
            training_args.output_dir = models
            training_args.seed = None
            training_args.resume = None if latest is None else models/latest["checkpoint"]
            atomic_json(output/"status.json", dict(status="training", completed_updates=restored,
                                                   requested_updates=total_updates))
            run_training(training_args, evaluation_points=registration["checkpoints"], evaluate=development)
            if total_updates == 0:
                atomic_json(output/"status.json", dict(status="initialized", completed_updates=0,
                                                       requested_updates=0))
                return
            result, directory = min(candidates.values(), key=lambda item: selection_key(item[0]))
            checkpoint = Path(result["checkpoint"]).resolve()
            atomic_json(output/"status.json", dict(status="confirming", completed_updates=total_updates,
                requested_updates=total_updates, best_checkpoint=str(checkpoint)))
            confirmation_dir = output/f"confirmation_{checkpoint.name.removeprefix('checkpoint_')}"
            confirmation = evaluate("confirmation", checkpoint, confirmation_dir)
            latency_dir = output/f"latency_{checkpoint.name.removeprefix('checkpoint_')}"
            latency = evaluate("latency", checkpoint, latency_dir)
            atomic_json(output/"selection.json", dict(checkpoint=str(checkpoint), requested_updates=total_updates,
                pooled_persistent_tbler=result["summary"]["candidate_persistent"],
                cold_k8_accuracy_target_met=confirmation["cold_k8_accuracy_target_met"],
                memory_benefit_demonstrated=confirmation["memory_benefit_demonstrated"],
                accuracy_criteria_met=confirmation["accuracy_criteria_met"], latency_target_met=latency["gate_passed"],
                development_result=str(directory/"results.json"), latency_result=str(latency_dir/"results.json"),
                confirmation_result=str(confirmation_dir/"results.json"), anchors=str(anchors),
                scope=registration["selection"]+"; no automatic baseline promotion"))
            atomic_json(output/"status.json", dict(status="complete", completed_updates=total_updates,
                requested_updates=total_updates, best_checkpoint=str(checkpoint)))
        except BaseException as exc:
            atomic_json(output/"status.json", dict(status="failed", error_type=type(exc).__name__, message=str(exc)))
            raise


if __name__ == "__main__":
    main()
