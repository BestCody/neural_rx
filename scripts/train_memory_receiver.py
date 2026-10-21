from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import fcntl
import json
import os
from pathlib import Path
import sys

from pilot_history_receiver.training_options import (add_execution_arguments, add_preparation_arguments,
    configure_gpu, requested_updates, resolve_preparation)
from pilot_history_receiver.checkpoints import recover_history, recover_latest
from artifact_io import atomic_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def parser():
    p = argparse.ArgumentParser(description="Train K2 and channel memory; run_memory_training.py adds periodic development evaluation")
    p.add_argument("--recipe", type=Path, default=ROOT/"config/memory_training.json")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int)
    p.add_argument("--steps", type=int, required=True, help="Total updates, including restored updates; no training-length ceiling")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Checkpoint and continuation directory under models/")
    p.add_argument("--resume", type=Path)
    add_execution_arguments(p)
    add_preparation_arguments(p)
    return p


def run_training(args, *, evaluation_points=(), evaluate=None):
    recipe = json.loads(args.recipe.read_text())
    stop = requested_updates(args.steps)
    points = sorted(set(evaluation_points))
    if any(type(point) is not int or point < 1 or point > stop for point in points):
        raise ValueError("Development updates must lie inside the requested training budget")
    if points and evaluate is None:
        raise ValueError("Development updates require an evaluator")
    resolve_preparation(args, recipe["effective_examples"])
    configure_gpu(args.gpu)
    from pilot_history_receiver.training import MemoryTrainer
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/"run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        latest = recover_latest(output, lock_held=True)
        if latest is not None and args.resume is None:
            raise FileExistsError("Use --resume for an existing training run")
        if latest is not None and args.resume.resolve() != output/latest["checkpoint"]:
            raise ValueError(f"Resume the latest published checkpoint: {output/latest['checkpoint']}")
        trainer = MemoryTrainer(recipe, args.seed, args.detector_batch_tbs, args.descriptor_chunk_tbs)
        trainer.episodes.render_batch_tbs = args.render_batch_tbs
        if args.resume is not None:
            trainer.restore(args.resume)
        if int(trainer.step.numpy()) > stop:
            raise ValueError("Requested stop precedes the restored update")
        recover_history(output/"history.jsonl", int(trainer.step.numpy()))
        if latest is None:
            trainer.save(output)
        run = {"binding": trainer.binding(), "schedule": trainer.schedule_state(),
                    "pid": os.getpid(), "evaluation_updates": points,
                    "requested_stop": stop, "status": "running", "data_gpu": args.data_gpu,
                    "prefetch_examples": (args.prefetch_examples or recipe["effective_examples"]) if args.data_gpu is not None else 0,
                    "channel_workers": args.channel_workers, "prefetch_memory_mib": args.prefetch_memory_mib,
                    "render_batch_tbs": args.render_batch_tbs,
                    "replica_gpu": args.replica_gpu, "replica_memory_mib": args.replica_memory_mib,
                    "preparation_gpu_memory_mib": args.preparation_gpu_memory_mib}
        atomic_json(output/"run.json", run)
        for name in ("completion.json", "failure.json"):
            (output/name).unlink(missing_ok=True)
        try:
            from pilot_history_receiver.training_prefetch import EpisodePrefetch

            def observe():
                step = int(trainer.step.numpy())
                checkpoint = output/f"checkpoint_{step:06d}"
                atomic_json(output/"run.json", dict(run, status="evaluating", last_completed_update=step,
                                                   schedule=trainer.schedule_state()))
                result = evaluate(checkpoint)
                if Path(result["checkpoint"]).resolve() != checkpoint:
                    raise ValueError("Development feedback evaluated a different checkpoint")
                trainer.observe_development(result)
                atomic_json(output/"run.json", dict(run, last_completed_update=step,
                                                   schedule=trainer.schedule_state()))

            if int(trainer.step.numpy()) in points:
                observe()
            boundaries = sorted({stop, *(point for point in points if point > int(trainer.step.numpy()))})
            with (output/"history.jsonl").open("a", buffering=1) as log:
                for boundary in boundaries:
                    first = int(trainer.step.numpy())+1
                    if first > boundary:
                        continue
                    with ExitStack() as workers:
                        if args.replica_gpu is not None:
                            from pilot_history_receiver.training_parallel import GradientReplica
                            trainer.replica = workers.enter_context(GradientReplica(
                                trainer, args.replica_gpu, args.replica_memory_mib))
                        producer = (nullcontext(None) if args.data_gpu is None else
                            EpisodePrefetch(recipe, trainer.seed, first, boundary, args.data_gpu,
                                capacity=args.prefetch_examples or None, channel_workers=args.channel_workers,
                                byte_capacity=args.prefetch_memory_mib*1024**2,
                                gpu_memory_mib=args.preparation_gpu_memory_mib,
                                render_batch_tbs=args.render_batch_tbs))
                        source = workers.enter_context(producer)
                        while int(trainer.step.numpy()) < boundary:
                            packets = None if source is None else source.get(int(trainer.step.numpy())+1)
                            record, _ = trainer.update(packets)
                            log.write(json.dumps(record, allow_nan=False)+"\n")
                            print(json.dumps(record, allow_nan=False), flush=True)
                            step = int(trainer.step.numpy())
                            if step % recipe["checkpoint_every"] == 0 or step == boundary:
                                log.flush()
                                os.fsync(log.fileno())
                                trainer.save(output)
                            atomic_json(output/"progress.json", record)
                    trainer.replica = None
                    if boundary in points:
                        observe()
            atomic_json(output/"completion.json", {"status": "complete", "updates": stop,
                        "checkpoint": str(output/f"checkpoint_{stop:06d}"), "schedule": trainer.schedule_state()})
            atomic_json(output/"run.json", dict(run, status="complete", last_completed_update=stop,
                                               schedule=trainer.schedule_state()))
        except BaseException as exc:
            (output/"completion.json").unlink(missing_ok=True)
            atomic_json(output/"failure.json", {"type": type(exc).__name__, "message": str(exc),
                        "last_completed_update": int(trainer.step.numpy())})
            atomic_json(output/"run.json", dict(run, status="failed", last_completed_update=int(trainer.step.numpy())))
            raise


def main():
    run_training(parser().parse_args())


if __name__ == "__main__":
    main()
