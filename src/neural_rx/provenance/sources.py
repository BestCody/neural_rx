from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from artifact_io import file_sha256

ROOT = Path(__file__).resolve().parent.parent
TRAINING_PROTOCOL = "cnn_memory_training"
TRAINING_SCHEMA = 5
TRAINING_SEED_NAMESPACE = "calibrated_feedback_joint_training_v1"


def _module_file(name, root):
    for base in (root, root / "scripts"):
        path = base.joinpath(*name.split("."))
        for candidate in (path.with_suffix(".py"), path / "__init__.py"):
            if candidate.is_file():
                return candidate
    return None


def source_records(paths, root=ROOT):
    root = Path(root).resolve()
    pending = [root / path for path in paths]
    records = {}
    while pending:
        path = pending.pop().resolve()
        name = path.relative_to(root).as_posix()
        if name in records:
            continue
        records[name] = file_sha256(path)
        if path.parent.name == 'native_ops':
            pending.extend(p for p in path.parent.glob('*.cc') if p.relative_to(root).as_posix() not in records)
        if path.suffix != ".py":
            continue
        for parent in path.parents:
            if parent == root:
                break
            initializer = parent / "__init__.py"
            if initializer.is_file():
                pending.append(initializer)
        package = list(Path(name).parts[:-1])
        if package and package[0] == "scripts":
            package = package[1:]
        for node in ast.walk(ast.parse(path.read_text(), filename=name)):
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                parts = package[:len(package) - node.level + 1] if node.level else []
                if node.module:
                    parts += node.module.split(".")
                module = ".".join(parts)
                modules = [module] + [".".join([*parts, item.name]) for item in node.names if item.name != "*"]
            else:
                continue
            for module in modules:
                dependency = _module_file(module, root) if module else None
                if dependency is not None:
                    pending.append(dependency)
    return dict(sorted(records.items()))


def evaluation_source_records(root=ROOT):
    return source_records(("scripts/temporal_benchmark_frozen.py",
                           "config/nrx_large.cfg", "config/temporal_benchmark_v1.json"), root)


def validate_evaluation_sources(records, root=ROOT):
    if records != evaluation_source_records(root):
        raise ValueError("Evaluation source changed; rerun the evaluation")


def training_source_records(radio_config, root=ROOT):
    return source_records(("scripts/train_memory_receiver.py",
                           f"config/{Path(radio_config).name}"), root)


def normalize_training_recipe(recipe):
    if recipe.get("protocol") != TRAINING_PROTOCOL or recipe.get("schema_version") != TRAINING_SCHEMA:
        raise ValueError("Unknown training contract or schema")
    return copy.deepcopy(recipe)


def validate_training_binding(saved, current):
    normalized = []
    for binding in (saved, current):
        expected = hashlib.sha256(json.dumps(binding["recipe"], sort_keys=True).encode()).hexdigest()
        if binding["recipe_sha256"] != expected:
            raise ValueError("Training recipe checksum differs")
        contract = {k: normalize_training_recipe(v) if k == "recipe" else v
                    for k, v in binding.items() if k not in ("sources", "recipe_sha256")}
        contract["execution"] = {k: v for k, v in contract["execution"].items() if k not in
            ("descriptor_chunk_tbs", "detector_batch_tbs")}
        normalized.append(contract)
    if normalized[0] != normalized[1]:
        raise ValueError("Resume recipe, architecture, seed, or execution semantics changed")
