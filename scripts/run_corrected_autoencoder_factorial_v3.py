#!/usr/bin/env python3
"""Second-pass audit layer for the corrected protocol-v2 AE factorial.

This module hardens the v2 runner without changing the research protocol. It
adds exact training-history validation, mandatory smoke provenance, evaluator
source/checkpoint binding, and recomputation of reported evaluation metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

from temporal_eval_metrics import log_bler_crossing
import run_corrected_autoencoder_factorial_v2 as v2


_ORIGINAL_PREFLIGHT = v2.preflight
_ORIGINAL_STABILITY = v2.stability
_ORIGINAL_EVALUATION_VALID = v2.evaluation_valid
_ORIGINAL_EVALUATE_CELL = v2.evaluate_cell

SMOKE_PATH = Path("/tmp/temporal_autoencoder_stability/summary.json")
EVAL_PROVENANCE_VERSION = 1


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _close(a, b, tol=1e-8) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def tracked_tree_clean_strict() -> bool:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=str(v2.SCRIPT_DIR.parent),
        text=True,
    )
    return not status.strip()


def stability_strict(summary: dict, d_mem: int) -> dict:
    base = _ORIGINAL_STABILITY(summary, d_mem)
    history = summary.get("history") or []

    expected_steps = list(range(0, v2.TRAIN["train_steps"], 25))
    if expected_steps[-1] != v2.TRAIN["train_steps"] - 1:
        expected_steps.append(v2.TRAIN["train_steps"] - 1)
    observed_steps = [int(row.get("step", -1)) for row in history]

    exact_history = observed_steps == expected_steps
    phase_ok = True
    compression_ok = True
    schedule_ok = True
    loss_equation_ok = True
    all_gradients_positive = True
    max_weighted_ratio = 0.0
    valid_fraction_ok = True
    gap_ok = True

    for row in history:
        step = int(row.get("step", -1))
        expected_phase = (
            "memory_only"
            if step < v2.TRAIN["memory_only_steps"]
            else "joint"
        )
        phase_ok = phase_ok and row.get("phase") == expected_phase
        compression_ok = compression_ok and row.get("compression") == "autoencoder"
        schedule_ok = schedule_ok and _close(row.get("schedule_change_fraction"), 0.0)

        data = row.get("loss_data")
        chest = row.get("loss_chest")
        aux = row.get("compression_aux_loss")
        total = row.get("loss")
        if all(_finite(x) for x in (data, chest, aux, total)):
            expected_total = (
                float(data)
                + v2.TRAIN["chest_weight"] * float(chest)
                + v2.TRAIN["ae_reconstruction_weight"] * float(aux)
            )
            loss_equation_ok = loss_equation_ok and _close(total, expected_total, 2e-5)
            if float(data) > 0:
                max_weighted_ratio = max(
                    max_weighted_ratio,
                    v2.TRAIN["ae_reconstruction_weight"] * float(aux) / float(data),
                )
            else:
                max_weighted_ratio = math.inf
        else:
            loss_equation_ok = False

        grad = row.get("gradient_norm")
        all_gradients_positive = (
            all_gradients_positive and _finite(grad) and float(grad) > 0.0
        )

        valid = row.get("memory_valid_fraction_per_tb") or []
        gaps = row.get("memory_gap_mean_per_tb") or []
        valid_fraction_ok = valid_fraction_ok and len(valid) == v2.TRAIN["seq_len"]
        gap_ok = gap_ok and len(gaps) == v2.TRAIN["seq_len"]
        if len(valid) == v2.TRAIN["seq_len"]:
            valid_fraction_ok = valid_fraction_ok and all(
                _finite(x) and 0.0 <= float(x) <= 1.0 for x in valid
            )
            # Fixed two-UE schedule: no memory at TB1, valid memory thereafter.
            valid_fraction_ok = valid_fraction_ok and all(
                _close(a, b, 1e-7)
                for a, b in zip(valid, [0.0, 1.0, 1.0, 1.0])
            )
        if len(gaps) == v2.TRAIN["seq_len"]:
            gap_ok = gap_ok and all(_finite(x) and float(x) >= 0.0 for x in gaps)
            gap_ok = gap_ok and all(
                _close(a, b, 1e-7)
                for a, b in zip(gaps, [0.0, 1.0, 1.0, 1.0])
            )

    strict_passed = bool(
        base.get("passed") is True
        and exact_history
        and phase_ok
        and compression_ok
        and schedule_ok
        and loss_equation_ok
        and all_gradients_positive
        and max_weighted_ratio < 1.0
        and valid_fraction_ok
        and gap_ok
    )
    return {
        **base,
        "expected_logged_points": len(expected_steps),
        "observed_logged_points": len(history),
        "exact_logged_step_sequence": exact_history,
        "phase_boundary_consistent": phase_ok,
        "compression_label_consistent": compression_ok,
        "fixed_schedule_consistent": schedule_ok,
        "loss_equation_consistent": loss_equation_ok,
        "all_logged_gradient_norms_positive": all_gradients_positive,
        "max_weighted_reconstruction_over_data": max_weighted_ratio,
        "memory_validity_pattern_consistent": valid_fraction_ok,
        "memory_gap_pattern_consistent": gap_ok,
        "passed": strict_passed,
    }


def _validate_smoke() -> dict:
    if not SMOKE_PATH.is_file():
        raise RuntimeError(
            f"required corrected-AE stability smoke is missing: {SMOKE_PATH}"
        )
    smoke = json.loads(SMOKE_PATH.read_text())
    failures = []

    def req(ok, label):
        if not ok:
            failures.append(label)

    req(smoke.get("purpose") == "stability_smoke_not_research_result", "purpose")
    req(int(smoke.get("steps", -1)) == 150, "steps")
    req(int(smoke.get("memory_only_steps", -1)) == 50, "memory_only_steps")
    req(smoke.get("passed") is True, "overall_passed")

    cases = {c.get("case"): c for c in (smoke.get("cases") or [])}
    req(set(cases) == {"mean_autoencoder_d8", "cnn_autoencoder_d16"}, "cases")
    for name, d_mem in (("mean_autoencoder_d8", 8), ("cnn_autoencoder_d16", 16)):
        case = cases.get(name) or {}
        protocol = case.get("autoencoder_protocol") or {}
        req(case.get("passed") is True, f"{name}:passed")
        req(protocol.get("version") == 2, f"{name}:protocol_v2")
        req(protocol.get("bounded_tanh_bottleneck") is True, f"{name}:bounded")
        req(
            protocol.get("reconstruction_upstream_state_detached") is True,
            f"{name}:detached",
        )
        req(
            protocol.get("scale_normalized_reconstruction_aux_loss") is True,
            f"{name}:normalized_reconstruction",
        )
        req(case.get("finite") is True, f"{name}:finite")
        req((case.get("gradient_check") or {}).get("passed") is True, f"{name}:gradient")
        req(
            _finite(case.get("max_memory_norm"))
            and float(case.get("max_memory_norm")) <= math.sqrt(d_mem) + 1e-3,
            f"{name}:memory_bound",
        )
        req(
            _finite(case.get("last_weighted_reconstruction_over_data"))
            and float(case.get("last_weighted_reconstruction_over_data")) < 1.0,
            f"{name}:reconstruction_ratio",
        )

    if failures:
        raise RuntimeError(
            "corrected-AE smoke provenance failed: " + ", ".join(failures)
        )
    return {
        "path": str(SMOKE_PATH),
        "sha256": _sha256(SMOKE_PATH),
        "cases": sorted(cases),
        "passed": True,
    }


def preflight_strict(current_commit: str) -> dict:
    # Make v2's preflight reject untracked source files as well as tracked edits.
    v2.tracked_tree_clean = tracked_tree_clean_strict
    report = _ORIGINAL_PREFLIGHT(current_commit)
    smoke = _validate_smoke()
    report["strict_smoke_provenance"] = smoke
    report["strict_second_pass"] = True
    v2.write_json(v2.ROOT / "corrected_autoencoder_preflight.json", report)
    print("CORRECTED_AE_SECOND_PASS_PREFLIGHT=" + json.dumps(report, indent=2), flush=True)
    return report


def evaluation_valid_strict(
    evaluation: dict,
    pooling: str,
    d_mem: int,
    checkpoint_sha: str,
):
    ok, failures = _ORIGINAL_EVALUATION_VALID(
        evaluation, pooling, d_mem, checkpoint_sha
    )
    failures = list(failures)
    curves = evaluation.get("curves") or {}
    stored = evaluation.get("snr_db_at_10pct_tbler") or {}

    for method in ("cold_k2", "cold_k8", "temporal_k2"):
        points = curves.get(method) or []
        try:
            recomputed = log_bler_crossing(points, target=0.1)
        except Exception as exc:
            failures.append(f"{method}:crossing_recompute:{type(exc).__name__}")
            continue
        saved = stored.get(method)
        if recomputed is None or saved is None:
            if recomputed is not None or saved is not None:
                failures.append(f"{method}:crossing_null_mismatch")
        elif not _close(recomputed, saved, 1e-9):
            failures.append(f"{method}:crossing_value_mismatch")

    expected_grid = evaluation.get("snr_grid_db") or []
    for idx in range(len(expected_grid)):
        batches = []
        blocks = []
        for method in ("cold_k2", "cold_k8", "temporal_k2"):
            points = curves.get(method) or []
            if idx >= len(points):
                continue
            p = points[idx]
            e = p.get("errors_tb2plus")
            n = p.get("blocks_tb2plus")
            if not (_finite(e) and _finite(n) and 0 <= int(e) <= int(n)):
                failures.append(f"{method}:{idx}:error_block_consistency")
            batches.append(p.get("batches"))
            blocks.append(n)
        if batches and len(set(batches)) != 1:
            failures.append(f"snr:{idx}:paired_batch_count_mismatch")
        if blocks and len(set(blocks)) != 1:
            failures.append(f"snr:{idx}:paired_block_count_mismatch")

    c2 = stored.get("cold_k2")
    c8 = stored.get("cold_k8")
    ct = stored.get("temporal_k2")
    if all(x is not None and _finite(x) for x in (c2, c8)):
        gap = float(c2) - float(c8)
        if not _close(evaluation.get("cold_iteration_gap_db"), gap, 1e-9):
            failures.append("derived:cold_gap")
        if ct is not None and _finite(ct):
            improvement = float(c2) - float(ct)
            recovered = improvement / gap if gap > 0 else None
            if not _close(
                evaluation.get("temporal_improvement_over_cold_k2_db"),
                improvement,
                1e-9,
            ):
                failures.append("derived:temporal_improvement")
            expected_percent = None if recovered is None else recovered * 100.0
            if expected_percent is not None and not _close(
                evaluation.get("gap_recovered_percent"), expected_percent, 1e-7
            ):
                failures.append("derived:gap_recovered_percent")

    return (ok and not failures), failures


def _eval_sidecar(pooling: str, d_mem: int) -> Path:
    return v2.evaluation_dir(pooling, d_mem) / "corrected_ae_evaluation_provenance.json"


def evaluate_cell_strict(
    gpu: int,
    ckpt: Path,
    pooling: str,
    d_mem: int,
    current_commit: str,
) -> dict:
    eval_script = v2.SCRIPT_DIR / "evaluate_temporal_ue_memory_v2.py"
    evaluator_sha = _sha256(eval_script)
    manifest = v2.load_json(v2.manifest_path(pooling, d_mem))
    checkpoint_sha = manifest["checkpoint_sha256"]
    sidecar_path = _eval_sidecar(pooling, d_mem)
    evaluation_file = v2.evaluation_dir(pooling, d_mem) / "evaluation.json"

    # An old evaluation is reusable only when it is bound to both the same
    # checkpoint and the same evaluator source. Otherwise force v2 to rerun it
    # by removing only the reuse stamp; the old curve data remains inspectable.
    provenance_ok = False
    if sidecar_path.is_file():
        try:
            sidecar = v2.load_json(sidecar_path)
            provenance_ok = bool(
                int(sidecar.get("version", -1)) == EVAL_PROVENANCE_VERSION
                and sidecar.get("checkpoint_sha256") == checkpoint_sha
                and sidecar.get("evaluator_sha256") == evaluator_sha
            )
        except Exception:
            provenance_ok = False
    if evaluation_file.is_file() and not provenance_ok:
        try:
            existing = v2.load_json(evaluation_file)
            existing.pop("corrected_ae_evaluation_stamp", None)
            v2.write_json(evaluation_file, existing)
            print(
                f"INVALIDATE_UNBOUND_CORRECTED_AE_EVALUATION={evaluation_file.parent}",
                flush=True,
            )
        except Exception:
            pass

    evaluation = _ORIGINAL_EVALUATE_CELL(
        gpu, ckpt, pooling, d_mem, current_commit
    )
    valid, failures = evaluation_valid_strict(
        evaluation, pooling, d_mem, checkpoint_sha
    )
    if not valid:
        raise RuntimeError(
            f"strict evaluation audit failed for {pooling}/d{d_mem}: {failures}"
        )

    sidecar = {
        "version": EVAL_PROVENANCE_VERSION,
        "pooling": pooling,
        "d_mem": int(d_mem),
        "checkpoint_sha256": checkpoint_sha,
        "evaluator": str(eval_script),
        "evaluator_sha256": evaluator_sha,
        "evaluation_json_sha256": _sha256(evaluation_file),
        "audit_runner_commit": current_commit,
    }
    v2.write_json(sidecar_path, sidecar)
    return evaluation


# Patch the v2 module globals that v2.main resolves at runtime.
v2.stability = stability_strict
v2.preflight = preflight_strict
v2.evaluation_valid = evaluation_valid_strict
v2.evaluate_cell = evaluate_cell_strict


def main():
    v2.main()


if __name__ == "__main__":
    main()
