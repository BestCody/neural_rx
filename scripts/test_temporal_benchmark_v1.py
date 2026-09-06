#!/usr/bin/env python3
"""Test the fixed Temporal Neural RX benchmark contract."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_temporal_benchmark import (
    pairing_signature,
    validate_pairing_signatures,
)
from rebind_temporal_benchmark_k3 import k2_latency_predecessor
from temporal_benchmark_cache import (
    compose_run_arrays,
    compose_run_bundle,
    write_candidate_output,
    write_reference_cache,
    write_static_reference_cache,
    load_static_reference_cache,
)
from temporal_benchmark_candidate import (
    build_candidate_manifest,
    validate_candidate_manifest,
    validate_matched_baseline,
)
from temporal_benchmark_io import (
    expected_array_shapes,
    load_run_bundle,
    validate_run_arrays,
)
from temporal_benchmark_plan import (
    build_calibration_artifact,
    build_confirmation_authorization,
    build_stage_plan,
    validate_calibration_artifact,
    validate_stage_plan,
)
from temporal_benchmark_protocol import (
    CONDITIONS,
    load_benchmark_protocol,
    stage_condition_names,
    stage_method_names,
    stage_reference_method_names,
    stage_seed_bundle,
)
from temporal_benchmark_runtime import (
    BenchmarkEpisode,
    evaluate_method_set,
)
from temporal_benchmark_scoring import (
    confirmation_precision_summary,
    paired_summary,
    score_run_group,
)


ROOT = Path(__file__).resolve().parents[1]


class ZeroAdapter:
    """Provide a deterministic contract fixture."""

    def __init__(self, manifest):
        self.manifest = manifest
        self.state = 0

    def start_episode(self):
        self.state = 0

    def start_episode_batch(self, batch_size):
        self.state = 0

    def reset_memory(self):
        self.state = 0

    def state_is_empty(self):
        return self.state == 0

    def decode_tb(self, inputs):
        users = len(inputs["ue_ids"])
        self.state += 1
        return {
            "hard_bits": np.zeros((users, 4), bool),
            "crc_status": np.ones(users, bool),
            "logits": np.zeros((users, 4), np.float32),
        }

    def decode_tb_batch(self, inputs):
        self.state += 1
        users = len(inputs[0]["ue_ids"])
        batch = len(inputs)
        return {
            "hard_bits": np.zeros((batch, users, 4), bool),
            "crc_status": np.ones((batch, users), bool),
            "logits": np.zeros((batch, users, 4), np.float32),
        }

    def synchronize(self):
        return None

    def checksums_unchanged(self):
        return True


class SyntheticScenarioBank:
    """Generate tiny deterministic episodes for runtime tests."""

    def generate(self, job, episode_index):
        tbs = int(job["num_tbs"])
        users = 2
        base = int(job["seed_streams"]["channel"]) + int(episode_index)
        active = np.ones((tbs, users), bool)
        ue_ids = np.tile(np.arange(users, dtype=np.int32), (tbs, 1))
        change_event = np.zeros((tbs, users), bool)
        if job["provider"] == "shared_geometry_change":
            change_event[job["provider_settings"]["change_tb"] - 1] = True
        if job["provider"] == "shared_geometry_schedule":
            active[4, 1] = False
            ue_ids[8:, 0] = 2
        bits = np.zeros((tbs, users, 4), bool)
        receiver = []
        oracle = []
        for snr_index, snr in enumerate(job["snr_db"]):
            receiver_tbs = []
            oracle_tbs = []
            for tb in range(tbs):
                point = float(base + 10 * snr_index + tb)
                inputs = {
                    "active": active[tb],
                    "ls_channel": np.full((users, 2), point, np.float32),
                    "noise_variance": np.asarray(10.0 ** (-snr / 10.0)),
                    "received_grid": np.full((2, 4), point, np.float32),
                    "slot_number": np.asarray(tb % 20, np.int32),
                    "ue_ids": ue_ids[tb],
                }
                receiver_tbs.append(inputs)
                oracle_tbs.append({
                    **inputs,
                    "perfect_channel": np.full(
                        (users, 2), point + 1.0, np.float32
                    ),
                })
            receiver.append(tuple(receiver_tbs))
            oracle.append(tuple(oracle_tbs))
        return BenchmarkEpisode(
            bits=bits,
            active=active,
            ue_ids=ue_ids,
            channel_change_event=change_event,
            receiver_inputs=tuple(receiver),
            oracle_inputs=tuple(oracle),
        )


class TemporalBenchmarkV1Test(unittest.TestCase):
    def setUp(self):
        self.protocol = load_benchmark_protocol(
            ROOT / "config" / "temporal_benchmark_v1.json"
        )
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.artifact = self.directory / "artifact.bin"
        self.artifact.write_bytes(b"benchmark fixture")
        self.manifests = self._manifests()
        self.calibration = self._calibration()
        self.plan = build_stage_plan(
            self.protocol,
            self.calibration,
            "screen",
            self.manifests["candidate"]["manifest_sha256"],
            self.manifests["matched_cold_k3"]["manifest_sha256"],
            self.manifests["matched_cold_k8"]["manifest_sha256"],
            self.manifests["historical_cold_k8"]["manifest_sha256"],
            self.manifests["perfect_csi_kbest"]["manifest_sha256"],
        )

    def test_k3_latency_protocol_has_exact_k2_predecessor(self):
        previous = k2_latency_predecessor(self.protocol)
        self.assertEqual(
            previous["protocol_sha256"],
            "a8df1130c018791d65790b2041aacd34c19891fedeebcdbb389518eb3714fc98",
        )
        self.assertIn("matched_cold_k2", previous["methods"])
        self.assertNotIn("matched_cold_k3", previous["methods"])
        self.assertEqual(
            previous["ranking"]["latency_reference"],
            "matched_cold_k2",
        )
        self.assertEqual(
            previous["execution"]["latency"]["reference_methods"],
            ["matched_cold_k2"],
        )

    def tearDown(self):
        self.temp.cleanup()

    def _manifest(self, role, recipe_digit, candidate_id=None):
        stage = self.protocol["stages"]["screen"]
        k = {
            "candidate": 2,
            "matched_cold_k3": 3,
            "matched_cold_k8": 8,
            "historical_cold_k8": 8,
            "perfect_csi_kbest": 0,
        }[role]
        is_kbest = role == "perfect_csi_kbest"
        seeds = [] if is_kbest else list(stage["training_recipe_seeds"])
        examples = 0 if is_kbest else stage["training_examples"]
        updates = 0 if is_kbest else stage["training_updates"]
        if role == "historical_cold_k8":
            seeds, examples, updates = [1999], 1, 1
        return build_candidate_manifest(
            candidate_id=candidate_id or f"fixture_{role}",
            role=role,
            architecture=f"fixture {role}",
            adapter_factory="test_temporal_benchmark_v1:make_adapter",
            benchmark_protocol_sha256=self.protocol["protocol_sha256"],
            radio=self.protocol["radio"],
            training_data_sha256=("1" if role != "historical_cold_k8" else "8") * 64,
            training_recipe_sha256=recipe_digit * 64,
            checkpoint_selection_sha256=(
                "2" if role != "historical_cold_k8" else "9"
            ) * 64,
            training_seeds=seeds,
            training_examples=examples,
            training_updates=updates,
            inference_iterations=k,
            persistent_memory=role == "candidate",
            memory_horizon_tbs=8 if role == "candidate" else None,
            checkpoint_artifacts=[] if is_kbest else [self.artifact],
            support_artifacts=[self.artifact] if is_kbest else [],
            structural_change=f"Fixture for {role}",
            hypothesis="The fixture obeys the benchmark contract.",
            expected_metric="worst_core_k8_gap",
            minimum_useful_improvement=0.005,
            maximum_gpu_hours=1.0,
            stop_rule="Stop after the frozen screen.",
        )

    def _manifests(self):
        return {
            "candidate": self._manifest(
                "candidate", "3", "fixture_candidate"
            ),
            "matched_cold_k3": self._manifest("matched_cold_k3", "4"),
            "matched_cold_k8": self._manifest("matched_cold_k8", "5"),
            "historical_cold_k8": self._manifest(
                "historical_cold_k8", "6"
            ),
            "perfect_csi_kbest": self._manifest(
                "perfect_csi_kbest", "7"
            ),
        }

    def _calibration(self):
        blocks = (
            self.protocol["calibration"]["episodes_per_condition"]
            * self.protocol["calibration"]["num_tbs"]
            * 2
        )
        counts = [1800, 1500, 1200, 900, 600, 400, 250, 180, 100, 40, 10]
        curves = {
            condition: [
                {"snr_db": float(snr), "errors": error, "blocks": blocks}
                for snr, error in zip(
                    self.protocol["calibration"]["snr_grid_db"], counts
                )
            ]
            for condition in CONDITIONS
        }
        return build_calibration_artifact(
            self.protocol,
            "a" * 64,
            self.manifests["matched_cold_k8"]["manifest_sha256"],
            curves,
        )

    def _full_arrays(self):
        shapes = expected_array_shapes(self.protocol, self.plan)
        conditions = [job["condition"] for job in self.plan["jobs"]]
        arrays = {
            "snr_db": np.asarray(
                [job["snr_db"] for job in self.plan["jobs"]], np.float64
            ),
            "episode_index": np.tile(
                np.arange(shapes["episode_index"][1], dtype=np.int32),
                (len(conditions), 1),
            ),
            "episode_fingerprint": np.zeros(
                shapes["episode_fingerprint"], np.uint8
            ),
            "active": np.ones(shapes["active"], bool),
            "ue_ids": np.tile(
                np.arange(2, dtype=np.int32),
                (len(conditions), shapes["active"][1], shapes["active"][2], 1),
            ),
            "channel_change_event": np.zeros(
                shapes["channel_change_event"], bool
            ),
        }
        for index, row in enumerate(
            arrays["episode_fingerprint"].reshape(-1, 32)
        ):
            row[:8] = np.frombuffer(
                int(index + 1).to_bytes(8, "little"), dtype=np.uint8
            )
        change_index = conditions.index("controlled_topology_change")
        change_tb = self.plan["jobs"][change_index]["provider_settings"][
            "change_tb"
        ]
        arrays["channel_change_event"][change_index, :, change_tb - 1] = True
        schedule_index = conditions.index("schedule_gap_reentry")
        arrays["active"][schedule_index, 0, 4, 1] = False
        arrays["ue_ids"][schedule_index, 0, 8:, 0] = 2
        periods = {
            "candidate_persistent": 7,
            "candidate_reset": 5,
            "matched_cold_k3": 6,
            "matched_cold_k8": 11,
            "historical_cold_k8": 10,
            "perfect_csi_kbest": 17,
        }
        flat_index = np.arange(np.prod(shapes["errors_candidate_persistent"]))
        for method in stage_method_names(self.protocol, self.plan["stage"]):
            arrays[f"errors_{method}"] = (
                flat_index % periods[method] == 0
            ).reshape(shapes[f"errors_{method}"]) & arrays["active"][:, None]
            arrays[f"crc_disagreement_{method}"] = np.zeros(
                shapes[f"crc_disagreement_{method}"], bool
            )
        latency_count = self.protocol["execution"]["latency"][
            "sample_count"
        ]
        arrays["latency_ms_candidate_persistent"] = np.linspace(
            5.0, 6.0, latency_count
        )
        arrays["latency_ms_matched_cold_k3"] = np.linspace(
            4.0, 5.0, latency_count
        )
        return arrays

    def _candidate_arrays(self, full):
        names = [
            "snr_db",
            "episode_index",
            "episode_fingerprint",
            "active",
            "ue_ids",
            "channel_change_event",
        ]
        for method in self.protocol["execution"]["candidate_methods"]:
            names += [f"errors_{method}", f"crc_disagreement_{method}"]
        names.append("latency_ms_candidate_persistent")
        return {name: full[name] for name in names}

    def _reference_arrays(self, full):
        names = [
            "snr_db",
            "episode_index",
            "episode_fingerprint",
            "active",
            "ue_ids",
            "channel_change_event",
        ]
        for method in self.protocol["execution"][
            "accuracy_reference_methods"
        ]:
            names += [f"errors_{method}", f"crc_disagreement_{method}"]
        for method in self.protocol["execution"]["latency"][
            "reference_methods"
        ]:
            names.append(f"latency_ms_{method}")
        return {name: full[name] for name in names}

    def _static_reference_arrays(self, full):
        names = [
            "snr_db",
            "episode_index",
            "episode_fingerprint",
            "active",
            "ue_ids",
            "channel_change_event",
        ]
        for method in self.protocol["execution"]["static_reference_methods"]:
            names += [f"errors_{method}", f"crc_disagreement_{method}"]
        return {name: full[name] for name in names}

    def test_protocol_plan_and_calibration_are_sealed(self):
        validate_calibration_artifact(self.calibration, self.protocol)
        validate_stage_plan(self.plan, self.protocol, self.calibration)
        self.assertEqual(len(self.plan["jobs"]), 4)
        self.assertEqual(
            tuple(job["condition"] for job in self.plan["jobs"]),
            stage_condition_names(self.protocol, "screen"),
        )
        seeds = stage_seed_bundle(
            self.protocol, "screen", "natural_umi", 0
        )
        self.assertEqual(len(set(seeds.values())), 4)

    def test_cold_reference_roles_do_not_duplicate_work(self):
        execution = self.protocol["execution"]
        self.assertEqual(
            execution["accuracy_reference_methods"],
            ["matched_cold_k8"],
        )
        self.assertEqual(
            execution["latency_only_reference_methods"],
            ["matched_cold_k3"],
        )
        self.assertEqual(
            execution["latency"]["reference_methods"],
            ["matched_cold_k3"],
        )

    def test_matched_baseline_allows_model_specific_recipe(self):
        candidate = self.manifests["candidate"]
        k8 = self.manifests["matched_cold_k8"]
        self.assertNotEqual(
            candidate["training"]["recipe_sha256"],
            k8["training"]["recipe_sha256"],
        )
        validate_matched_baseline(candidate, k8)

    def test_role_contract_rejects_wrong_iteration_count(self):
        broken = copy.deepcopy(self.manifests["candidate"])
        broken["inference"]["iterations"] = 3
        from temporal_generalization_protocol import seal_payload
        broken = seal_payload(broken, field="manifest_sha256")
        with self.assertRaisesRegex(ValueError, "K=2"):
            validate_candidate_manifest(broken, self.protocol)

    def test_cache_composition_and_scoring(self):
        full = self._full_arrays()
        candidate_dir = self.directory / "candidate"
        reference_dir = self.directory / "reference"
        candidate_path = write_candidate_output(
            candidate_dir,
            protocol=self.protocol,
            plan=self.plan,
            candidate=self.manifests["candidate"],
            training_seed=self.protocol["stages"]["screen"][
                "training_recipe_seeds"
            ][0],
            arrays=self._candidate_arrays(full),
            preflight={
                name: True
                for name in self.protocol["execution"]["candidate_preflight"]
            },
            elapsed_seconds=2.0,
        )
        baselines = {
            name: self.manifests[name]
            for name in stage_reference_method_names(
                self.protocol, self.plan["stage"]
            )
        }
        reference_path = write_reference_cache(
            reference_dir,
            protocol=self.protocol,
            plan=self.plan,
            baselines=baselines,
            training_seed=self.protocol["stages"]["screen"][
                "training_recipe_seeds"
            ][0],
            arrays=self._reference_arrays(full),
            preflight={
                name: True
                for name in self.protocol["execution"][
                    "matched_reference_preflight"
                ]
            },
            elapsed_seconds=4.0,
        )
        run_path = compose_run_bundle(
            self.directory / "run",
            protocol=self.protocol,
            plan=self.plan,
            candidate_output_path=candidate_path,
            reference_cache_path=reference_path,
        )
        payload, loaded = load_run_bundle(
            run_path, self.protocol, self.plan
        )
        score = score_run_group([payload], [loaded], self.protocol)
        self.assertEqual(score["candidate_id"], "fixture_candidate")
        self.assertIsNone(score["late_drift"])
        self.assertIsNone(score["adaptation"])
        self.assertIsNone(score["snr_at_target"])
        self.assertGreater(
            score["latency"]["relative_to"]["matched_cold_k3"][
                "p95_ratio"
            ],
            1.0,
        )

    def test_cache_refuses_unpaired_episode(self):
        full = self._full_arrays()
        candidate = self._candidate_arrays(full)
        references = self._reference_arrays(full)
        references = {name: value.copy() for name, value in references.items()}
        references["episode_fingerprint"][0, 0, 0] ^= 1
        with self.assertRaisesRegex(ValueError, "episode_fingerprint"):
            compose_run_arrays(candidate, references, self.protocol, self.plan)

    def test_run_refuses_unplanned_snr(self):
        arrays = self._full_arrays()
        arrays["snr_db"] = arrays["snr_db"].copy()
        arrays["snr_db"][0, 0] += 0.25
        with self.assertRaisesRegex(ValueError, "SNR values"):
            validate_run_arrays(arrays, self.protocol, self.plan)

    def test_run_refuses_partial_latency_trace(self):
        arrays = self._full_arrays()
        arrays["latency_ms_matched_cold_k3"] = arrays[
            "latency_ms_matched_cold_k3"
        ][:-1]
        with self.assertRaisesRegex(ValueError, "sample count"):
            validate_run_arrays(arrays, self.protocol, self.plan)

    def test_primary_gap_uses_only_calibrated_anchor(self):
        count = len(self.protocol["stages"]["development"]["condition_names"])
        shape = (count, 3, 2, 64, 2)
        active = np.ones((count, 2, 64, 2), bool)
        left = np.zeros(shape, bool)
        right = np.zeros(shape, bool)
        left[:, 0] = True
        left[:, 2] = True
        arrays = {
            "errors_candidate_persistent": left,
            "errors_matched_cold_k8": right,
            "active": active,
        }
        summary = paired_summary(
            [arrays],
            "candidate_persistent",
            "matched_cold_k8",
            self.protocol,
            "development",
            100,
            np.random.RandomState(1),
        )
        self.assertEqual(summary["difference"], 0.0)

    def test_leaderboard_refuses_changed_reference(self):
        arrays = self._full_arrays()
        payload = {
            "stage": "screen",
            "training_seed": self.protocol["stages"]["screen"][
                "training_recipe_seeds"
            ][0]
        }
        first = pairing_signature([payload], [arrays], self.protocol)
        changed = {name: value.copy() for name, value in arrays.items()}
        changed["errors_matched_cold_k8"][0, 0, 0, 0, 0] ^= True
        second = pairing_signature([payload], [changed], self.protocol)
        with self.assertRaisesRegex(ValueError, "matched_cold_k8"):
            validate_pairing_signatures(first, second)

    def test_runtime_owns_pairing_and_reset(self):
        methods = tuple(self.protocol["execution"]["candidate_methods"])
        adapters = {
            method: ZeroAdapter(self.manifests["candidate"])
            for method in methods
        }
        arrays, preflight, elapsed = evaluate_method_set(
            self.protocol,
            self.plan,
            SyntheticScenarioBank(),
            adapters,
            methods,
            latency_methods=("candidate_persistent",),
        )
        self.assertTrue(all(preflight.values()))
        self.assertGreater(elapsed, 0.0)
        self.assertGreaterEqual(
            len(arrays["latency_ms_candidate_persistent"]), 10
        )
        self.assertFalse(np.any(arrays["errors_candidate_persistent"]))

    def test_runtime_rejects_wrong_timing_set_before_generation(self):
        methods = tuple(self.protocol["execution"]["candidate_methods"])
        adapters = {
            method: ZeroAdapter(self.manifests["candidate"])
            for method in methods
        }
        with self.assertRaisesRegex(ValueError, "latency methods"):
            evaluate_method_set(
                self.protocol,
                self.plan,
                SyntheticScenarioBank(),
                adapters,
                methods,
            )

    def test_runtime_times_paired_references(self):
        methods = tuple(
            self.protocol["execution"]["accuracy_reference_methods"]
        )
        latency_methods = tuple(
            self.protocol["execution"]["latency"]["reference_methods"]
        )
        adapter_methods = tuple(dict.fromkeys(methods + latency_methods))
        adapters = {
            method: ZeroAdapter(self.manifests[method])
            for method in adapter_methods
        }
        arrays, preflight, _ = evaluate_method_set(
            self.protocol,
            self.plan,
            SyntheticScenarioBank(),
            adapters,
            methods,
            latency_methods=latency_methods,
        )
        self.assertTrue(all(preflight.values()))
        self.assertIn("errors_matched_cold_k8", arrays)
        self.assertNotIn("errors_matched_cold_k3", arrays)
        self.assertNotIn("latency_ms_matched_cold_k8", arrays)
        for method in self.protocol["execution"]["latency"][
            "reference_methods"
        ]:
            self.assertGreaterEqual(
                len(arrays[f"latency_ms_{method}"]), 10
            )

    def test_confirmation_uses_progressive_episode_schedule(self):
        authorization = build_confirmation_authorization(
            self.protocol,
            self.calibration,
            self.manifests["candidate"]["manifest_sha256"],
            "f" * 64,
        )
        plan = build_stage_plan(
            self.protocol,
            self.calibration,
            "confirmation",
            self.manifests["candidate"]["manifest_sha256"],
            self.manifests["matched_cold_k3"]["manifest_sha256"],
            self.manifests["matched_cold_k8"]["manifest_sha256"],
            self.manifests["historical_cold_k8"]["manifest_sha256"],
            self.manifests["perfect_csi_kbest"]["manifest_sha256"],
            authorization=authorization,
            episodes_per_condition=128,
        )
        self.assertEqual(plan["episodes_per_condition"], 128)
        with self.assertRaisesRegex(ValueError, "episode count"):
            build_stage_plan(
                self.protocol,
                self.calibration,
                "confirmation",
                self.manifests["candidate"]["manifest_sha256"],
                self.manifests["matched_cold_k3"]["manifest_sha256"],
                self.manifests["matched_cold_k8"]["manifest_sha256"],
                self.manifests["historical_cold_k8"]["manifest_sha256"],
                self.manifests["perfect_csi_kbest"]["manifest_sha256"],
                authorization=authorization,
                episodes_per_condition=96,
            )

    def test_confirmation_precision_requests_only_needed_batch(self):
        result = confirmation_precision_summary(
            self.protocol,
            "confirmation",
            64,
            {"worst_condition_ci95": [-0.02, 0.02]},
            {"ci95": [-0.005, 0.005]},
        )
        self.assertFalse(result["target_met"])
        self.assertEqual(result["next_episode_count"], 128)
        passed = confirmation_precision_summary(
            self.protocol,
            "confirmation",
            64,
            {"worst_condition_ci95": [-0.005, 0.005]},
            {"ci95": [-0.004, 0.004]},
        )
        self.assertTrue(passed["target_met"])
        self.assertIsNone(passed["next_episode_count"])

    def test_static_references_are_cached_without_training_seed(self):
        authorization = build_confirmation_authorization(
            self.protocol,
            self.calibration,
            self.manifests["candidate"]["manifest_sha256"],
            "e" * 64,
        )
        original = self.plan
        self.plan = build_stage_plan(
            self.protocol,
            self.calibration,
            "confirmation",
            self.manifests["candidate"]["manifest_sha256"],
            self.manifests["matched_cold_k3"]["manifest_sha256"],
            self.manifests["matched_cold_k8"]["manifest_sha256"],
            self.manifests["historical_cold_k8"]["manifest_sha256"],
            self.manifests["perfect_csi_kbest"]["manifest_sha256"],
            authorization=authorization,
        )
        try:
            full = self._full_arrays()
            baselines = {
                name: self.manifests[name]
                for name in self.protocol["execution"][
                    "static_reference_methods"
                ]
            }
            path = write_static_reference_cache(
                self.directory / "static",
                protocol=self.protocol,
                plan=self.plan,
                baselines=baselines,
                arrays=self._static_reference_arrays(full),
                preflight={
                    name: True
                    for name in self.protocol["execution"][
                        "static_reference_preflight"
                    ]
                },
                elapsed_seconds=3.0,
            )
            payload, static_arrays = load_static_reference_cache(
                path, self.protocol, self.plan
            )
            self.assertNotIn("training_seed", payload)
            combined = compose_run_arrays(
                self._candidate_arrays(full),
                self._reference_arrays(full),
                self.protocol,
                self.plan,
                static_arrays,
            )
            for method in stage_method_names(
                self.protocol, "confirmation"
            ):
                self.assertIn(f"errors_{method}", combined)
        finally:
            self.plan = original


if __name__ == "__main__":
    unittest.main()
