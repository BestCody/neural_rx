#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cjlr_final_runtime import load_v18_segment_safe_config
from temporal_benchmark_adapters import V18BenchmarkAdapter


ROOT = Path(__file__).resolve().parent.parent


class TemporalReceiverV18Test(unittest.TestCase):
    def test_changed_schedule_is_rejected(self):
        source = (
            ROOT / "config" / "temporal_receiver_v18_segment_safe.json"
        )
        value = json.loads(source.read_text(encoding="utf-8"))
        value["receiver"]["support_width"] = 128
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "settings changed"):
                load_v18_segment_safe_config(path)

    def test_segment_safe_config_is_valid(self):
        value = load_v18_segment_safe_config(
            ROOT / "config" / "temporal_receiver_v18_segment_safe.json"
        )
        self.assertEqual(value["memory"]["minimum_prior_innovations"], 1)
        self.assertEqual(value["memory"]["refractory_tbs"], 0)
        self.assertEqual(value["receiver"]["learned_iterations"], 2)
        self.assertEqual(
            value["checkpoint"]["binding"],
            "sealed_candidate_manifest",
        )

    def test_v18_uses_streamed_memory(self):
        memory = V18BenchmarkAdapter._memory_type()
        self.assertEqual(
            memory.__name__,
            "OnlineStreamedSegmentSafePilotDomainMemory",
        )


if __name__ == "__main__":
    unittest.main()
