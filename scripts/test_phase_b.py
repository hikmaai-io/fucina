#!/usr/bin/env python3
# ABOUTME: Unit tests for Phase-B corpus, policy, and quality-gate tooling.
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from build_calibration_corpus import build
from derive_precision_policy import derive
from derive_residency_plan import derive as derive_residency
from quality_gate import compare, parse_report


class PhaseBTests(unittest.TestCase):
    def test_corpus_is_deduplicated_and_provenance_carried(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "code.jsonl").write_text('{"text":"fix bug"}\n{"text":"fix bug"}\n')
            (root / "red.txt").write_text("detect injection\n")
            recipe = {"seed": 1, "target_tokens": 10, "categories": {"code": .5, "red": .5},
                      "sources": [
                          {"path": "code.jsonl", "category": "code", "provenance": "local", "license": "MIT"},
                          {"path": "red.txt", "category": "red", "provenance": "local", "license": "CC0"}]}
            rp, out = root / "recipe.json", root / "out.jsonl"
            rp.write_text(json.dumps(recipe))
            manifest = build(rp, out)
            rows = [json.loads(x) for x in out.read_text().splitlines()]
            self.assertEqual(manifest["unique_input_records"], 2)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all("provenance" in x and "sha256" in x for x in rows))

    def test_policy_never_emits_int2_without_capability(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "i.json"
            tensors = {f"layers.0.mlp.experts.{e}.{p}.weight": (e + 1) / 10
                       for e in range(10) for p in ("gate_proj", "up_proj", "down_proj")}
            tensors["layers.0.input_layernorm.weight"] = 1
            path.write_text(json.dumps({"format": "fucina-imatrix-v1", "model": "m",
                                        "tensor_importance": tensors}))
            safe = derive(path)
            self.assertNotIn("int2", safe["codec_tensor_counts"])
            self.assertEqual(safe["tensor_policy"]["layers.0.input_layernorm.weight"]["tier"], "critical")
            aggressive = derive(path, sub4_kernel=True)
            self.assertGreater(aggressive["codec_tensor_counts"].get("int2", 0), 0)

    def test_residency_plan_prefers_hot_experts_and_respects_budgets(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "i.json"
            heat = [{"layer": 0, "assignments": 100, "experts": [
                {"expert": 0, "count": 70, "frequency": .7, "mean_weight": .8, "importance": 1},
                {"expert": 1, "count": 20, "frequency": .2, "mean_weight": .5, "importance": .5},
                {"expert": 2, "count": 10, "frequency": .1, "mean_weight": .2, "importance": .2}]}]
            path.write_text(json.dumps({"format": "fucina-imatrix-v1", "model": "m",
                                        "expert_heat_map": heat}))
            # hidden=intermediate=16, 4 bits => 384 bytes/expert. Fit exactly one/tier.
            gib = 384 / 1024**3
            plan = derive_residency(path, gib, gib, hidden=16, intermediate=16, bits=4)
            self.assertEqual(plan["placement"]["layers.0.experts.0"]["tier"], "vram")
            self.assertEqual(plan["placement"]["layers.0.experts.1"]["tier"], "host")
            self.assertEqual(plan["placement"]["layers.0.experts.2"]["tier"], "ssd")
            self.assertEqual(plan["occupancy_bytes"]["vram"], 384)
            self.assertAlmostEqual(plan["calibration_route_fraction"]["vram"], .7)

    def test_quality_gate(self):
        with tempfile.TemporaryDirectory() as td:
            report = Path(td) / "r.md"
            report.write_text("**Final Score**: **100** / 100\n**Quality**: 100 / 100\n"
                              "**Deployability**: **90** / 100\n| Error Rate | 0.0 |\n")
            parsed = parse_report(report)
            self.assertTrue(compare(parsed, {**parsed, "quality": 99.5}, 1, 1, 0)[0])
            self.assertFalse(compare(parsed, {**parsed, "quality": 98}, 1, 1, 0)[0])


if __name__ == "__main__":
    unittest.main()
