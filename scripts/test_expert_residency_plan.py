#!/usr/bin/env python3
# ABOUTME: Adversarial and golden tests for DS4 expert-profile replay and residency planning.
# ABOUTME: Keeps malformed telemetry from becoming an unbounded or nondeterministic preload policy.
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from expert_residency_plan import build_plan, build_report, load_profile, write_json_atomic


U64_MAX = (1 << 64) - 1


def tiny_profile() -> dict:
    trace = [
        {"layer": 0, "experts": [0, 1]},
        {"layer": 1, "experts": [0]},
        {"layer": 0, "experts": [1, 2]},
        {"layer": 1, "experts": [0, 2]},
        {"layer": 0, "experts": [0]},
    ]
    selected_rows = [[4, 7, 2, 0], [6, 0, 3, 0]]
    selection_events = [[2, 2, 1, 0], [2, 0, 1, 0]]
    layers = []
    for layer in range(2):
        layers.append({
            "layer": layer,
            "event_count": 3 if layer == 0 else 2,
            "active_expert_uniqueness": 3 if layer == 0 else 2,
            "adjacent_intersection_count": 1,
            "adjacent_union_count": 6 if layer == 0 else 2,
            "experts": [
                {"expert": expert,
                 "selection_events": selection_events[layer][expert],
                 "selected_rows": selected_rows[layer][expert]}
                for expert in range(4)
            ],
        })
    return {
        "format": "fucina-expert-profile-v1",
        "geometry": {"layers": 2, "experts": 4},
        "configured_slots": 2,
        "max_events": 100,
        "events_recorded": len(trace),
        "events_dropped": 0,
        "layers": layers,
        "streamer": {
            "cache_hits": 11,
            "cache_misses": 13,
            "ssd_reads": 52,
            "ssd_bytes": 123456,
            "checksum_failures": 0,
            "prefetch_advice": 20,
        },
        "trace": trace,
    }


class ExpertResidencyPlanTests(unittest.TestCase):
    def load(self, value: dict) -> dict:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "profile.json"
            path.write_text(json.dumps(value))
            return load_profile(path)

    def test_golden_global_lru_curves(self):
        profile = self.load(tiny_profile())
        report = build_report(profile, [1, 2, 3])
        curves = {row["capacity"]: row for row in report["capacity_curves"]}
        self.assertEqual((curves[1]["hits"], curves[1]["misses"]), (0, 8))
        self.assertEqual(curves[1]["fallback_events"], 3)
        self.assertEqual((curves[2]["hits"], curves[2]["misses"]), (1, 7))
        self.assertEqual((curves[3]["hits"], curves[3]["misses"]), (2, 6))

    def test_plan_is_loader_compatible_and_ranked_stably(self):
        profile = self.load(tiny_profile())
        plan = build_plan(profile, slots=3, source="profile.json", source_sha256="00" * 32)
        self.assertEqual(plan["format"], "fucina-expert-residency-v1")
        self.assertEqual(len(plan["placement"]), 8)
        hot = [key for key, value in plan["placement"].items() if value["tier"] == "vram"]
        # selection-events, then selected-rows, then (layer, expert).
        self.assertEqual(hot, ["layers.0.experts.1", "layers.1.experts.0", "layers.0.experts.0"])
        for key, value in plan["placement"].items():
            self.assertRegex(key, r"^layers\.\d+\.experts\.\d+$")
            self.assertIn(value["tier"], ("vram", "ssd"))
            self.assertIsInstance(value["importance"], int)

    def test_repeated_outputs_are_byte_identical(self):
        profile = self.load(tiny_profile())
        plan = build_plan(profile, slots=3, source="same.json", source_sha256="ab" * 32)
        report = build_report(profile, [1, 2, 3])
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td) / "a.json", Path(td) / "b.json"
            write_json_atomic(a, plan)
            first = a.read_bytes()
            write_json_atomic(a, plan)
            write_json_atomic(b, plan)
            self.assertEqual(first, a.read_bytes())
            self.assertEqual(first, b.read_bytes())
            write_json_atomic(a, report)
            report_first = a.read_bytes()
            write_json_atomic(a, report)
            self.assertEqual(report_first, a.read_bytes())

    def test_rejects_adversarial_profiles(self):
        cases = []

        def changed(mutator):
            value = copy.deepcopy(tiny_profile())
            mutator(value)
            cases.append(value)

        changed(lambda p: p.__setitem__("format", "wrong"))
        changed(lambda p: p["geometry"].__setitem__("layers", -1))
        changed(lambda p: p["geometry"].__setitem__("experts", 10**100))
        changed(lambda p: p["trace"][0]["experts"].__setitem__(0, 4))
        changed(lambda p: p["trace"][0].__setitem__("experts", [1, 0]))
        changed(lambda p: p["trace"][0].__setitem__("experts", [0, 0]))
        changed(lambda p: p["streamer"].__setitem__("ssd_bytes", U64_MAX + 1))
        changed(lambda p: p.__setitem__("events_recorded", len(p["trace"]) + 1))
        changed(lambda p: p["layers"][0]["experts"][0].__setitem__("selected_rows", -1))
        changed(lambda p: p["layers"][0].__setitem__("event_count", 1))
        changed(lambda p: p["trace"][0].__setitem__("layer", True))

        for i, value in enumerate(cases):
            with self.subTest(case=i), tempfile.TemporaryDirectory() as td:
                path = Path(td) / "bad.json"
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    load_profile(path)

    def test_rejects_duplicate_json_keys_and_oversized_input(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.json"
            path.write_text('{"format":"fucina-expert-profile-v1","format":"x"}')
            with self.assertRaises(ValueError):
                load_profile(path)
            with path.open("wb") as out:
                out.truncate(64 * 1024 * 1024 + 1)
            with self.assertRaises(ValueError):
                load_profile(path)

    def test_rejects_invalid_cli_configuration(self):
        profile = self.load(tiny_profile())
        for slots in (-1, 0, 9, True):
            with self.subTest(slots=slots), self.assertRaises(ValueError):
                build_plan(profile, slots=slots, source="x", source_sha256="00" * 32)
        for capacities in ([], [0], [1, 1], [4097], [True]):
            with self.subTest(capacities=capacities), self.assertRaises(ValueError):
                build_report(profile, capacities)


if __name__ == "__main__":
    unittest.main()
