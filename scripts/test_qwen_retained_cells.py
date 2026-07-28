import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qwen_retained_cells.py"


def serving(values):
    return {"concurrency": [
        {"N": n, "agg_decode_tps": value, "median_ttft_ms": 10+n, "p95_ttft_ms": 12+n}
        for n, value in zip((1, 2, 4, 8, 16, 32), values)
    ]}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RetainedCellsTest(unittest.TestCase):
    def fixture(self, tmp):
        tmp = Path(tmp)
        baseline = serving([100] * 6)
        vllm = serving([80, 80, 80, 80, 80, 120])
        for name in ("dense", "moe"):
            (tmp / f"{name}-base.json").write_text(json.dumps(baseline))
            (tmp / f"{name}-vllm.json").write_text(json.dumps(vllm))

        summary = tmp / "candidate-summary.json"
        summary.write_text("{}\n")
        commit = "a" * 40
        model_hashes = {"dense": "b" * 64, "moe": "c" * 64}
        qualification_roots = {}
        for name in ("dense", "moe"):
            qualification = tmp / "qualification" / name
            qualification.mkdir(parents=True)
            (qualification / "manifest.json").write_text(json.dumps({
                "source_commit": commit,
                "independent_server_starts": 3,
                "model_fingerprints": [{"sha256": model_hashes[name]}],
            }))
            qualification_roots[name] = f"qualification/{name}"

        provenance = {
            "schema_version": 1,
            "record_type": "qwen_retained_baseline_refreeze",
            "gate_owner_countersign": {
                "status": "COUNTERSIGNED", "date": "2026-07-28", "decision": "test"
            },
            "source_commits": {
                "length_contract": commit, "evidence": "d" * 40, "merged_main": "e" * 40
            },
            "candidate_summary": "candidate-summary.json",
            "qualification_roots": qualification_roots,
            "generation_contract": {
                "max_tokens": 128, "min_tokens": 128, "ignore_eos": True,
                "completion_tokens_per_request": 128, "decode_intervals_per_request": 127,
                "concurrency": [1, 2, 4, 8, 16, 32], "diverse_prompts": True,
            },
            "aggregation": {"statistic": "cellwise_median", "independent_server_starts": 3},
            "hashes": {
                "source_config_sha256": "1" * 64,
                "bench_serving_py_sha256": "2" * 64,
                "prompt_pool_sha256": "3" * 64,
                "short_prompt_sha256": "4" * 64,
                "candidate_summary_sha256": sha256(summary),
                "model_sha256": model_hashes,
                "baseline_sha256": {
                    name: sha256(tmp / f"{name}-base.json") for name in ("dense", "moe")
                },
            },
            "baselines": {name: f"{name}-base.json" for name in ("dense", "moe")},
        }
        provenance_path = tmp / "benchmark-evidence/provenance.json"
        provenance_path.parent.mkdir()
        provenance_path.write_text(json.dumps(provenance))

        config = {
            "schema_version": 2,
            "metric": "agg_decode_tps",
            "max_winning_regression": 0.02,
            "concurrency": [1, 2, 4, 8, 16, 32],
            "bench": {
                "max_tokens": 128, "min_tokens": 128, "ignore_eos": True,
                "independent_server_starts": 3, "aggregation": "cellwise_median",
            },
            "baseline_provenance": "benchmark-evidence/provenance.json",
            "models": {
                "dense": {"baseline": "dense-base.json", "vllm_reference": "dense-vllm.json", "winning_cells": [1,2,4,8,16], "target_cell": 32},
                "moe": {"baseline": "moe-base.json", "vllm_reference": "moe-vllm.json", "winning_cells": [1,2,4,8,16,32]},
            },
        }
        # check() resolves references two parents above config, mirroring benchmark-evidence/configs.
        cfg_dir = tmp / "benchmark-evidence/configs"
        cfg_dir.mkdir(parents=True)
        path = cfg_dir / "config.json"
        path.write_text(json.dumps(config))
        return path, tmp

    def run_check(self, config, dense, moe, out):
        return subprocess.run([
            "python3", str(SCRIPT), "check", "--config", str(config),
            "--candidate", f"dense={dense}", "--candidate", f"dense={dense}",
            "--candidate", f"dense={dense}", "--candidate", f"moe={moe}",
            "--candidate", f"moe={moe}", "--candidate", f"moe={moe}",
            "--json-out", str(out),
        ], text=True, capture_output=True)

    def test_all_eleven_winning_cells_pass_at_two_percent_floor(self):
        with tempfile.TemporaryDirectory() as td:
            config, root = self.fixture(td)
            dense = root / "dense-candidate.json"
            moe = root / "moe-candidate.json"
            dense.write_text(json.dumps(serving([98, 99, 100, 101, 102, 70])))
            moe.write_text(json.dumps(serving([98, 99, 100, 101, 102, 98])))
            result = self.run_check(config, dense, moe, root / "summary.json")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("GATE PASS", result.stdout)

    def test_winning_cell_below_floor_fails_but_dense32_is_target_only(self):
        with tempfile.TemporaryDirectory() as td:
            config, root = self.fixture(td)
            dense = root / "dense.json"
            moe = root / "moe.json"
            dense.write_text(json.dumps(serving([100, 100, 100, 100, 100, 1])))
            moe.write_text(json.dumps(serving([100, 100, 97.9, 100, 100, 100])))
            result = self.run_check(config, dense, moe, root / "summary.json")
            self.assertEqual(result.returncode, 1)
            self.assertIn("moe N=4", result.stderr)
            self.assertNotIn("dense N=32", result.stderr)

    def test_three_runs_use_cellwise_median(self):
        with tempfile.TemporaryDirectory() as td:
            config, root = self.fixture(td)
            paths = []
            for i, value in enumerate((70, 100, 110)):
                path = root / f"run-{i}.json"
                path.write_text(json.dumps([serving([value] * 6)]))
                paths.append(path)
            cmd = ["python3", str(SCRIPT), "check", "--config", str(config)]
            for name in ("dense", "moe"):
                for path in paths:
                    cmd += ["--candidate", f"{name}={path}"]
            result = subprocess.run(cmd, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_gate_requires_exactly_three_candidate_starts(self):
        with tempfile.TemporaryDirectory() as td:
            config, root = self.fixture(td)
            candidate = root / "candidate.json"
            candidate.write_text(json.dumps(serving([100] * 6)))
            result = subprocess.run([
                "python3", str(SCRIPT), "check", "--config", str(config),
                "--candidate", f"dense={candidate}", "--candidate", f"moe={candidate}",
            ], text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("expected exactly 3", result.stderr)

    def test_missing_contract_or_provenance_fails_closed(self):
        for missing in ("min_tokens", "baseline_provenance"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as td:
                config, _ = self.fixture(td)
                value = json.loads(config.read_text())
                if missing == "min_tokens":
                    del value["bench"][missing]
                else:
                    del value[missing]
                config.write_text(json.dumps(value))
                result = subprocess.run([
                    "python3", str(SCRIPT), "validate", "--config", str(config),
                ], text=True, capture_output=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn(missing.replace("_", " "), result.stderr.replace("_", " "))

    def test_canonical_baselines_are_exact_candidate_medians(self):
        config_path = ROOT / "benchmark-evidence/configs/qwen-retained-12.json"
        config = json.loads(config_path.read_text())
        summary = json.loads((
            ROOT / "benchmark-evidence/results/2026-07-28-integration-p1-qwen/retained-gate-summary.json"
        ).read_text())
        for name in ("dense", "moe"):
            baseline = json.loads((ROOT / config["models"][name]["baseline"]).read_text())
            actual = {cell["N"]: cell["agg_decode_tps"] for cell in baseline["concurrency"]}
            expected = {cell["N"]: cell["candidate"] for cell in summary["models"][name]["cells"]}
            self.assertEqual(actual, expected)
            self.assertNotEqual(config["models"][name]["baseline"],
                                config["models"][name]["vllm_reference"])


if __name__ == "__main__":
    unittest.main()
