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


class RetainedCellsTest(unittest.TestCase):
    def fixture(self, tmp):
        tmp = Path(tmp)
        baseline = serving([100] * 6)
        vllm = serving([80, 80, 80, 80, 80, 120])
        for name in ("dense", "moe"):
            (tmp / f"{name}-base.json").write_text(json.dumps(baseline))
            (tmp / f"{name}-vllm.json").write_text(json.dumps(vllm))
        config = {
            "schema_version": 1,
            "metric": "agg_decode_tps",
            "max_winning_regression": 0.02,
            "concurrency": [1, 2, 4, 8, 16, 32],
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
            "--candidate", f"dense={dense}", "--candidate", f"moe={moe}",
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


if __name__ == "__main__":
    unittest.main()
