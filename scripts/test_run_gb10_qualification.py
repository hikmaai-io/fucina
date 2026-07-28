#!/usr/bin/env python3
import json
import os
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path


class QualificationRunnerTest(unittest.TestCase):
    def test_archives_three_independent_server_starts(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "evidence"
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text('{"model":"fixture"}\n')
            starts = root / "starts.txt"
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]

            server = root / "server.sh"
            server.write_text(
                "#!/usr/bin/env bash\n"
                'echo "$FUCINA_RUN_INDEX" >> "$STARTS_FILE"\n'
                'exec python3 -m http.server "$PORT" --bind 127.0.0.1\n'
            )
            server.chmod(0o755)
            benchmark = root / "benchmark.sh"
            benchmark.write_text(
                "#!/usr/bin/env bash\n"
                "python3 - <<'PY'\n"
                "import json, os\n"
                "with open(os.environ['FUCINA_RAW_RESULTS'], 'w') as f:\n"
                "    json.dump([{'run': int(os.environ['FUCINA_RUN_INDEX']), 'ttft_ms': [1.0, 2.0]}], f)\n"
                "PY\n"
            )
            benchmark.chmod(0o755)

            env = os.environ.copy()
            env.update({
                "EVIDENCE_ROOT": str(evidence),
                "STARTS_FILE": str(starts),
                "PORT": str(port),
            })
            result = subprocess.run([
                str(repo / "scripts/run_gb10_qualification.sh"),
                "--claim", "smoke",
                "--model", str(model),
                "--server-command", str(server),
                "--benchmark-command", str(benchmark),
                "--ready-url", f"http://127.0.0.1:{port}",
                "--ready-timeout", "10",
                "--allow-dirty",
            ], cwd=repo, env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            run_roots = list(evidence.iterdir())
            self.assertEqual(len(run_roots), 1)
            run_root = run_roots[0]
            manifest = json.loads((run_root / "manifest.json").read_text())
            self.assertEqual(manifest["independent_server_starts"], 3)
            self.assertEqual(len(manifest["model_fingerprints"]), 1)
            self.assertEqual(starts.read_text().splitlines(), ["1", "2", "3"])
            for index in range(1, 4):
                raw = json.loads((run_root / f"run-{index}/raw-results.json").read_text())
                self.assertIsInstance(raw, list)
                self.assertEqual(raw[0]["run"], index)
                self.assertTrue((run_root / f"run-{index}/server.pid").is_file())


if __name__ == "__main__":
    unittest.main()
