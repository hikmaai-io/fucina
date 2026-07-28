#!/usr/bin/env python3
"""Pure-CPU tests for scripts/gemma_qualification.py."""

import argparse
import importlib.util
import json
import pathlib
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODULE = pathlib.Path(__file__).with_name("gemma_qualification.py")
SPEC = importlib.util.spec_from_file_location("gemma_qualification", MODULE)
gq = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gq)


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    reused_tokens = 0

    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path == "/metrics":
            body = json.dumps({
                "prefix_cache": {"reused_tokens": self.reused_tokens},
                "speculation": {"verify_forwards": 3, "drafted": 6,
                                "accepted": 4, "emitted": 7},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self):
        self.assert_path("/v1/completions")
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        self.server.requests.append(request)
        events = [
            {"choices": [{"text": " alpha", "token_id": 11}]},
            {"choices": [{"text": " beta", "token_id": 12}]},
            {"choices": [], "usage": {
                "prompt_tokens": 300, "completion_tokens": 2, "total_tokens": 302,
                "prompt_tokens_details": {"cached_tokens": 256},
            }},
        ]
        body = "".join("data: " + json.dumps(x) + "\n\n" for x in events)
        body += "data: [DONE]\n\n"
        encoded = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def assert_path(self, wanted):
        if self.path != wanted:
            raise AssertionError(f"got path {self.path}, want {wanted}")


class FakeServer:
    def __enter__(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
        self.httpd.requests = []
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.httpd.server_address
        self.url = f"http://{host}:{port}"
        return self

    def __exit__(self, *_args):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)


class SpeculationServer:
    """Tiny streaming server whose counters prove the MTP path did real work."""

    def __init__(self, engaged):
        self.engaged = engaged
        self.count = 0
        self.lock = threading.Lock()

    def __enter__(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                if self.path != "/metrics":
                    self.send_error(404)
                    return
                multiplier = owner.count if owner.engaged else 0
                body = json.dumps({
                    "prefix_cache": {"reused_tokens": 0},
                    "speculation": {"verify_forwards": multiplier,
                                    "drafted": multiplier * 3,
                                    "accepted": multiplier * 2,
                                    "emitted": multiplier * 3},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                self.rfile.read(length)
                with owner.lock:
                    owner.count += 1
                events = [
                    {"choices": [{"text": " one", "token_id": 101}]},
                    {"choices": [{"text": " two", "token_id": 102}]},
                    {"choices": [], "usage": {
                        "prompt_tokens": 50, "completion_tokens": 2, "total_tokens": 52,
                        "prompt_tokens_details": {"cached_tokens": 0}}},
                ]
                encoded = ("".join("data: " + json.dumps(x) + "\n\n" for x in events)
                           + "data: [DONE]\n\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.httpd.server_address
        self.url = f"http://{host}:{port}"
        return self

    def __exit__(self, *_args):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)


class DistributionTests(unittest.TestCase):
    def test_percentiles_retain_raw_values(self):
        got = gq.distribution([4, 1, 3, 2])
        self.assertEqual(got["count"], 4)
        self.assertEqual(got["p50"], 2.5)
        self.assertEqual(got["raw"], [4.0, 1.0, 3.0, 2.0])
        self.assertGreaterEqual(got["p99"], got["p95"])

    def test_usage_invariants(self):
        got = gq.assert_usage_invariants(
            {"prompt_tokens": 512, "completion_tokens": 32, "total_tokens": 544}, 256)
        self.assertEqual(got["new_prefill_tokens"], 256)
        with self.assertRaises(gq.QualificationError):
            gq.assert_usage_invariants(
                {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 13}, 0)
        with self.assertRaises(gq.QualificationError):
            gq.assert_usage_invariants(
                {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}, 11)


class HTTPTests(unittest.TestCase):
    def test_stream_keeps_token_trace_and_physical_cache_accounting(self):
        with FakeServer() as server:
            got = gq.completion_request(server.url, "gemma", "hello", 2)
        self.assertEqual(got["token_trace"], [" alpha", " beta"])
        self.assertEqual(got["token_ids"], [11, 12])
        self.assertEqual(got["usage"], {
            "prompt_tokens": 300,
            "cached_tokens": 256,
            "new_prefill_tokens": 44,
            "completion_tokens": 2,
            "total_tokens": 302,
        })
        self.assertEqual(got["cached_tokens_source"], "usage.prompt_tokens_details")
        self.assertEqual(len(server.httpd.requests), 1)
        self.assertTrue(server.httpd.requests[0]["stream_options"]["include_usage"])
        self.assertEqual(server.httpd.requests[0]["temperature"], 0.0)


class MTPTests(unittest.TestCase):
    def test_probe_requires_activity_and_token_equality(self):
        with tempfile.TemporaryDirectory() as directory, \
                SpeculationServer(True) as mtp, SpeculationServer(False) as plain:
            out = pathlib.Path(directory) / "mtp.json"
            args = argparse.Namespace(mtp_url=mtp.url, plain_url=plain.url,
                                      model="gemma", batch=2, max_tokens=32,
                                      out=str(out), allow_text_fallback=False)
            self.assertEqual(gq.mtp_probe(args), 0)
            evidence = json.loads(out.read_text())
        self.assertTrue(evidence["gate"]["mtp_engaged"])
        self.assertTrue(evidence["gate"]["token_trace_equal"])
        self.assertTrue(evidence["gate"]["plain_decode_verified"])
        self.assertGreater(evidence["mtp"]["speculation_delta"]["accepted"], 0)
        self.assertEqual(evidence["plain"]["speculation_delta"]["accepted"], 0)


class OracleTests(unittest.TestCase):
    def artifacts(self):
        return ({"first_token_logits": [0.0, 1.0, -2.0],
                 "greedy_token_ids": list(range(40))},
                {"first_token_logits": [0.0001, 1.0001, -2.0001],
                 "greedy_token_ids": list(range(40))})

    def test_logits_and_32_tokens_pass(self):
        ref, got = self.artifacts()
        result = gq.compare_oracles(ref, got, atol=0.001, rtol=0)
        self.assertTrue(result["passed"])
        self.assertEqual(result["logits"]["width"], 3)
        self.assertEqual(len(result["greedy_32"]["candidate"]), 32)

    def test_token_mismatch_fails(self):
        ref, got = self.artifacts()
        got["greedy_token_ids"][31] = 999
        self.assertFalse(gq.compare_oracles(ref, got, 0.001, 0)["passed"])

    def test_nan_is_rejected(self):
        ref, got = self.artifacts()
        got["first_token_logits"][0] = float("nan")
        with self.assertRaises(gq.QualificationError):
            gq.compare_oracles(ref, got, 0.001, 0)


class ConfigTests(unittest.TestCase):
    @staticmethod
    def config():
        subjects = []
        for size, quant in sorted(gq.MATRIX):
            subjects.append({
                "id": f"gemma-{size}-{quant.lower()}",
                "size_b": size,
                "quant": quant,
                "artifact": "/model",
                "engines": {
                    "fucina": {"command": ["fucina"], "base_url": "http://fucina"},
                    "vllm": {"command": ["vllm"], "base_url": "http://vllm"},
                },
            })
        return {"independent_starts": 3, "subjects": subjects}

    def test_full_six_cell_matrix(self):
        gq.validate_config(self.config())

    def test_missing_cell_is_rejected(self):
        config = self.config()
        config["subjects"].pop()
        with self.assertRaises(gq.QualificationError):
            gq.validate_config(config)

    def test_requires_three_independent_starts(self):
        config = self.config()
        config["independent_starts"] = 2
        with self.assertRaises(gq.QualificationError):
            gq.validate_config(config)

    def test_cli_oracle_emits_archive(self):
        ref, got = OracleTests().artifacts()
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "ref.json").write_text(json.dumps(ref))
            (root / "got.json").write_text(json.dumps(got))
            rc = gq.main(["oracle", "--reference", str(root / "ref.json"),
                          "--candidate", str(root / "got.json"),
                          "--out", str(root / "out.json")])
            self.assertEqual(rc, 0)
            archive = json.loads((root / "out.json").read_text())
            self.assertEqual(archive["schema_version"], gq.SCHEMA_VERSION)
            self.assertTrue(archive["passed"])


if __name__ == "__main__":
    unittest.main()
