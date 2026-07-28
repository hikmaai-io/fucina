#!/usr/bin/env python3
"""Gemma-4 qualification harness for Fucina and equivalent vLLM artifacts.

The harness is dependency-free and deliberately talks only to the common OpenAI
``/v1/completions`` streaming API.  It retains raw event timings, usage, cache,
startup, memory, and speculation counters in an archival JSON document.

Subcommands:
  run         launch the matrix described by a JSON config and benchmark it
  mtp-probe   prove batched MTP activation and compare per-token SSE traces
  oracle      compare first-token logits and 32 greedy token ids
  validate    validate a matrix config without loading a model
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import platform
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence

SCHEMA_VERSION = "fucina.gemma-qualification.v1"
MATRIX = {(size, quant) for size in (12, 31) for quant in ("Q4_0-QAT", "Q8_0", "NVFP4")}
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
         "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
         "xray yankee zulu").split()


class QualificationError(RuntimeError):
    """A release-gate or harness-contract failure."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile; raw arrays remain authoritative."""
    if not values:
        return None
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def distribution(values: Iterable[float]) -> Dict[str, Any]:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return {
        "count": len(xs),
        "p50": percentile(xs, 0.50),
        "p95": percentile(xs, 0.95),
        "p99": percentile(xs, 0.99),
        "raw": xs,
    }


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def artifact_fingerprint(path: str, supplied: str = "") -> Dict[str, Any]:
    """Fingerprint a file or a directory manifest.

    Full model-directory hashing can add minutes and duplicate model loading.  A
    precomputed, trusted SHA-256 may therefore be supplied in the matrix.  Without
    one, files are fully hashed; directories hash every regular file's relative
    path, size, and SHA-256.
    """
    p = pathlib.Path(path)
    if supplied:
        return {"algorithm": "sha256", "value": supplied, "source": "configured"}
    if not p.exists():
        raise QualificationError(f"artifact does not exist: {path}")
    if p.is_file():
        return {"algorithm": "sha256", "value": sha256_file(p), "source": "computed-full"}
    entries = []
    for child in sorted(x for x in p.rglob("*") if x.is_file()):
        entries.append({"path": str(child.relative_to(p)), "size": child.stat().st_size,
                        "sha256": sha256_file(child)})
    return {"algorithm": "sha256", "value": sha256_json(entries),
            "source": "computed-directory", "files": len(entries)}


def http_json(url: str, payload: Any = None, timeout: float = 30) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def optional_metrics(base_url: str) -> Optional[Dict[str, Any]]:
    try:
        value = http_json(base_url.rstrip("/") + "/metrics", timeout=5)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def nested_number(value: Optional[Dict[str, Any]], *keys: str) -> Optional[float]:
    cur: Any = value
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return float(cur) if isinstance(cur, (int, float)) else None


def cached_from_usage(usage: Dict[str, Any]) -> Optional[int]:
    details = usage.get("prompt_tokens_details") or {}
    value = details.get("cached_tokens")
    return int(value) if isinstance(value, (int, float)) else None


def assert_usage_invariants(usage: Dict[str, Any], cached_tokens: int) -> Dict[str, int]:
    """Assert the shared accounting contract on every measured response."""
    prompt = int(usage.get("prompt_tokens", -1))
    completion = int(usage.get("completion_tokens", -1))
    total = int(usage.get("total_tokens", -1))
    if prompt < 0 or completion < 0:
        raise QualificationError(f"missing/negative usage counters: {usage!r}")
    if total != prompt + completion:
        raise QualificationError(
            f"usage total invariant failed: {total} != {prompt}+{completion}")
    if not 0 <= cached_tokens <= prompt:
        raise QualificationError(
            f"cached-token invariant failed: 0 <= {cached_tokens} <= {prompt}")
    return {"prompt_tokens": prompt, "cached_tokens": cached_tokens,
            "new_prefill_tokens": prompt - cached_tokens,
            "completion_tokens": completion, "total_tokens": total}


def metric_cached_delta(before: Optional[Dict[str, Any]],
                        after: Optional[Dict[str, Any]]) -> Optional[int]:
    a = nested_number(before, "prefix_cache", "reused_tokens")
    b = nested_number(after, "prefix_cache", "reused_tokens")
    if a is None or b is None:
        return None
    return max(0, int(b - a))


def completion_request(base_url: str, model: str, prompt: str, max_tokens: int,
                       ignore_eos: bool = True, timeout: float = 600) -> Dict[str, Any]:
    """Issue one streaming greedy request and retain per-token SSE event times.

    Both Fucina and vLLM emit one text piece per committed token.  ``token_trace``
    keeps those boundaries for the MTP losslessness gate; if an implementation
    includes token ids in an SSE extension they are retained too.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload.update({"ignore_eos": True, "min_tokens": max_tokens})
    metrics_before = optional_metrics(base_url)
    data = json.dumps(payload).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    event_offsets: List[float] = []
    pieces: List[str] = []
    token_ids: List[int] = []
    usage: Optional[Dict[str, Any]] = None
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                event = json.loads(body)
            except json.JSONDecodeError:
                continue
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            piece = choice.get("text")
            if piece:
                event_offsets.append(time.perf_counter() - started)
                pieces.append(piece)
                ids = choice.get("token_ids")
                if isinstance(ids, list):
                    token_ids.extend(int(x) for x in ids)
                elif isinstance(choice.get("token_id"), int):
                    token_ids.append(int(choice["token_id"]))
    wall = time.perf_counter() - started
    metrics_after = optional_metrics(base_url)
    if usage is None:
        raise QualificationError("stream ended without final usage; include_usage contract failed")
    cached = cached_from_usage(usage)
    cache_source = "usage.prompt_tokens_details"
    if cached is None:
        cached = metric_cached_delta(metrics_before, metrics_after)
        cache_source = "metrics.prefix_cache.reused_tokens"
    if cached is None:
        # No compatible cache telemetry exists (common on an uninstrumented
        # comparison server). Cold is conservatively zero, never a logical LCP.
        cached, cache_source = 0, "unreported-assumed-cold"
    accounting = assert_usage_invariants(usage, cached)
    completion_tokens = accounting["completion_tokens"]
    itls = [event_offsets[i] - event_offsets[i - 1] for i in range(1, len(event_offsets))]
    decode_window = ((event_offsets[-1] - event_offsets[0]) if len(event_offsets) > 1 else 0.0)
    per_stream = ((completion_tokens - 1) / decode_window
                  if completion_tokens > 1 and decode_window > 0 else None)
    return {
        "ttft_s": event_offsets[0] if event_offsets else None,
        "itl_s": itls,
        "event_offsets_s": event_offsets,
        "wall_s": wall,
        "per_stream_decode_tok_s": per_stream,
        "token_trace": pieces,
        "token_ids": token_ids,
        "text": "".join(pieces),
        "usage": accounting,
        "cached_tokens_source": cache_source,
    }


def prompt_words(count: int, nonce: str) -> str:
    words = [WORDS[i % len(WORDS)] for i in range(count)]
    return (f"Qualification nonce {nonce}. " + " ".join(words) +
            "\nContinue with a concise technical explanation:")


def cold_warm_probe(base_url: str, model: str, reps: int, prefix_words: int,
                    max_tokens: int) -> Dict[str, Any]:
    cold, warm = [], []
    for i in range(reps):
        nonce = f"{uuid.uuid4().hex}-{i}"
        base = prompt_words(prefix_words, nonce)
        first = completion_request(base_url, model, base, max_tokens)
        # Exact extension of the completed first request. Only execution-skipped
        # tokens reported by the server/metrics count as cached.
        second_prompt = base + first["text"] + "\nNow state one additional consequence:"
        second = completion_request(base_url, model, second_prompt, max_tokens)
        cold.append(first)
        warm.append(second)
    return {"cold": cold, "warm_prefix": warm,
            "cold_ttft_s": distribution(x["ttft_s"] for x in cold),
            "warm_ttft_s": distribution(x["ttft_s"] for x in warm)}


def burst_probe(base_url: str, model: str, concurrency: int, reps: int,
                max_tokens: int) -> Dict[str, Any]:
    bursts = []
    for rep in range(reps):
        prompts = [prompt_words(64, f"burst-{rep}-{i}-{uuid.uuid4().hex}")
                   for i in range(concurrency)]
        barrier = threading.Barrier(concurrency)

        def worker(prompt: str) -> Dict[str, Any]:
            barrier.wait()
            return completion_request(base_url, model, prompt, max_tokens)

        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(pool.map(worker, prompts))
        wall = time.perf_counter() - started
        generated = sum(row["usage"]["completion_tokens"] for row in rows)
        bursts.append({"wall_s": wall, "completion_tokens": generated,
                       "aggregate_tok_s": generated / wall if wall > 0 else None,
                       "streams": rows})
    all_streams = [row for burst in bursts for row in burst["streams"]]
    return {
        "concurrency": concurrency,
        "bursts": bursts,
        "ttft_s": distribution(x["ttft_s"] for x in all_streams),
        "itl_s": distribution(v for x in all_streams for v in x["itl_s"]),
        "aggregate_tok_s": distribution(x["aggregate_tok_s"] for x in bursts),
        "per_stream_decode_tok_s": distribution(
            x["per_stream_decode_tok_s"] for x in all_streams),
    }


def read_proc_kb(pid: int, field: str) -> int:
    try:
        for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def descendants(root_pid: int) -> List[int]:
    parent: Dict[int, int] = {}
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split()
            parent[int(entry.name)] = int(fields[3])
        except (OSError, ValueError, IndexError):
            continue
    out, frontier = {root_pid}, [root_pid]
    while frontier:
        p = frontier.pop()
        children = [pid for pid, ppid in parent.items() if ppid == p and pid not in out]
        out.update(children)
        frontier.extend(children)
    return sorted(out)


def gpu_memory_mib(pids: Sequence[int]) -> Optional[float]:
    try:
        raw = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL,
            timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    wanted = set(pids)
    total = 0.0
    found = False
    for line in raw.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) in wanted:
            try:
                total += float(parts[1])
                found = True
            except ValueError:
                pass
    return total if found else None


class MemorySampler:
    def __init__(self, pid: int, interval: float = 0.2):
        self.pid, self.interval = pid, interval
        self.samples: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        t0 = time.perf_counter()
        while not self._stop.is_set():
            pids = descendants(self.pid)
            self.samples.append({
                "offset_s": time.perf_counter() - t0,
                "rss_mib": sum(read_proc_kb(p, "VmRSS") for p in pids) / 1024,
                "hwm_mib": sum(read_proc_kb(p, "VmHWM") for p in pids) / 1024,
                "gpu_mib": gpu_memory_mib(pids),
                "pids": pids,
            })
            self._stop.wait(self.interval)

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=2)
        gpu = [x["gpu_mib"] for x in self.samples if x["gpu_mib"] is not None]
        return {"peak_rss_mib": max((x["rss_mib"] for x in self.samples), default=0),
                "peak_hwm_mib": max((x["hwm_mib"] for x in self.samples), default=0),
                "peak_gpu_mib": max(gpu) if gpu else None,
                "samples": self.samples}


class ManagedServer:
    def __init__(self, spec: Dict[str, Any], substitutions: Dict[str, Any], log_path: pathlib.Path):
        self.spec, self.substitutions, self.log_path = spec, substitutions, log_path
        self.proc: Optional[subprocess.Popen] = None
        self.log = None
        self.memory: Optional[MemorySampler] = None
        self.startup_s: Optional[float] = None

    @property
    def base_url(self) -> str:
        return str(self.spec["base_url"]).format(**self.substitutions).rstrip("/")

    def start(self) -> None:
        command = [str(x).format(**self.substitutions) for x in self.spec["command"]]
        env = os.environ.copy()
        env.update({str(k): str(v).format(**self.substitutions)
                    for k, v in self.spec.get("env", {}).items()})
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("wb")
        started = time.perf_counter()
        self.proc = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
        self.memory = MemorySampler(self.proc.pid)
        self.memory.start()
        deadline = started + float(self.spec.get("startup_timeout_s", 900))
        ready_path = self.spec.get("ready_path", "/readyz")
        last = "not ready"
        while time.perf_counter() < deadline:
            if self.proc.poll() is not None:
                raise QualificationError(
                    f"server exited {self.proc.returncode} before ready; see {self.log_path}")
            try:
                with urllib.request.urlopen(self.base_url + ready_path, timeout=2) as r:
                    if 200 <= r.status < 300:
                        self.startup_s = time.perf_counter() - started
                        return
            except Exception as exc:
                last = str(exc)
            time.sleep(0.2)
        raise QualificationError(f"readiness timeout for {self.base_url}: {last}")

    def stop(self) -> Dict[str, Any]:
        memory = self.memory.stop() if self.memory else {"samples": []}
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGINT)
                self.proc.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except OSError:
                    pass
                self.proc.wait(timeout=10)
        if self.log:
            self.log.close()
        return memory


def validate_config(config: Dict[str, Any], require_full_matrix: bool = True) -> None:
    if int(config.get("independent_starts", 0)) < 3:
        raise QualificationError("independent_starts must be >= 3")
    subjects = config.get("subjects")
    if not isinstance(subjects, list) or not subjects:
        raise QualificationError("subjects must be a non-empty list")
    seen = set()
    for subject in subjects:
        key = (int(subject.get("size_b", 0)), subject.get("quant"))
        if key not in MATRIX:
            raise QualificationError(f"unsupported Gemma matrix cell: {key}")
        if key in seen:
            raise QualificationError(f"duplicate Gemma matrix cell: {key}")
        seen.add(key)
        if not subject.get("artifact"):
            raise QualificationError(f"{key}: artifact is required")
        for engine_name in ("fucina", "vllm"):
            spec = subject.get("engines", {}).get(engine_name)
            if not spec or not isinstance(spec.get("command"), list) or not spec.get("base_url"):
                raise QualificationError(f"{key}: engines.{engine_name} command/base_url required")
    if require_full_matrix and seen != MATRIX:
        missing = sorted(MATRIX - seen)
        raise QualificationError(f"matrix is incomplete; missing {missing}")


def command_output(command: Sequence[str], substitutions: Dict[str, Any], cwd: str = "") -> Any:
    argv = [str(x).format(**substitutions) for x in command]
    raw = subprocess.check_output(argv, text=True, cwd=(cwd or None), timeout=3600)
    return json.loads(raw)


def compare_oracles(reference: Dict[str, Any], candidate: Dict[str, Any],
                    atol: float, rtol: float) -> Dict[str, Any]:
    ref_logits = reference.get("first_token_logits")
    got_logits = candidate.get("first_token_logits")
    ref_tokens = reference.get("greedy_token_ids")
    got_tokens = candidate.get("greedy_token_ids")
    if not isinstance(ref_logits, list) or not isinstance(got_logits, list):
        raise QualificationError("oracle artifacts require first_token_logits arrays")
    if len(ref_logits) != len(got_logits) or not ref_logits:
        raise QualificationError(
            f"logit width mismatch: reference={len(ref_logits)} candidate={len(got_logits)}")
    if not isinstance(ref_tokens, list) or not isinstance(got_tokens, list):
        raise QualificationError("oracle artifacts require greedy_token_ids arrays")
    if len(ref_tokens) < 32 or len(got_tokens) < 32:
        raise QualificationError("oracle artifacts must contain at least 32 greedy token ids")
    max_abs, max_rel, mismatches = 0.0, 0.0, 0
    for ref, got in zip(ref_logits, got_logits):
        ref, got = float(ref), float(got)
        if not math.isfinite(ref) or not math.isfinite(got):
            raise QualificationError("oracle logits contain NaN/Inf")
        absolute = abs(got - ref)
        relative = absolute / max(abs(ref), 1e-30)
        max_abs, max_rel = max(max_abs, absolute), max(max_rel, relative)
        if absolute > atol + rtol * abs(ref):
            mismatches += 1
    exact_tokens = [int(x) for x in ref_tokens[:32]] == [int(x) for x in got_tokens[:32]]
    result = {"logits": {"width": len(ref_logits), "atol": atol, "rtol": rtol,
                          "max_abs_error": max_abs, "max_rel_error": max_rel,
                          "mismatch_count": mismatches, "passed": mismatches == 0},
              "greedy_32": {"reference": ref_tokens[:32], "candidate": got_tokens[:32],
                            "passed": exact_tokens},
              "passed": mismatches == 0 and exact_tokens}
    return result


def oracle_hook(subject: Dict[str, Any], substitutions: Dict[str, Any],
                raw_dir: pathlib.Path) -> Optional[Dict[str, Any]]:
    spec = subject.get("oracle")
    if not spec:
        return None
    reference_path = pathlib.Path(str(spec["reference"]).format(**substitutions))
    reference = json.loads(reference_path.read_text())
    if spec.get("candidate"):
        candidate_path = pathlib.Path(str(spec["candidate"]).format(**substitutions))
        candidate = json.loads(candidate_path.read_text())
    elif spec.get("candidate_command"):
        candidate = command_output(spec["candidate_command"], substitutions, spec.get("cwd", ""))
        candidate_path = raw_dir / "oracle-candidate.json"
        candidate_path.write_text(json.dumps(candidate, indent=2) + "\n")
    else:
        raise QualificationError("oracle needs candidate or candidate_command")
    result = compare_oracles(reference, candidate, float(spec.get("atol", 1e-3)),
                             float(spec.get("rtol", 1e-3)))
    result.update({"reference": str(reference_path), "reference_sha256": sha256_file(reference_path),
                   "candidate": str(candidate_path), "candidate_sha256": sha256_file(candidate_path)})
    if not result["passed"]:
        raise QualificationError(f"oracle gate failed for {subject['id']}: {result}")
    return result


def summarize_starts(starts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool raw arrays across independent starts while retaining each start verbatim."""
    summary: Dict[str, Any] = {
        "independent_starts": len(starts),
        "startup_to_ready_s": distribution(x["startup_to_ready_s"] for x in starts),
        "peak_rss_mib": distribution(x["memory"]["peak_rss_mib"] for x in starts),
        "peak_gpu_mib": distribution(x["memory"]["peak_gpu_mib"] for x in starts),
        "cold_ttft_s": distribution(
            row["ttft_s"] for start in starts for row in start["cold_warm"]["cold"]),
        "warm_prefix_ttft_s": distribution(
            row["ttft_s"] for start in starts for row in start["cold_warm"]["warm_prefix"]),
        "warm_cached_tokens": distribution(
            row["usage"]["cached_tokens"] for start in starts
            for row in start["cold_warm"]["warm_prefix"]),
        "throughput": {},
        "request_failures": 0,
    }
    concurrencies = sorted({probe["concurrency"] for start in starts
                            for probe in start["throughput"]})
    for concurrency in concurrencies:
        probes = [probe for start in starts for probe in start["throughput"]
                  if probe["concurrency"] == concurrency]
        streams = [stream for probe in probes for burst in probe["bursts"]
                   for stream in burst["streams"]]
        summary["throughput"][str(concurrency)] = {
            "ttft_s": distribution(x["ttft_s"] for x in streams),
            "itl_s": distribution(v for x in streams for v in x["itl_s"]),
            "aggregate_tok_s": distribution(
                burst["aggregate_tok_s"] for probe in probes for burst in probe["bursts"]),
            "per_stream_decode_tok_s": distribution(
                x["per_stream_decode_tok_s"] for x in streams),
        }
    return summary


def comparison_summary(engines: Dict[str, Any]) -> Dict[str, Any]:
    """Publish explicit Fucina/vLLM p50 ratios (raw engine arrays remain canonical)."""
    fucina, vllm = engines["fucina"]["summary"], engines["vllm"]["summary"]

    def ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
        return (a / b) if a is not None and b not in (None, 0) else None

    result: Dict[str, Any] = {
        "ratio_definition": "fucina_p50 / vllm_p50",
        "startup_to_ready": ratio(fucina["startup_to_ready_s"]["p50"],
                                  vllm["startup_to_ready_s"]["p50"]),
        "peak_rss": ratio(fucina["peak_rss_mib"]["p50"], vllm["peak_rss_mib"]["p50"]),
        "peak_gpu": ratio(fucina["peak_gpu_mib"]["p50"], vllm["peak_gpu_mib"]["p50"]),
        "cold_ttft": ratio(fucina["cold_ttft_s"]["p50"], vllm["cold_ttft_s"]["p50"]),
        "warm_prefix_ttft": ratio(fucina["warm_prefix_ttft_s"]["p50"],
                                  vllm["warm_prefix_ttft_s"]["p50"]),
        "throughput": {},
    }
    for concurrency in sorted(set(fucina["throughput"]) & set(vllm["throughput"]), key=int):
        f, v = fucina["throughput"][concurrency], vllm["throughput"][concurrency]
        result["throughput"][concurrency] = {
            "ttft": ratio(f["ttft_s"]["p50"], v["ttft_s"]["p50"]),
            "itl": ratio(f["itl_s"]["p50"], v["itl_s"]["p50"]),
            "aggregate_tok_s": ratio(f["aggregate_tok_s"]["p50"],
                                     v["aggregate_tok_s"]["p50"]),
            "per_stream_decode_tok_s": ratio(f["per_stream_decode_tok_s"]["p50"],
                                             v["per_stream_decode_tok_s"]["p50"]),
        }
    return result


def run_subject(config: Dict[str, Any], subject: Dict[str, Any], out_dir: pathlib.Path) -> Dict[str, Any]:
    subject_dir = out_dir / subject["id"]
    raw_dir = subject_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    vllm_artifact = subject.get("vllm_artifact", subject["artifact"])
    substitutions = {"artifact": subject["artifact"], "vllm_artifact": vllm_artifact,
                     "model": subject.get("model", subject["id"]), "subject": subject["id"]}
    record: Dict[str, Any] = {
        "id": subject["id"], "family": "gemma-4", "size_b": subject["size_b"],
        "quant": subject["quant"],
        "artifacts": {
            "fucina": {"path": subject["artifact"], "fingerprint": artifact_fingerprint(
                subject["artifact"], subject.get("artifact_sha256", ""))},
            "vllm": {"path": vllm_artifact, "fingerprint": artifact_fingerprint(
                vllm_artifact, subject.get("vllm_artifact_sha256", ""))},
        },
        "engines": {},
    }
    record["oracle"] = oracle_hook(subject, substitutions, raw_dir)
    workload = config.get("workload", {})
    for engine_name in ("fucina", "vllm"):
        starts = []
        for index in range(1, int(config["independent_starts"]) + 1):
            log_path = raw_dir / f"{engine_name}-start-{index}.log"
            managed = ManagedServer(subject["engines"][engine_name], substitutions, log_path)
            memory: Dict[str, Any] = {}
            try:
                managed.start()
                cold_warm = cold_warm_probe(
                    managed.base_url, substitutions["model"],
                    int(workload.get("cold_warm_reps", 3)),
                    int(workload.get("prefix_words", 2048)),
                    int(workload.get("max_tokens", 64)))
                throughput = [burst_probe(
                    managed.base_url, substitutions["model"], int(n),
                    int(workload.get("burst_reps", 3)),
                    int(workload.get("max_tokens", 64)))
                    for n in workload.get("concurrency", [1, 2, 4, 8])]
                final_metrics = optional_metrics(managed.base_url)
                starts.append({"index": index, "startup_to_ready_s": managed.startup_s,
                               "cold_warm": cold_warm, "throughput": throughput,
                               "final_metrics": final_metrics,
                               "server_log": str(log_path.relative_to(subject_dir))})
            finally:
                memory = managed.stop()
            if starts:
                starts[-1]["memory"] = memory
        record["engines"][engine_name] = {"starts": starts,
                                           "summary": summarize_starts(starts)}
    record["comparison"] = comparison_summary(record["engines"])
    manifest_path = subject_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": SCHEMA_VERSION,
                                         "created_at": utc_now(), "subject": record},
                                        indent=2) + "\n")
    return record


def host_metadata() -> Dict[str, Any]:
    def output(argv: List[str]) -> Optional[str]:
        try:
            return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT,
                                           timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"hostname": platform.node(), "platform": platform.platform(),
            "python": platform.python_version(),
            "nvidia_smi": output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                   "--format=csv,noheader"]),
            "nvcc": output(["nvcc", "--version"])}


def git_metadata() -> Dict[str, Any]:
    def git(*args: str) -> Optional[str]:
        try:
            return subprocess.check_output(["git", *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain"))}


def run_matrix(args: argparse.Namespace) -> int:
    config_path = pathlib.Path(args.config)
    config = json.loads(config_path.read_text())
    validate_config(config, require_full_matrix=not args.allow_partial_matrix)
    selected = [x for x in config["subjects"] if not args.subject or x["id"] in args.subject]
    if not selected:
        raise QualificationError("no subjects selected")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "kind": "gemma-qualification-archive",
        "run_id": str(uuid.uuid4()), "created_at": utc_now(),
        "config": str(config_path), "config_sha256": sha256_file(config_path),
        "source": git_metadata(), "host": host_metadata(), "subjects": [],
    }
    for subject in selected:
        archive["subjects"].append(run_subject(config, subject, out_dir))
        (out_dir / "manifest.json").write_text(json.dumps(archive, indent=2) + "\n")
    print(out_dir / "manifest.json")
    return 0


def speculation_delta(before: Optional[Dict[str, Any]], after: Optional[Dict[str, Any]]) -> Dict[str, int]:
    result = {}
    for key in ("verify_forwards", "drafted", "accepted", "emitted"):
        a, b = nested_number(before, "speculation", key), nested_number(after, "speculation", key)
        result[key] = max(0, int((b or 0) - (a or 0)))
    return result


def trace_key(row: Dict[str, Any]) -> List[Any]:
    return row["token_ids"] if row.get("token_ids") else row["token_trace"]


def mtp_probe(args: argparse.Namespace) -> int:
    prompts = [prompt_words(96, f"mtp-{i}") for i in range(args.batch)]

    def run(base_url: str) -> Dict[str, Any]:
        before = optional_metrics(base_url)
        barrier = threading.Barrier(args.batch)

        def worker(prompt: str) -> Dict[str, Any]:
            barrier.wait()
            return completion_request(base_url, args.model, prompt, args.max_tokens)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch) as pool:
            rows = list(pool.map(worker, prompts))
        after = optional_metrics(base_url)
        return {"rows": rows, "speculation_delta": speculation_delta(before, after)}

    mtp, plain = run(args.mtp_url), run(args.plain_url)
    equality = []
    for i, (a, b) in enumerate(zip(mtp["rows"], plain["rows"])):
        equal = trace_key(a) == trace_key(b)
        equality.append({"stream": i, "equal": equal, "mtp_trace": trace_key(a),
                         "plain_trace": trace_key(b)})
    delta = mtp["speculation_delta"]
    engaged = delta["verify_forwards"] > 0 and delta["drafted"] > 0 and delta["accepted"] > 0
    result = {"schema_version": SCHEMA_VERSION, "kind": "gemma-batch-mtp-probe",
              "created_at": utc_now(), "batch": args.batch, "max_tokens": args.max_tokens,
              "mtp": mtp, "plain": plain,
              "gate": {"mtp_engaged": engaged,
                       "token_trace_equal": all(x["equal"] for x in equality),
                       "streams": equality}}
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    if not engaged or not result["gate"]["token_trace_equal"]:
        raise QualificationError(f"batch MTP gate failed; see {args.out}")
    print(args.out)
    return 0


def oracle_cli(args: argparse.Namespace) -> int:
    ref_path, got_path = pathlib.Path(args.reference), pathlib.Path(args.candidate)
    result = compare_oracles(json.loads(ref_path.read_text()), json.loads(got_path.read_text()),
                             args.atol, args.rtol)
    result.update({"schema_version": SCHEMA_VERSION, "kind": "gemma-oracle-comparison",
                   "reference_sha256": sha256_file(ref_path),
                   "candidate_sha256": sha256_file(got_path), "created_at": utc_now()})
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise QualificationError(f"oracle gate failed; see {args.out}")
    print(args.out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="launch and run qualification matrix")
    run.add_argument("--config", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--subject", action="append", default=[])
    run.add_argument("--allow-partial-matrix", action="store_true")
    run.set_defaults(func=run_matrix)
    validate = sub.add_parser("validate", help="validate matrix config")
    validate.add_argument("--config", required=True)
    validate.add_argument("--allow-partial-matrix", action="store_true")
    validate.set_defaults(func=lambda a: (validate_config(json.loads(pathlib.Path(a.config).read_text()),
                                                           not a.allow_partial_matrix) or 0))
    mtp = sub.add_parser("mtp-probe", help="compare MTP batch server with plain decode server")
    mtp.add_argument("--mtp-url", required=True)
    mtp.add_argument("--plain-url", required=True)
    mtp.add_argument("--model", required=True)
    mtp.add_argument("--batch", type=int, default=4)
    mtp.add_argument("--max-tokens", type=int, default=32)
    mtp.add_argument("--out", required=True)
    mtp.set_defaults(func=mtp_probe)
    oracle = sub.add_parser("oracle", help="compare trusted and candidate oracle artifacts")
    oracle.add_argument("--reference", required=True)
    oracle.add_argument("--candidate", required=True)
    oracle.add_argument("--atol", type=float, default=1e-3)
    oracle.add_argument("--rtol", type=float, default=1e-3)
    oracle.add_argument("--out", required=True)
    oracle.set_defaults(func=oracle_cli)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return int(args.func(args))
    except QualificationError as exc:
        print(f"qualification failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
