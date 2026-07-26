# Qwen3.6 agentic tool-call parser incident

Date: 2026-07-25

Scope: Qwen3/3.5/3.6 tool-call decoding over OpenAI-compatible streaming and non-streaming HTTP, plus the Qwen3.6-35B-A3B source-FP8 admission failure found during GB10 qualification.

## Root cause

The Qwen3.6 27B-FP8, 35B-A3B-FP8, and Unsloth NVFP4 artifacts use the same 7,764-character `tokenizer_config.json` chat template. That template instructs the model to emit the Qwen XML-shaped dialect:

```text
<tool_call>
<function=NAME>
<parameter=KEY>
VALUE
</parameter>
</function>
</tool_call>
```

The exact instruction copied into fucina is visible at `internal/chat/qwen.go:55-64`. It is not the legacy Qwen3/3.5 JSON body (`<tool_call>{"name":...,"arguments":{...}}</tool_call>`). The MoE and native-NVFP4 models followed the template faithfully, while the dense model happened to emit the legacy form often enough to score 81/100. The 35B-A3B run scored 12/100 and the NVFP4 run failed in the same way. This is a wire-parser compatibility bug, not evidence of a checkpoint or quantization defect; this fix does not alter CUDA kernels, loader policy, or checkpoint precision.

The old HTTP failure amplified the parser miss: a `<tool_call>` span was hidden while waiting for its close, and malformed/unterminated finalization could discard the buffered body. The observed warning was `unterminated tool call ... no tool_calls emitted`; a runaway reached the independent 2,048-token safety cap and returned an empty assistant turn.

## Fix and exact code evidence

- `internal/chat/qwen.go:344-383` walks every sequential `<tool_call>` span, retains surrounding prose, assigns unique call IDs, and preserves malformed or partial spans as content rather than deleting them.
- `internal/chat/qwen.go:458-469` auto-detects the emitted body: leading `{` selects legacy JSON; otherwise the XML parser is used. There is no request, model, or quantization flag.
- `internal/chat/qwen.go:471-530` retains both legacy direct `{name,arguments}` and nested OpenAI-style `{function:{...}}` JSON compatibility, including JSON-string arguments.
- `internal/chat/qwen.go:533-593` parses complete XML function/parameter blocks. Parameter values are delimited only by `</parameter>`, so multiline text and ordinary `<`/`>` characters remain data. A missing function or parameter close is never dispatchable.
- `internal/chat/qwen.go:595-641` accepts spaces, tabs, LF, and CRLF around XML-shaped tags and `=`.
- `internal/chat/qwen.go:644-718` uses declared JSON Schema types first, then infers unambiguous objects, arrays, booleans, nulls, and JSON numbers; all other values are encoded as strings. Generated `function.arguments` is always valid JSON.
- `internal/server/server.go:1600-1631` emits a structured streaming delta as soon as each complete span closes; `internal/server/server.go:1740-1762` parses or exposes a final open span instead of silently dropping it.
- `internal/server/server.go:1645-1660` keeps the established 2,048-token/8-call runaway gates and includes the cap-triggering token in fallback content. Gemma's permissive legacy parser is not used to recover an open partial span (`internal/server/server.go:1751-1758,1835-1845,2637-2648`).
- `internal/chat/qwen.go:317-321` forces `tool_choice="required"` by opening a function call in the generation prompt. The server validates named choices and records the required contract at `internal/server/server.go:1000-1015`; no parsed call can finish as a clean `stop` (`internal/server/server.go:1767-1769,1854-1860,2364-2370,2688-2689`).

## CPU regression coverage

Parser tests include the captured Qwen3.6 XML call, typed and multiline values, `<`/`>` in values, CRLF/whitespace variants, legacy JSON, mixed JSON+XML calls, sequential calls, surrounding content, malformed closed input, and unterminated input (`internal/chat/qwen_test.go:122-295`).

HTTP tests cover XML streaming deltas, mixed content plus a call, legacy-JSON streaming compatibility, multiple sequential deltas, streaming and collected fallback, the 2,048-token cap fallback, safe recovery after a complete `</function>`, `tool_choice="required"`, malformed required output, named forcing, and unknown-name rejection (`internal/server/qwen_dialect_test.go:299-628`). `TestGemmaDialectParity` remains the byte-for-byte renderer gate (`internal/chat/qwen_test.go:380-411`).

CPU-only command:

```sh
go test ./internal/chat ./internal/tokenizer ./internal/server ./internal/server/batch -count=1
```

## GB10 hardware investigation and measured result

Hardware: GB10, CUDA 13, official `Qwen3.6-35B-A3B-FP8`, context 32768 (runtime cap 25280), port 8084. Every server run held `/tmp/fucina_gpu.lock`. The binary was rebuilt with `make fucina` before the final checks.

### Why source FP8 could not admit a sequence

The failure was before paged-KV admission and before the grouped expert GEMM. The safetensors loader correctly bound every core FP8 `WeightRef.scale` to its separately allocated BF16 128x128 block-scale grid. Later, the generic post-repack descriptor pass in `gemma4_engine_create` rebound all Qwen references and unconditionally set `ref.scale=nullptr`. Q4_K and NVFP4 did not consume that field, masking the bug. Source FP8 did.

With launch blocking and layer/phase synchronization, the first five-token warmup reached layer 0 and failed in `dequant_fp8_block_to_bf16_kernel` for `linear_attn.in_proj_qkv`: `fmt=FORMAT_FP8_BLOCK`, `in=2048`, `out=8192`, and `scale=NULL`. CUDA reported `an illegal memory access was encountered` before `moe_ffn` entry. The poisoned context then made multisequence admission and every serial fallback return `-1`; the slot and memory-plan messages were secondary symptoms. Expert-slab layout, FP8 expert scales, grouped expert dispatch, and paged KV had not run yet.

`cuda/gemma4_kernels.cu` now preserves the loader-owned scale pointer when the generic pass rebinds an FP8 descriptor, while still clearing scale for non-FP8 encodings.

### Source-FP8 retest

With `FUCINA_MOE_FP8=1` after that fix, startup reported:

```text
qwen35 allocation decision: source=block-FP8 mixer=FP8 experts=FP8 d_weights=31.41 GiB
qwen35 memory plan ready: ... slots=4/32 ...
qwen35 M4 batch graph captured ...
batch decode graphs warmed (B=1..4)
```

There were no `SeqAddMultiseq` or `AddSeq` errors. HTTP prefill and decode both ran: a plain deterministic request returned `content="Paris"`, `completion_tokens=1`, `finish_reason="stop"`; a forced weather-tool request returned 46 generated tokens at 29.8 tok/s instead of the former zero-token response. The latter was malformed model text, not a dispatchable call, so the hardened HTTP path exposed it as non-empty fallback content.

This proves the admission crash precisely and repairs the opt-in source-FP8 data path, but it does **not** qualify source FP8 as the agentic default: its longer prose and tool syntax were still corrupt on this checkpoint.

### Default and quality decision

The engineering resolution is therefore **(b)**. The default remains the previously serving requantized path (`mixer=Q4_K experts=NVFP4`); source FP8 remains an explicit diagnostic mode. The final default run reported `d_weights=0.88 GiB`, `weights+scratch=25.46 GiB`, `slots=4/32`, warmed all batch graphs, and admitted real requests without fallback admission errors.

Two real HTTP checks against that default on port 8084 produced:

- plain deterministic check: `content="Paris"`, `completion_tokens=1`, `finish_reason="stop"`;
- forced named weather tool: HTTP 200 with 128 completion tokens and non-empty malformed fallback content, rather than the previous empty zero-token turn.

The 15-scenario `tool-eval-bench v2.0.4 --short` hardware quality check completed with **0/100 (0/15 passed)**; report: `runs/qwen36-35b-default-final/2026/07/2026-07-25T18-25-16.196987Z_0d1b6a6d.md`. This is a failed agentic-quality gate and is recorded rather than relabeled as success. The HTTP parser fix prevents dead/empty turns and safely parses valid XML or legacy JSON, but it cannot turn the malformed MoE token stream into valid calls. A complete 69-scenario claim is not made.

## Shared-path audit

The minimal failing chat request was compared directly with the official Qwen3.6 tokenizer, not inferred from decoded text. `AutoTokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` and fucina render the same bytes:

```text
<|im_start|>user
Write one short sentence about the sea.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Both produce the same 20 IDs:

```text
[248045, 846, 198, 7734, 799, 2716, 11316, 883, 279, 9117,
 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
```

The HF tokenizer reports a base vocabulary of 248,044 plus 33 added tokens; the model deliberately pads its embedding/head space to 248,320. Every ID above is valid. Fucina uses each ID directly as the row index into the checkpoint's `[248320, hidden]` BF16 embedding table; there is no BOS insertion, ID remap, or off-by-one. `TestQwen36_SeaPromptHFParity` locks the rendered bytes and IDs. Debug request dumps now include `TOKEN IDS (EXACT ENGINE INPUT)` after the rendered prompt.

The model config was also checked rather than interpreting the generic startup labels: the 40-layer MoE pattern is exactly 30 `linear_attention` layers and 10 `full_attention` layers at indices 3, 7, …, 39. The “30 sliding + 10 global” startup line is generic accounting terminology; Qwen's 30 entries dispatch through GDN and do not use sliding softmax attention. Qwen's 10 FULL layers use their own per-slot absolute-position fp16 K/V arrays, not the generic paged sliding/global pool. RoPE uses the runtime position with head_dim 256, partial rotary width 64, and theta 10,000,000. GDN decode derives the runtime MoE geometry (`NQ=16`, `NKV=2`, `NVH=32`, `INNER=4096`, `CONVD=8192`); the graph-on/off row gate exercises the same recurrent and FULL-KV state over 24 accumulating steps.

## MoE graph-router numerical corruption and resolution

A follow-up GB10 investigation treated the failed batched self-test as a hard correctness gate rather than tolerating the narrow eight-token oracle. The historical result in `benchmark-evidence/results/2026-07-19-qwen35-burst-ttft2/qwen-gates.log` was reproduced before the fix:

```text
MoE FP8 engine oracle-parity: 8/8
qwen35 M4 seq 0: B=3 vs B=1 6/24; graph-on vs graph-off 6/24
qwen35 M4 self-chain: FAIL
FAIL — ... (oracle 8/8, self-test FAIL)
```

The defect was in router execution before expert arithmetic, not in tool parsing, the NVFP4 requantizer, grouped-GEMM indexing/scales, top-k normalization, shared-expert accumulation, KV, or GDN state. The component isolation was:

- Default `mixer=Q4_K experts=NVFP4`, `FUCINA_MOE_Q4K=1`, and source `FUCINA_MOE_FP8=1` all failed the same graph/self-chain test. Changing the routed-expert storage and GEMM therefore did not remove the corruption.
- `FUCINA_NO_UNIFY=1` passed all 24 decode steps for all three ragged rows, graph-on/off, sampling, self-chain, and the 8/8 oracle while leaving the default NVFP4 expert conversion/GEMMs and shared expert intact. This isolated the difference to the unified router branch.
- The prompt's first tokens came from non-graph prefill and were coherent; corruption began on subsequent graph decode. That ruled out a pure load-time requantization error and matched stale per-step scratch rather than accumulating KV/GDN drift.

The exact bug was the short-decode router's `cublasSgemm`. CUDA graphs capture the decode body on a temporary stream, but the shared cuBLAS handle remained bound to the engine stream. The router SGEMM consequently executed outside capture. Replays updated the hidden state but reused router logits produced at capture time, selecting and weighting the wrong experts on every later token. This explains why kernels remained fast, why all expert formats failed, why graph-off could produce valid text, and why an eight-token oracle generated partly during prefill could pass.

`cuda/gemma4_kernels.cu` now always uses the explicit-stream router GEMV for decode-sized calls. Wide prefill retains SGEMM and explicitly binds the cuBLAS handle to the caller's stream. Router softmax/top-k/renormalization, grouped expert GEMMs, shared-expert addition, and NVFP4 weight conversion are unchanged.

### Measured after the router fix

Both required gates passed on GB10 with the official Qwen3.5-35B-A3B-FP8 oracle artifact:

```text
make qwen35-moe-fp8-engine-test
# row-independence PASS; graph-on==off PASS; M3 parity PASS (8/8);
# sampling PASS; self-chain PASS; final PASS

make qwen35-moe-fp8-test
# PASS — 8/8 greedy parity vs torch FP8 oracle
```

The official Qwen3.6-35B-A3B-FP8 was then rebuilt and served under `/tmp/fucina_gpu.lock` on port 8086 with the default `mixer=Q4_K experts=NVFP4`. Its internal 24-step ragged-row checks were 24/24 for B=3 versus B=1 and graph-on versus graph-off, and self-chain passed. (The test binary's Qwen3.5-specific pinned continuation is not a Qwen3.6 cross-version oracle.) Real HTTP checks produced:

- exact requested plain prompt, greedy, `max_tokens=60`: **“The sea whispers ancient secrets to the shore.”** — 9 completion tokens, `finish_reason="stop"`, 38.8 generated tok/s including request timing;
- forced `get_weather` tool request: `finish_reason="tool_calls"`, name `get_weather`, arguments `{"city":"Paris"}` — 17 completion tokens, 25.4 generated tok/s including the 270-token tool-schema prefill.

This changes the earlier qualification result: the parser fix and source-FP8 admission fix were valid, but the remaining malformed output was caused by the graph-capture router arithmetic defect. Greedy prose and structured calls are coherent after fixing that defect; the eight-token oracle remains a regression gate, not the sole quality criterion.

### Exact failing-prompt reference and raw-completion verification

A final clean rerun used official snapshot `95a723d08a9490559dae23d0cff1d9466213d989`, a freshly rebuilt binary, port 8086, and `/tmp/fucina_gpu.lock`. Direct local `transformers.AutoTokenizer` output and fucina's `FUCINA_DEBUG=1` request dump agreed one-by-one on all 20 chat IDs listed above. The CPU renderer/tokenizer/server tests also passed with that snapshot's `tokenizer.json`.

The exact 20-ID failing chat prompt was then passed through the standalone source-FP8 `qwen35_moe_fp8_forward_greedy` path, replacing the historical five canned France IDs. It generated:

```text
[760, 9117, 85111, 13437, 22857, 310, 279, 29199, 13, 248046, 198, 248044]
```

This decodes to `The sea whispers ancient secrets to the shore.<|im_end|>\n<|endoftext|>`. The production Q4_K/NVFP4 graph path generated the same nine lexical token IDs and returned **“The sea whispers ancient secrets to the shore.”** at 40.8 tok/s. End-to-end token equality on the actual repro made an additional per-layer dump unnecessary after the graph-on/off self-test had already isolated the old divergence to the out-of-capture router SGEMM.

Finally, raw `/v1/completions` received the verbatim eight IDs `[7734, 799, 2716, 11316, 883, 279, 9117, 13]` with no ChatML tokens and produced coherent English (`Thinking Process: ...`) for the full 60-token cap at 57.8 tok/s. As expected for an untemplated base completion, it did not follow the chat instruction as tightly; importantly, it was not corrupt. The exact chat acceptance prompt is therefore coherent on both standalone source FP8 and the production quantized graph path.

## Final GB10 qualification

The complete 69-scenario `tool-eval-bench v2.0.4` run on the router-fixed Qwen3.6-35B-A3B-FP8 path scored **80/100 (110/138, Good)** in **347 s**, up from the reproduced pre-fix **12/100** run that took **1,299 s**. The accepted report is run `2026-07-25T18-58-59.079320Z_179ae609`. It selected `get_weather` correctly for Berlin (`{"location":"Berlin","units":"celsius"}` in the full tool schema) and retained coherent prose. This puts sparse MoE agentic quality at practical parity with the same-day dense Qwen3.6-27B-FP8 result (**81/100**).

A subsequent merge-candidate rerun made the exact scalar continuation-prefill path the production default because the old approximate tensor-core continuation scored only 2/25 against scalar/one-shot after router replay was corrected. That stricter candidate scored **78/100 (107/138, Good)** in **370.3 s**; TC-33 changed from pass to fail and TC-53 from pass to partial. Both numbers are retained: **12→80** measures the graph-router incident fix requested by this qualification, while **78** is the final exact-continuation branch result. No 80-point claim is attached to the later binary.

### Final correctness and regression gates

On the final branch and the official Qwen3.5-35B-A3B-FP8 oracle checkpoint:

```text
B=3(graph) vs B=1(per-kernel): 24/24 for all three ragged rows
graph-on vs graph-off:         24/24 for all three ragged rows
self-chain:                    PASS
engine oracle:                 8/8
standalone torch oracle:       8/8
gpu-gates:                     PASS
production continuation:       scalar/one-shot 25/25
```

This resolves—but does not rewrite—the historical `FAIL` at `benchmark-evidence/results/2026-07-19-qwen35-burst-ttft2/qwen-gates.log:417` (`oracle 8/8, self-test FAIL`). Explicit reruns also passed `qwen35-state-test` (16/16), `qwen35-shard-test` (8/8 frontier and oracle), and `qwen35-http-session-restart-test` (11 cached tokens restored; only the 7-token suffix prefilled).

The full MoE decode microbench on Qwen3.6-35B-A3B-FP8 measured B=1 at **55.7, 59.4, and 59.0 tok/s** over three fresh 128-step runs: **59.0 tok/s median** (16.96 ms/step median). This is above the 46–53 tok/s historical served baseline, so moving decode routing onto the captured stream did not cost throughput. B=2/4/8 median aggregate rates were **95.4/150.3/219.8 tok/s**.

The dense Qwen3.6-27B-FP8 smoke measured **11.90 tok/s** over 64 B=1 graph steps and returned both the exact prose **“The sea whispers ancient secrets to the shore.”** and a correct forced call, `get_weather` with `{"location":"Berlin"}`. This is a brief regression smoke, not a replacement for its existing 81/100 quality run.

### Remaining gaps

- **Closed 2026-07-26:** structured `response_format` now carries per-slot grammar state through
  continuous batching. TC-64/65/66/67/69-style hardware requests on Qwen3.6-35B-A3B-FP8 were
  5/5 schema-valid (previously 5/5 HTTP 501); see [`batched-structured-output.md`](batched-structured-output.md).
- **TC-60 Cross-Turn Sleeper Injection still fails** and remains safety-critical.
- The exact scalar continuation path restores 25/25 correctness but is slower than the quarantined tensor-core candidate for base>0 chunks (about 2.04 s versus 1.09 s for the measured 1,376-token continuation). Decode throughput is unaffected.
