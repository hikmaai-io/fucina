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

The remaining quality defect is common to source-FP8, Q4_K-expert, and NVFP4-expert checks and therefore lies after admission in shared MoE forward/model compatibility, not in the fixed scale-pointer crash or the HTTP parser. Before any future default change, capture layer/router/shared-expert parity for this exact Qwen3.6 revision against `cuda/qwen35_moe_fp8_ref.py`; require coherent prose plus structured calls in the full suite. Never use a narrow eight-token oracle as the default gate.
