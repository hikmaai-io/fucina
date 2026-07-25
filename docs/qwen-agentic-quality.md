# Qwen3.6 agentic tool-call parser incident

Date: 2026-07-25

Scope: Qwen3/3.5/3.6 tool-call decoding over OpenAI-compatible streaming and non-streaming HTTP. No CUDA build, model load, or GPU test was run for this fix.

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

## Hardware re-benchmark (run when the GPU is free)

Build once, then run the same complete 69-scenario suite against all three artifacts. These commands are intentionally serial and preserve separate reports:

```sh
make fucina

PORT=8081 MODEL=/opt/spark/models/hub/models--Qwen--Qwen3.6-27B-FP8 \
  OUTPUT_DIR=./runs/qwen36-27b-fp8-parser-fix \
  scripts/tool_eval_bench.sh 2>&1 | tee ./runs/qwen36-27b-fp8-parser-fix.log

PORT=8082 MODEL=/opt/spark/models/hub/models--Qwen--Qwen3.6-35B-A3B-FP8 \
  OUTPUT_DIR=./runs/qwen36-35b-a3b-fp8-parser-fix \
  scripts/tool_eval_bench.sh 2>&1 | tee ./runs/qwen36-35b-a3b-fp8-parser-fix.log

PORT=8083 MODEL=/opt/spark/models/unsloth/Qwen3.6-35B-A3B-NVFP4 \
  OUTPUT_DIR=./runs/qwen36-35b-a3b-nvfp4-parser-fix \
  scripts/tool_eval_bench.sh 2>&1 | tee ./runs/qwen36-35b-a3b-nvfp4-parser-fix.log
```

Acceptance checks: XML calls appear as OpenAI `tool_calls`, streaming and non-streaming results agree, TC-45 honors `required`, no malformed partial call is dispatched, and no unterminated span produces an empty response. Keep the existing dense, MoE, session, and Gemma protection gates unchanged.
