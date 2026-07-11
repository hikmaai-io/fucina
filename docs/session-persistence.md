# Disk-backed session persistence (Phase D)

A session file persists one conversation's engine state across process
restarts, so a later run resumes generation without re-prefilling the saved
prefix. Format and loaders live in `internal/session`.

## File format (`FUCINASESS1`, version 1)

All integers little-endian; layout in `internal/session/session.go`:

| section | contents |
|---|---|
| magic + version | `"FUCINASESS1\0"`, u32 version |
| metadata | u32 length, JSON (`created_at`, `n_tokens`, `engine_kind`, model identity, opaque client blob), FNV-1a64 |
| token history | u32 count, int32 ids, FNV-1a64 |
| engine state | u64 length, opaque snapshot bytes, FNV-1a64 |

Model identity = model path (informational) + FNV-1a64 of the model config
(`config.json`, or the leading 1 MiB of a GGUF) + a state-geometry probe (the
engine's snapshot size for a fixed 16-token sequence). Load rejects bad
magic/version, hostile section lengths (checked before allocation), checksum
mismatches, trailing bytes, and — via `Meta.Validate` — sessions saved for a
different model, engine kind, or state geometry.

Two engine kinds, never interchangeable:

- `flat-kv` — the flat single-sequence engine (Gemma): attention KV only
  (`gemma4_engine_kv_save`).
- `q35-slot` — the Qwen3.5/3.6 hybrid per-slot snapshot
  (`gemma4_engine_q35_state_save`): fp16 full-layer K/V **plus the GatedDeltaNet
  recurrent state and causal-conv rings per linear layer** — verified present in
  the C snapshot path; without them a hybrid session could not resume.

## REPL

`/save FILE` and `/load FILE` in both REPLs (the chat transcript rides in the
metadata client blob so future prompts re-render byte-identically):

- Dense (Gemma): `/save` exports the live KVCache sequence; `/load` restores it
  immediately — the next turn reuses the whole session as a prefix.
- Paged (Qwen hybrid): `/save` rebuilds slot state by prefilling the rendered
  transcript once (the cost of saving, not of every turn); `/load` keeps the
  snapshot and every subsequent turn restores it into a fresh slot and prefills
  only the suffix. The prefill stats line reports `(R from session, N new)`.

An exactly-matching prompt still re-runs its final token: logits are not part
of the saved state; the "zero prefill" guarantee is for the saved prefix.

## Server

Start with `--session-dir DIR`. A request carrying `"session": "NAME"` (a
validated name, never a path) loads `DIR/NAME.fcsess` before prefill — seeding
it into the KVCache snapshot pool, so a matching prompt restores instead of
prefilling — and saves the updated conversation back after generation.
Corrupt, truncated, or wrong-model session files are a 400 with the precise
reason. Missing files are created on first save. Resume effectiveness is
observable in `/metrics` (`kv_cache.reused_tokens`) and the per-request prefill
log line (`N cached, M new`).

Scope: the session field rides the single-flight KVCache path. The
continuous-batching path keeps its in-memory per-conversation state cache
(BatchAdapter); wiring disk sessions into the scheduler is future work — use
the REPL commands for Qwen hybrid models meanwhile.

## Gates

```bash
go test ./internal/session/ ./internal/server/   # format, corruption, resume
make session-restart-test                        # GPU: save → restart → /load →
                                                 # continue with zero re-prefill
```
