# Security Policy

> [!NOTE]
> **fucina is experimental software provided as-is, with no support or warranty.** There is no
> security SLA. The notes below are best-effort.

## Scope

fucina is a single-target inference engine (NVIDIA DGX Spark GB10). It serves an HTTP API; if you
expose it, treat it as you would any unhardened internal service:

- Authentication is **optional and off by default**: `--api-key`/`FUCINA_API_KEY` gates `/v1/*`
  with a constant-time bearer-token check, but with no key set (the default) `/v1/*` is open to
  anyone who can reach the port. `/health`, `/healthz`, `/readyz`, `/metrics` are always open
  regardless. `--max-concurrent` bounds the admission queue (excess requests get `503`) but this is
  a resource-exhaustion guard, not a rate limiter. Do **not** expose fucina directly to untrusted
  networks without setting `--api-key` and/or putting it behind your own auth/proxy.
- **Concurrency model differs by architecture.** Gemma-4 defaults to a single logical sequence
  (`--n-slots 1`); concurrent serving there is opt-in via `--batch`. Every Qwen3/Qwen3.5/Qwen3.6
  checkpoint is **always** served through continuous batching (multiple concurrent sequences share
  a paged KV cache and a per-step scheduler) — this is closer to a multi-tenant server than the
  Gemma-4 default, so apply the same access controls (auth, network isolation) you would to any
  multi-tenant inference service when serving a Qwen checkpoint to more than one trusted caller.
- Model weights (GGUF or safetensors) are loaded from local paths you supply. Only load weights
  you trust — a HuggingFace checkpoint is untrusted input like any other file format.

## Reporting a vulnerability

Please report suspected vulnerabilities **privately**, not via public issues:

- Email **security@hikmaai.io** with a description, affected version/commit, and reproduction steps.
- Use GitHub's **private vulnerability reporting** ("Report a vulnerability" under the Security tab)
  if enabled.

We will acknowledge on a best-effort basis. As an experimental project, fixes are not guaranteed
within any fixed timeframe.

## Supported versions

Only the latest commit on the default branch is considered. There are no backported security fixes.
