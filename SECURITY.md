# Security Policy

> [!NOTE]
> **fucina is experimental software provided as-is, with no support or warranty.** There is no
> security SLA. The notes below are best-effort.

## Scope

fucina is a single-target inference engine (NVIDIA DGX Spark GB10). It serves an HTTP API; if you
expose it, treat it as you would any unhardened internal service:

- The OpenAI-compatible server has **no authentication or rate limiting**. Do **not** expose it
  directly to untrusted networks — put it behind your own auth/proxy.
- It is designed for a **single logical sequence / single slot** (`--n-slots 1`); it is not a
  multi-tenant server.
- Model weights (GGUF) are loaded from local paths you supply. Only load weights you trust.

## Reporting a vulnerability

Please report suspected vulnerabilities **privately**, not via public issues:

- Email **security@hikmaai.io** with a description, affected version/commit, and reproduction steps.
- Use GitHub's **private vulnerability reporting** ("Report a vulnerability" under the Security tab)
  if enabled.

We will acknowledge on a best-effort basis. As an experimental project, fixes are not guaranteed
within any fixed timeframe.

## Supported versions

Only the latest commit on the default branch is considered. There are no backported security fixes.
