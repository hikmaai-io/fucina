# Release checklist — fucina v0.1.0

Internal checklist for the first public release as `github.com/hikmaai-io/fucina`.

> **State of this branch (`release-prep`):** the **docs and metadata are already fucina-branded**
> (README, LICENSE, NOTICE, CONTRIBUTING, SECURITY, CI, templates). The **code rename is NOT done
> yet** — the Go module path, binary, `cmd/` dir, cgo lib name, and `GEM4D_*` env vars still say
> `gem4d`. Run the rename below on a **quiet tree** and **build-verify on the GB10** before tagging.

## 0 · Decisions (done)
- [x] Name: **fucina** · Org: **hikmaai-io** → `github.com/hikmaai-io/fucina`
- [x] First release scope: **dense Gemma 4 12B only** (DiffusionGemma held for v0.2)
- [ ] Cofounder / board sign-off that this is a company-owned OSS release (IP)

## 1 · Code rename (run on a CLEAN tree; requires a GB10 to verify)

Commit or stash all WIP first (`git status` must be clean), then:

```sh
set -e

# a) module path: org + name, across every file that references it
git grep -lZ 'github.com/mauromedda/gem4d' \
  | xargs -0 sed -i 's#github.com/mauromedda/gem4d#github.com/hikmaai-io/fucina#g'

# b) all remaining brand tokens. SAFE: the literal "gem4d" never occurs inside
#    "gemma4" / "gemma-4" / "scripts/benchmark_gem4.py", so the model name and the
#    benchmark script are untouched. (Verify with the grep in step d.)
git grep -lZ 'gem4d'  | xargs -0 sed -i 's/gem4d/fucina/g'
git grep -lZ 'GEM4D_' | xargs -0 sed -i 's/GEM4D_/FUCINA_/g'

# c) rename the command package directory (binary output already became `fucina` via (b))
git mv cmd/gem4d cmd/fucina

# d) scrub hardcoded personal paths in the Makefile DEFAULTS (keep them ?= overridable):
#      DG_GGUF     ?= /home/mauromedda/unsloth/...    ->  ./models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf
#      CUTLASS_DIR ?= /home/mauromedda/.venv/...      ->  /path/to/cutlass   (documented in README)
#    (edit by hand; they are build-config defaults, not code)

# e) tidy + CPU verification
gofmt -w .
go vet ./internal/server/ ./internal/tokenizer/ ./internal/sampler/ ./internal/chat/
go test ./internal/server/ ./internal/tokenizer/ ./internal/sampler/ ./internal/chat/ -count=1

# f) confirm nothing stale remains, and the model name survived intact
git grep -n 'gem4d\|GEM4D_\|mauromedda' || echo "no stale brand tokens ✅"
git grep -c 'gemma4\|gemma-4' | tail -1   # model refs must still be present

# g) FULL build + smoke — ON THE GB10 ONLY
make clean && make
strings fucina | grep -q 'uploading.*weights to device' && echo "device-upload path linked ✅"
make smoke
```

**Keep unchanged (do NOT rename):** the `gemma4_*` C symbols, `gemma4-assistant` arch id, and all
`gemma-4` / `Gemma 4` model identifiers — those name the *model*, not the project.

## 2 · Pre-flight (done on this branch)
- [x] `LICENSE` (Apache-2.0) + `NOTICE` (Gemma / llama.cpp / CUTLASS / unsloth attributions)
- [x] README: fucina brand, name etymology, accurate GB10-only taglines, no-support/experimental banner
- [x] `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`
- [x] `.github/` issue + PR templates (hardware gate), `config.yml` (no blank issues), CI (CPU tests)
- [x] `.gitignore` covers models, GGUFs, sqlite, runs/, data/, build artifacts
- [x] History clean: linear, no committed binaries, no co-author trailers

## 3 · Ship
- [ ] Create/confirm repo `github.com/hikmaai-io/fucina` (description: "Gemma 4 inference forged for the NVIDIA DGX Spark GB10 — experimental, no support")
- [ ] Set topics: `gemma`, `cuda`, `llm-inference`, `dgx-spark`, `blackwell`, `go`
- [ ] Enable Discussions; enable private vulnerability reporting (for SECURITY.md)
- [ ] Push `main`; verify CI is green
- [ ] Tag **`v0.1.0`** + GitHub Release (source release — no universal binary; optionally attach a GB10 build)

## 4 · Launch (optional)
- [ ] Short post leaning on the real hook: *parity-to-ahead of llama.cpp on the DGX Spark GB10*
- [ ] Be explicit everywhere: experimental, single-hardware, best-effort support

## 5 · Post-launch
- [ ] Triage with the "experimental / no SLA" expectation set in SECURITY.md & CONTRIBUTING.md
- [ ] Roadmap: sm_120 (RTX 50-series) port; DiffusionGemma as v0.2
