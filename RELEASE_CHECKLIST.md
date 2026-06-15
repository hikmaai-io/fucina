# Release checklist — fucina v0.1.0

Internal checklist for the first public release as `github.com/hikmaai-io/fucina`.

> **State of this branch (`release-prep`):** the rename is **done** (module path
> `github.com/hikmaai-io/fucina`, `cmd/fucina`, `libfucina.a`, `FUCINA_*` env vars), the diffusion
> engine is folded in, personal paths are scrubbed, and the **full build + dense smoke passed on the
> GB10**. What remains is the public push (step 3).

## 0 · Decisions (done)
- [x] Name: **fucina** · Org: **hikmaai-io** → `github.com/hikmaai-io/fucina`
- [x] First release scope: **dense Gemma 4 12B + experimental DiffusionGemma 26B-A4B** (`-dm`)

## 1 · Code rename (run on a CLEAN tree; requires a GB10 to verify)

Commit or stash all WIP first (`git status` must be clean), then run the rename script and
verify on the GB10:

```sh
./scripts/release/rename.sh        # rewrites code/build files only (docs already say fucina)
# then follow its printed steps:
#   - scrub the Makefile DG_GGUF / CUTLASS_DIR defaults (personal paths)
#   - gofmt -w .  &&  go vet + go test on the pure-Go packages
#   - make clean && make            # GB10 ONLY
#   - strings fucina | grep -q 'uploading.*weights to device'
#   - make smoke
```

The script is scoped to `*.go *.cu *.cuh *.h go.mod Makefile scripts/*.py`, does the module-path
move (`mauromedda/gem4d` → `hikmaai-io/fucina`), the `gem4d`→`fucina` and `GEM4D_`→`FUCINA_`
rewrites, and `git mv cmd/gem4d cmd/fucina`.

**Keep unchanged (do NOT rename):** the `gemma4_*` C symbols, `gemma4-assistant` arch id, and all
`gemma-4` / `Gemma 4` model identifiers — those name the *model*, not the project. (The literal
`gem4d` is never a substring of `gemma4`/`gemma-4`, so the rewrite leaves them intact.)

## 2 · Pre-flight (done on this branch)
- [x] `LICENSE` (Apache-2.0) + `NOTICE` (Gemma / llama.cpp / CUTLASS / unsloth attributions)
- [x] README: fucina brand, name etymology, accurate GB10-only taglines, no-support/experimental banner
- [x] `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`
- [x] `.github/` issue + PR templates (hardware gate), `config.yml` (no blank issues), CI (CPU tests)
- [x] `.gitignore` covers models, GGUFs, sqlite, runs/, data/, build artifacts
- [x] History clean: linear, no committed binaries, no co-author trailers
- [x] `CHANGELOG.md` (v0.1.0) + `docs/launch/RELEASE_NOTES_v0.1.0.md`
- [x] `scripts/release/rename.sh` and `scripts/release/publish.sh` (executable)

## 3 · Ship (after the rename + a green GB10 build)
- [ ] Merge `release-prep` (+ the rename commit) into `main`; ensure the tree is clean
- [ ] `./scripts/release/publish.sh` — creates the **public** repo `hikmaai-io/fucina`, sets the
      description + topics, enables Issues/Discussions + private vuln reporting, and pushes. (It
      prompts for confirmation; publishing is irreversible.)
- [ ] Verify CI is green on `main`
- [ ] Tag + release (the script prints these):
      `git tag -a v0.1.0 -m 'fucina v0.1.0' && git push origin v0.1.0`
      `gh release create v0.1.0 --title 'fucina v0.1.0' --notes-file docs/launch/RELEASE_NOTES_v0.1.0.md`

## 4 · Launch (optional)
- [ ] Draft the launch posts (blog · Show HN · X thread · r/LocalLLaMA) — hook: *parity-to-ahead of
      llama.cpp on the DGX Spark GB10*
- [ ] Be explicit everywhere: experimental, single-hardware, best-effort support

## 5 · Post-launch
- [ ] Triage with the "experimental / no SLA" expectation set in SECURITY.md & CONTRIBUTING.md
- [ ] Roadmap: harden the experimental DiffusionGemma path; sm_120 (RTX 50-series) port
