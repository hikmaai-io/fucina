<!-- fucina is experimental and single-target (DGX Spark GB10). See CONTRIBUTING.md. -->

## What & why

## Type
- [ ] Bug fix
- [ ] Feature
- [ ] Performance
- [ ] Docs / chore

## Verification
<!-- For CUDA/engine changes, the correctness bar is bit-exactness. -->
- [ ] `gofmt -l .` clean, `go vet` + pure-Go tests pass (`make check`)
- [ ] CUDA changes (if any) built on a **GB10** (`make`) and `make smoke` runs
- [ ] Greedy output byte-identical to baseline (or provably reassociation-only) — say which harness/prompts
- [ ] `compute-sanitizer` memcheck clean for touched kernels

## Notes
- [ ] Commits signed off (`git commit -s`)
