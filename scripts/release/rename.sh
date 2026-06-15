#!/usr/bin/env bash
#
# rename.sh — rebrand the SOURCE/BUILD tree from gem4d -> fucina and move the
# module path to github.com/hikmaai-io/fucina.
#
# Docs (*.md) already carry the fucina brand and intentionally reference
# "formerly gem4d", so they are NOT touched here.
#
# Run on a CLEAN working tree, then BUILD-VERIFY ON THE GB10 (see the tail).
#
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ -n "$(git status --porcelain)" ]; then
  echo "error: working tree is not clean — commit or stash WIP first." >&2
  exit 1
fi

# Code/build files that mention the brand.
# NOTE: git grep uses -z (lowercase) for NUL output, not -Z.
mapfile -d '' FILES < <(
  git grep -lz -e 'gem4d' -e 'GEM4D_' -- \
    '*.go' '*.cu' '*.cuh' '*.h' 'go.mod' 'Makefile' 'scripts/*.py'
)

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "no source files reference the old brand (already renamed?)."
else
  for f in "${FILES[@]}"; do
    # Order matters: module path (org+name) first, then the bare brand token,
    # then the env prefix. 'gem4d' is never a substring of 'gemma4' / 'gemma-4'
    # / 'benchmark_gem4', so the MODEL name and the bench script stay intact.
    sed -i \
      -e 's#github.com/mauromedda/gem4d#github.com/hikmaai-io/fucina#g' \
      -e 's/gem4d/fucina/g' \
      -e 's/GEM4D_/FUCINA_/g' \
      "$f"
  done
  echo "rewrote ${#FILES[@]} source/build file(s)."
fi

# Command package directory (binary output name follows the Makefile target).
if [ -d cmd/gem4d ]; then
  git mv cmd/gem4d cmd/fucina
  echo "moved cmd/gem4d -> cmd/fucina"
fi

cat <<'NEXT'

Mechanical rename done. Remaining steps:

  1. Scrub the personal paths in the Makefile DEFAULTS (keep them ?= overridable):
       DG_GGUF     ?= ./models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf
       CUTLASS_DIR ?= /path/to/cutlass     # documented in the README
  2. gofmt -w .
  3. CPU checks:
       go vet  ./internal/server/ ./internal/tokenizer/ ./internal/sampler/ ./internal/chat/
       go test ./internal/server/ ./internal/tokenizer/ ./internal/sampler/ ./internal/chat/ -count=1
  4. FULL build + smoke — ON THE GB10 ONLY:
       make clean && make
       strings fucina | grep -q 'uploading.*weights to device' && echo "device-upload path linked"
       make smoke
  5. Sanity:
       git grep -n 'gem4d\|GEM4D_\|mauromedda' -- '*.go' '*.cu' '*.cuh' '*.h' Makefile go.mod \
         || echo 'no stale brand tokens'
       git grep -c 'gemma4\|gemma-4' | tail -1   # model refs must still be present

Keep unchanged (these name the MODEL, not the project):
  gemma4_* C symbols, the gemma4-assistant arch id, all gemma-4 / "Gemma 4" identifiers.
NEXT
