# GB10 qualification evidence

`scripts/run_gb10_qualification.sh` is the common evidence wrapper for GB10
performance claims. It deliberately performs **exactly three independent
server process starts**. A benchmark command runs once after each process
becomes ready and must write its unsummarized samples as a top-level JSON array.

Example:

```sh
scripts/run_gb10_qualification.sh \
  --claim qwen35-dense-c32 \
  --model /opt/spark/models/Qwen3.5-9B \
  --server-command './fucina -m /opt/spark/models/Qwen3.5-9B --port 18080' \
  --benchmark-command 'python3 scripts/my_probe.py --raw "$FUCINA_RAW_RESULTS"' \
  --ready-url http://127.0.0.1:18080/readyz
```

The benchmark and server commands receive:

- `FUCINA_RUN_INDEX` (`1`, `2`, or `3`)
- `FUCINA_RUN_DIR`
- `FUCINA_RAW_RESULTS` (required benchmark output path)
- `FUCINA_SOURCE_COMMIT`

Each archive under `benchmark-evidence/qualification/` contains:

- `manifest.json`: claim, full source commit, dirty flag, host identity, model
  fingerprints, and the three raw-result paths
- `source-commit.txt` and `source-working-tree.patch`
- full SHA-256 model fingerprint and per-file hashes
- `nvidia-smi`, driver query, `nvcc --version`, CUDA version, and `uname`
- commands, readiness response, stdout/stderr, PID, and timestamps per start
- `run-{1,2,3}/raw-results.json`, retained as arrays rather than only percentiles

The runner refuses a dirty source tree by default. `--allow-dirty` exists for
harness tests and exploratory runs; evidence used for a release claim should
never use it. Model hashing reads every model byte once before the three starts,
so plan for this setup cost with large sharded checkpoints.
