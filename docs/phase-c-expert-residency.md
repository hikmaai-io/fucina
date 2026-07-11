# Phase C: expert residency and streaming

Phase C consumes the measured expert heat map rather than assuming all experts are equally hot.
The implementation is intentionally split at the CUDA boundary so storage policy can be tested
without a 35B checkpoint.

## C1 foundations

Create a deterministic placement plan:

```bash
python3 scripts/derive_residency_plan.py /tmp/model.imatrix.json \
  --expert-vram-gib 8 --expert-host-gib 16 \
  --out /tmp/model.residency.json
```

The planner uses exact calibration route counts, stable tie-breaking, model geometry, and an
effective NVFP4 bits-per-weight including scale overhead. Its manifest reports occupancy and the
fraction of calibration routes served by VRAM, host, and SSD.

An earlier generic Go three-tier cache controller (`internal/engine/expertstore`) was removed: it
duplicated bookkeeping the CUDA runtime below implements directly at the layer where the copies and
GEMMs happen, and nothing wired it to the engine.

## CUDA runtime integration

The Qwen grouped-NVFP4 path has an opt-in bounded-memory SSD mode:

```bash
FUCINA_EXPERT_STREAM_SSD=/fast-nvme/qwen-experts.bin \
FUCINA_EXPERT_STREAM_SLOTS=512 \
./fucina -m /path/to/Qwen3.5-35B-A3B-NVFP4 ...
```

At startup it writes the transformed 16.88 GiB grouped-NVFP4 expert store, drops the full device
slabs, and retains a configurable compact slot pool. After routing, logical expert IDs are mapped
to slots; only active expert records are read and uploaded. The CUTLASS grouped GEMM consumes the
slot map while retaining logical assignment offsets. If a prefill activates more experts than the
slot cap, it processes deterministic chunks rather than exceeding the cap. A cross-layer LRU keeps
hot `(layer,expert)` records resident across decode tokens. Every weight and scale record is
checksummed on first use (mismatch aborts), and graph capture is disabled because SSD I/O is not
capturable. Normal serving remains unchanged when the environment variable is not set. The CLI
equivalents are `--expert-store <file>` and `--expert-slots <n>`.

A host-RAM staging variant was tried and removed: GB10 CPU/GPU memory is physically unified, so
staging from host memory saves nothing.

### Measured gate

On the local GB10 with eight slots, the SSD path reduced transformed expert device staging from
16.88 GiB to about **0.01 GiB** and passed the Qwen3.6 NVFP4 oracle/self-test: 8/8 oracle tokens,
24/24 batched row independence, graph-on/off fallback parity, and France→Paris 8/8. A 512-slot
cache uses about 0.84 GiB and avoids reloading overlapping active experts. The mode is a constrained
memory fallback, not the default throughput path; synchronous NVMe misses remain materially slower
than full residency.

The residency-plan manifest is not yet automatically loaded by the C engine. Runtime placement is
demand-driven with an LRU slot budget; policy-seeded startup residency and asynchronous next-layer
prefetch remain optimization work, not correctness blockers.

Run the foundation gates with:

```bash
make phase-b-test
make go-test
```
