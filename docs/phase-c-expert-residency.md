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

`internal/engine/expertstore` implements the runtime-independent three-tier cache controller:

- immutable SSD records addressed by offset and length;
- SHA-256 verification before promotion;
- bounded host and VRAM LRU metadata;
- an uploader interface for CUDA allocation/copy and eviction;
- concurrent prefetch workers and cancellation-aware scheduling;
- an online, bounded per-layer transition predictor over top-k expert IDs;
- VRAM/host hit, SSD read, promotion, eviction, checksum, byte, occupancy, prefetch, and useful-prefetch metrics.

The package is tested under an artificial one-expert VRAM cap, including host promotion, eviction,
checksum failure, concurrent prefetch, useful-prefetch accounting, and transition prediction.

## Remaining integration

The current Qwen CUDA loader still materializes grouped expert slabs at startup. The next C1/C2
increment must expose a slot-based CUDA uploader and remap router expert IDs to resident slots,
then connect the controller to lookahead prefetch. Until that integration, residency manifests are
planning artifacts and the runtime status says so explicitly; no memory-saving claim is made.

Run the foundation gates with:

```bash
make phase-b-test
make go-test
```
