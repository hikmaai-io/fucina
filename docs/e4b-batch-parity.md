# E4B batched decode parity

## Root cause

The historical short-prompt failure was not a position, KV-ring, PLE-index, or
sliding-window off-by-one. The per-slot path already copied `Slot::n_past` into
`pos[b]`, wrote K/V at `pos[b] % cap`, selected each slot's own cache pointers,
and used `lo = max(0, pos-window+1)`. For prompt lengths 1, 2, and 3 those
positions and mask lengths were correct.

Batched and independent decode were instead evaluating different numerical
models. Safetensors independent decode used FP8 Q/K, NVFP4 V/O/FFN, and the FP8
tied head, while batched decode silently used the BF16 weights and BF16 head.
Most greedy decisions had enough margin to hide that difference. The length-2
prompt reached a near tie between tokens 818 and 64753, exposing the repeatable
phase-like token drift.

Two additional arithmetic differences violated the stronger row-invariance
requirement even though replacing attention alone did not change the recorded
2/8 failure: independent decode used split-K online-softmax attention while
batching used a serial-dot reduction, and BF16 PLE GEMMs used cuBLAS `T=B`
rather than independent `T=1` calls. The fix aligns these paths as well instead
of relying on current argmax margins.

## Fix and invariants

The E4B batch path now follows the independent decode arithmetic:

1. Batched FP8 and NVFP4 GEMV kernels reuse each weight load across rows but keep
   each row's accumulation order and warp reduction identical to the one-row
   kernels. Scratch is allocated and indexed for 32 independent rows.
2. The FP8 tied head is used for safetensors batching, matching independent
   decode. Native GGUF Q4_0/Q6_K batching remains unchanged.
3. Each row uses the same split-K attention and combine kernels as independent
   decode, with its own position and KV pointers. Partial scratch is reused only
   after the preceding row's combine on the same stream.
4. PLE and explicit BF16-fallback projections run as independent `T=1` cuBLAS
   calls so batch composition cannot select a different reduction tiling.

No prompt-length special case is present. The regression gate includes prompt
lengths 1 through 5 in one ragged batch and compares eight greedy tokens per row
against independent generation.
