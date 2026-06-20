# fucina 2-bit store — on-disk format (v1)

The LEAN 2-bit quantized store produced by `produce_2bit_store.py`. The engine
loads this directly (no GGUF re-parse) to materialize the quantized weights.

## Container

A directory with exactly two files:

| file        | purpose                                                        |
|-------------|----------------------------------------------------------------|
| `store.json`| UTF-8 JSON header: format/version + one record per tensor.     |
| `store.bin` | Concatenated tensor payloads; regions located by header offsets.|

(A third file `produce_summary.json` is written for convenience — size, blended
bits, per-role rel_mse, runtime. It is NOT required to decode the store.)

## `store.json`

Top-level keys:

```
format       "fucina-2bit-store"
version      1
block        16                     codec block size for nf2/nf3
arch         <gguf general.architecture>   e.g. "gemma4"
source_gguf  <absolute path of the source GGUF>
recipe       human-readable recipe string
n_tensors    <int>
tensors      [ <tensor record>, ... ]   in GGUF file order
```

Each **tensor record**:

| key           | meaning                                                          |
|---------------|------------------------------------------------------------------|
| `name`        | original GGUF tensor name (e.g. `blk.0.attn_q.weight`)            |
| `dims`        | list[int], GGUF dim order (fastest-varying first)                |
| `n_elem`      | product(dims)                                                    |
| `src_type`    | GGUF source ggml type name (`Q4_0`/`Q4_1`/`Q4_K`/`F32`)          |
| `variant`     | `nf2` (2-bit) \| `nf3` (3-bit) \| `f32` (raw, norms/scalars)     |
| `role`        | embed/attn_k/attn_v/attn_q/attn_o/ffn_*/other (informational)   |
| `n_blocks`    | number of 16-elt codec blocks (`ceil(n_elem/16)`; 0 for f32)    |
| `codes_off`   | byte offset into `store.bin` of the packed codes                 |
| `codes_bytes` | length of the packed codes region                               |
| `scales_off`  | byte offset of the E4M3 block-scale array (= codes_off+codes_bytes)|
| `scales_bytes`| length of the scale array (== n_blocks bytes)                   |
| `raw_off`     | byte offset of the raw f32 payload (variant `f32` only)         |
| `raw_bytes`   | length of the raw f32 payload (variant `f32` only)              |

## `store.bin` payloads

### Quantized tensors (`variant` = `nf2` | `nf3`)

Two contiguous regions, codes then scales:

1. **codes** — bit-packed small codes, **LSB-first**, in codec order:
   block 0 elt 0, block 0 elt 1, …, block 0 elt 15, block 1 elt 0, ….
   `nf2` = 2 bits/code (4 levels), `nf3` = 3 bits/code (8 levels). The whole
   tensor is one bitstream of `n_blocks*16*bits` bits, rounded up to whole bytes
   (`codes_bytes = ceil(n_blocks*16*bits/8)`). Pack/unpack with numpy
   `packbits/unpackbits(..., bitorder="little")`.
2. **scales** — `n_blocks` bytes, one **E4M3 (OCP, bias 7)** block scale per
   block, stored as a `uint8`. Decode with the E4M3 decoder
   (`mxfp2.e4m3_decode`).

The last block of a tensor may be zero-padded up to 16 elements; the padding is
dropped on decode using `n_elem`.

### Raw tensors (`variant` = `f32`)

`n_elem` little-endian `float32`, verbatim, at `raw_off` (length `raw_bytes =
n_elem*4`). Used for 1-D RMSNorm gains, per-layer scalars, and `rope_freqs` —
not matmul weights, never quantized.

## Decode (per quantized tensor)

```
levels  = LEVELS_NF2 or LEVELS_NF3            # unit-scale codebook (mxfp2.py)
codes   = unpackbits(codes_region, LSB-first)[:n_blocks*16] -> reshape(nb,16)
scale   = e4m3_decode(scales_region)          # [nb] float32
w_hat   = levels[codes] * scale[:, None]      # [nb,16]
weights = w_hat.reshape(-1)[:n_elem]
```

`verify_store.py` is a reference decoder that does exactly this and checks each
tensor against the GGUF fp32 dequant.

## Codebooks

`nf2`/`nf3` are symmetric normal-float codebooks (quantiles of a standard
normal), normalized so the outermost level maps to ±1; the per-block E4M3 scale
stretches them to the block's range. Exact level tables live in `mxfp2.py`
(`LEVELS_NF2`, `LEVELS_NF3`). The encoder is the per-block MSE-optimal scale
search (`mxfp2.quantize_mse`); the decoder only needs the codebook + scale.

## Bits/elt accounting

| variant | codes | scale          | total                | bit/elt |
|---------|-------|----------------|----------------------|---------|
| nf2     | 16×2b | 1×E4M3 (8b)    | 40 b / 16 elt        | 2.5     |
| nf3     | 16×3b | 1×E4M3 (8b)    | 56 b / 16 elt        | 3.5     |

LEAN recipe (embed+attn_k+attn_v → nf3, rest → nf2) blends to **~2.62 bit/elt**,
**~10 GB** for the 31B weights.
