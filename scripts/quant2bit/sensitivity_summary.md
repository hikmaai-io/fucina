# Stage B2 — Per-tensor 2-bit sensitivity sweep (NF2)

Reference = Q4_0/Q4_1/Q4_K GGUF dequant (the 4-bit weights we ship from). Errors are the **incremental** 4-bit -> ~2.5-bit loss (Lever F_bytes), not the true-BF16 floor.

- Quantizable weight tensors swept: **411**
- F32 norm/scalar tensors excluded (not matmul weights): **422**
- Format: NF2 codebook, 16-elt block, 1 E4M3 scale + 16x2-bit = 0.3125 B/elt = 2.5 bit/elt -> 31B = 9.69 GB
- Headline metric below: **nf2 absmax rel_mse** (the shippable NVFP4-style encoder, run on ALL 411 tensors). The per-block MSE-optimal floor is reported per-role below (run on one representative tensor per role; ~24x slower).

Overall rel_mse (absmax): mean=0.1545 median=0.1534 p90=0.1601 max=0.1975

## MSE-optimal floor by role (one tensor each)

| role | tensor | absmax rel_mse | mse rel_mse | mse cos | mse sqnr_dB |
|---|---|---|---|---|---|
| attn_k | blk.0.attn_k.weight | 0.1511 | 0.0986 | 0.9494 | 10.06 |
| attn_o | blk.0.attn_output.weight | 0.1456 | 0.0944 | 0.9517 | 10.25 |
| attn_q | blk.0.attn_q.weight | 0.1509 | 0.0987 | 0.9494 | 10.06 |
| attn_v | blk.0.attn_v.weight | 0.1507 | 0.0984 | 0.9495 | 10.07 |
| embed | token_embd.weight | 0.1739 | 0.1086 | 0.9441 | 9.64 |
| ffn_down | blk.0.ffn_down.weight | 0.1590 | 0.1043 | 0.9464 | 9.82 |
| ffn_gate | blk.0.ffn_gate.weight | 0.1536 | 0.1007 | 0.9483 | 9.97 |
| ffn_up | blk.0.ffn_up.weight | 0.1524 | 0.0994 | 0.9490 | 10.03 |

## Worst 20 tensors by rel_mse

| rank | tensor | role | layer | src | rel_mse | cos | sqnr_dB |
|---|---|---|---|---|---|---|---|
| 1 | token_embd.weight | embed | -1 | Q4_K | 0.1975 | 0.9129 | 7.04 |
| 2 | blk.41.attn_k.weight | attn_k | 41 | Q4_0 | 0.1819 | 0.9180 | 7.40 |
| 3 | blk.23.attn_k.weight | attn_k | 23 | Q4_0 | 0.1756 | 0.9201 | 7.55 |
| 4 | blk.22.attn_v.weight | attn_v | 22 | Q4_0 | 0.1733 | 0.9212 | 7.61 |
| 5 | blk.29.attn_k.weight | attn_k | 29 | Q4_0 | 0.1728 | 0.9213 | 7.63 |
| 6 | blk.53.attn_k.weight | attn_k | 53 | Q4_0 | 0.1719 | 0.9218 | 7.65 |
| 7 | blk.35.attn_k.weight | attn_k | 35 | Q4_0 | 0.1698 | 0.9225 | 7.70 |
| 8 | blk.28.attn_v.weight | attn_v | 28 | Q4_0 | 0.1679 | 0.9233 | 7.75 |
| 9 | blk.37.attn_k.weight | attn_k | 37 | Q4_0 | 0.1678 | 0.9232 | 7.75 |
| 10 | blk.47.attn_k.weight | attn_k | 47 | Q4_0 | 0.1676 | 0.9234 | 7.76 |
| 11 | blk.30.attn_v.weight | attn_v | 30 | Q4_0 | 0.1675 | 0.9233 | 7.76 |
| 12 | blk.57.attn_k.weight | attn_k | 57 | Q4_0 | 0.1661 | 0.9239 | 7.80 |
| 13 | blk.36.attn_v.weight | attn_v | 36 | Q4_0 | 0.1655 | 0.9243 | 7.81 |
| 14 | blk.9.ffn_down.weight | ffn_down | 9 | Q4_0 | 0.1655 | 0.9241 | 7.81 |
| 15 | blk.27.attn_k.weight | attn_k | 27 | Q4_0 | 0.1647 | 0.9243 | 7.83 |
| 16 | blk.45.attn_q.weight | attn_q | 45 | Q4_0 | 0.1645 | 0.9245 | 7.84 |
| 17 | blk.34.attn_v.weight | attn_v | 34 | Q4_0 | 0.1643 | 0.9246 | 7.84 |
| 18 | blk.28.ffn_down.weight | ffn_down | 28 | Q4_0 | 0.1643 | 0.9244 | 7.84 |
| 19 | blk.43.attn_v.weight | attn_v | 43 | Q4_0 | 0.1643 | 0.9254 | 7.84 |
| 20 | blk.29.ffn_down.weight | ffn_down | 29 | Q4_0 | 0.1642 | 0.9245 | 7.85 |

## Best 5 tensors (lowest error)

| tensor | role | layer | rel_mse |
|---|---|---|---|
| blk.0.attn_output.weight | attn_o | 0 | 0.1449 |
| blk.52.attn_v.weight | attn_v | 52 | 0.1457 |
| blk.4.ffn_down.weight | ffn_down | 4 | 0.1458 |
| blk.6.ffn_down.weight | ffn_down | 6 | 0.1470 |
| blk.2.attn_output.weight | attn_o | 2 | 0.1473 |

## Error by role (rel_mse)

| role | n | mean | median | p90 | max |
|---|---|---|---|---|---|
| embed | 1 | 0.1975 | 0.1975 | 0.1975 | 0.1975 |
| attn_k | 60 | 0.1584 | 0.1569 | 0.1676 | 0.1819 |
| attn_v | 50 | 0.1575 | 0.1571 | 0.1643 | 0.1733 |
| ffn_down | 60 | 0.1541 | 0.1537 | 0.1587 | 0.1655 |
| attn_q | 60 | 0.1540 | 0.1535 | 0.1581 | 0.1645 |
| ffn_gate | 60 | 0.1538 | 0.1534 | 0.1581 | 0.1611 |
| ffn_up | 60 | 0.1533 | 0.1527 | 0.1570 | 0.1632 |
| attn_o | 60 | 0.1506 | 0.1503 | 0.1532 | 0.1608 |

## Error by layer band (rel_mse)

| band | n | mean | median | p90 | max |
|---|---|---|---|---|---|
| global | 1 | 0.1975 | 0.1975 | 0.1975 | 0.1975 |
| early(0-4) | 35 | 0.1523 | 0.1525 | 0.1559 | 0.1600 |
| mid(5-54) | 341 | 0.1548 | 0.1538 | 0.1606 | 0.1819 |
| late(55-59) | 34 | 0.1530 | 0.1513 | 0.1591 | 0.1661 |

## Per-role per-layer-band mean rel_mse

| role | early | mid | late | global |
|---|---|---|---|---|
| attn_k | 0.1536 | 0.1590 | 0.1570 | - |
| attn_o | 0.1484 | 0.1507 | 0.1523 | - |
| attn_q | 0.1523 | 0.1543 | 0.1524 | - |
| attn_v | 0.1549 | 0.1578 | 0.1569 | - |
| embed | - | - | - | 0.1975 |
| ffn_down | 0.1524 | 0.1545 | 0.1510 | - |
| ffn_gate | 0.1527 | 0.1541 | 0.1516 | - |
| ffn_up | 0.1521 | 0.1537 | 0.1505 | - |

## Measurement notes

- Each tensor's error is measured on a block-aligned subsample capped at 8M
  elems (>=500k independent 16-elt blocks). rel_mse/cos/sqnr of a per-block
  quantizer are sampling-invariant under block-aligned subsampling; verified
  on `blk.10.ffn_down` (115M elems): full rel_mse 0.15544 vs 8M-cap 0.15585
  (0.3% delta). max_abs_err is the only metric that can mildly underestimate.
- The MSE-optimal floor (per-block scale search) was run on one representative
  tensor per role (subsampled to 16M elems); it is ~24x slower than absmax and
  the role-to-role gap is uniform, so it is not needed on all 411 tensors.
- Structural note (not a bug): 10 layers (5,11,17,23,29,35,41,47,53,59 — the
  global-attention layers with head_count_kv=4 and 16384-wide q/o) have NO
  separate `attn_v.weight` tensor in the GGUF, hence attn_v n=50 not 60. All
  60 attn_k/attn_q are present. 411 quantizable weight tensors total.

## VERDICT — uniform 2-bit is viable; protect only embed + KV projections

The sweep finds **no catastrophic per-tensor cliff**. The whole model lives in a
very tight error band: 410 of 411 tensors fall in rel_mse 0.145–0.182, and the
single highest is token_embd at 0.197. The standard deviation across tensors is
~0.006 (mean 0.1545). This is the signature of a *uniform* quantization budget,
not a "few fragile tensors" problem.

Sensitivity ordering (highest reconstruction error first):
1. **embed (token_embd)** — rel_mse 0.197, the clear #1 and a +0.04 gap over the
   field. It is already kept at Q4_K (not Q4_0) in the shipping GGUF, which is
   the tell that the producers also treat it as sensitive. KEEP >= 4-bit.
2. **attn_k / attn_v (the KV projections)** — the two worst *roles* (mean 0.1584
   / 0.1575) and they own 12 of the top-13 worst tensors. They are the narrowest
   matmuls (n=2048–4096 columns), so each 16-elt block sees a less Gaussian,
   heavier-tailed distribution that NF2's 4 levels fit worst. They are also tiny
   (~1.5–3% of weight bytes) and feed the FP8 KV cache, so protecting them is
   nearly free in bytes but disproportionately valuable for attention fidelity.
3. attn_q / ffn_down / ffn_gate / ffn_up — the bulk, all clustered at ~0.153.
4. **attn_o is the BEST role** (mean 0.1506) — counter to the usual "protect the
   output projection" heuristic; here it is the safest to push to 2-bit.

Layer position barely matters: early 0.1523 / mid 0.1548 / late 0.1530. There is
a faint mid-layer bump (attn_k/v peak around layers 22–53) but no early/late
catastrophe — first and last transformer blocks are NOT special here. So a
layer-based protection scheme buys almost nothing; a **role-based** one does.

### Recommended mixed-precision budget (de-risk M3/M4)
- **token_embd**: keep at >= 4-bit (Q4_K, already the case). +0 GB vs today.
- **attn_k + attn_v**: keep at 4-bit (NVFP4). Cost: these are ~3.4% of the 31B
  weights -> ~0.33 GB extra over all-2-bit. Cheap insurance for attention.
- **everything else** (attn_q, attn_o, ffn_gate/up/down): uniform 2-bit NF2.
- Resulting size: ~9.7 GB all-2-bit baseline + ~0.3 GB KV-proj protection +
  embed already-4-bit -> still well under 11 GB, hitting the F_bytes lever.

### Caveat (the real open risk)
All numbers are the **incremental** 4-bit(Q4_0)->2.5-bit(NF2) loss vs the GGUF
reference, NOT vs true BF16. A per-block-quantizer rel_mse of ~0.15 (SQNR ~8 dB,
cos ~0.92; ~0.10 / ~10 dB / ~0.95 with the MSE-optimal scale) is large in
absolute terms — this is a *weight-reconstruction* proxy, not a quality metric.
Whether it holds Gemma-4-31B *quality* must still be confirmed with a forward
pass (logit-KL / perplexity), which needs the engine or torch and was out of
scope offline. The B2 result de-risks the *shape* of the problem (uniform, no
fragile tensors, protect embed+KV) but does not by itself prove end-task quality.
