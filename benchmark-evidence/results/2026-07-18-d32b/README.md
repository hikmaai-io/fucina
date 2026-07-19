# D32B — mixer ILP + fresh contemporaneous vLLM (2026-07-18)

Full analysis: `docs/qwen35-d32b.md`. Quiescent GB10 sm_121a, CUDA 13, GPU_CLOCK_MAX 2400.
vLLM (`hellohal2064/vllm-qwen3.5-gb10:latest`) and fucina runs strictly serialized
(docker stop + nvidia-smi quiescence between phases; vLLM ignores the GPU flock).

## Files

- `vllm-q35dense-fresh.json` — fresh vLLM dense 9B sweep (conc 1..32).
- `vllm-q35moe-fresh.json`   — fresh vLLM MoE 35B sweep.
- `fucina-q35dense-d32b.json` — fucina D32B dense 9B sweep (BIGCHUNK=12+MINBLK=4).
- `fucina-q35dense-d32b-rep2.json` — dense N=16/32 repeat (stability).
- `fucina-q35moe-d32b.json`  — fucina D32B MoE 35B sweep.
- `baseline-{dense,moe}-d32b.json` — frozen protection-gate baselines.

## Headline (agg_decode_tps)

| N | dense fucina | dense vLLM | MoE fucina | MoE vLLM |
|---|---|---|---|---|
| 2  | 59.3  | 44.2  | 101.4 | 74.1  |
| 4  | 117.3 | 85.8  | 134.0 | 111.7 |
| 8  | 204.6 | 164.4 | 229.8 | 155.2 |
| 16 | 313.1 | 280.8 | 320.1 | 207.2 |
| 32 | **438.8** | **521.8** | **472.4** | 321.3 |

fucina sweeps all MoE cells + dense N=1..16; dense N=32 is the sole loss (−16% vs fresh
vLLM, was −22% vs 07-11). Protection gate: PASS both models.
