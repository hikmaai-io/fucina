<!-- ABOUTME: Raw benchmark inventory for Qwen3.5 exact clean-prefix GDN short-burst work. -->
<!-- ABOUTME: Separates three-start baseline/candidate serving from opt-in attribution runs. -->
# Qwen3.5 burst TTFT2 evidence

Revision base: `6da987a`; branch `perf/qwen35-burst-ttft2`. GB10 `sm_121a`, CUDA 13.
Every GPU run was serialized with `/tmp/fucina_gpu.lock`.

- `baseline/`: three isolated server starts/checkpoint, N=1/2/4/8/16/32 plus 3,500-token probe.
- `candidate/`: same matrix with clean GDN, warmed isolated MoE N=16 protection repeats, and
  three-start median JSON.
- `summary.json`: medians, deltas, and parsed CUDA-event phases.
- `q35{moe,dense}-phase-bench.*`: incumbent admission phases at M=1/2/4/8/16/32.
- `q35{moe,dense}-clean-phase.*`: clean-GDN admission phases.
- `q35{moe,dense}-attribution-*`: scheduler admission shape, wait, engine, and first-decode probe.
- `*-clean-exact.log`: lengths 1..65 and mixed M/state/logit/continuation exact gates.
- `*-protection-gate.log`, `dense32-golden.log`, `qwen-gates.log`, `go-test.log`, `full-build.log`:
  acceptance evidence.
- `moe-engine-rollback-control.log`: known pre-existing MoE B>1 self-consistency failure with the
  optimization disabled; oracle remains 8/8.
