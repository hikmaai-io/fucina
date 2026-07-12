# fucina - Gemma 4 12B inference engine for DGX Spark GB10
# Q4_0 (QAT) and Q8_0 GGUFs, CUDA only (sm_121 Blackwell GB10)

# ─── Toolchain ─────────────────────────────────────────────────────────
# DGX Spark GB10: CUDA 13.0, CUDA arch sm_121, /usr/local/cuda-13
# Go 1.26.4 installed at /usr/local/go/bin
NVCC   := /usr/local/cuda-13/bin/nvcc
CUDA_HOME := /usr/local/cuda-13
GO     := /usr/local/go/bin/go

# ─── Architecture ──────────────────────────────────────────────────────
# sm_121a = Blackwell GB10 (DGX Spark) with arch-specific features enabled
# (FP8/NVFP4 block-scaled MMA, tcgen05). Superset of sm_121; this project is
# GB10-only so the arch-specific cubin is strictly better.
CUDA_ARCH := sm_121a
# BLACKWELL_NATIVE_FP4 gates the native Q4_0 (FP4-class) tiled-MMQ prefill path: the
# projection GEMMs read native Q4_0 weights once via dp4a (no BF16 materialize, no
# per-layer dequant) for small/mid token batches — the agentic suffix-prefill hot path.
# Default ON for GB10; build with BLACKWELL_NATIVE_FP4=0 to fall back to the BF16 path.
BLACKWELL_NATIVE_FP4 ?= 1
NVCCFLAGS := -arch=$(CUDA_ARCH) -O3 -lineinfo --use_fast_math \
             -DBLACKWELL_NATIVE_FP4=$(BLACKWELL_NATIVE_FP4) \
             -Xcompiler -O3 -Xcompiler -pthread \
             --threads 8

CGO_CFLAGS   := -I$(CUDA_HOME)/include
CGO_LDFLAGS  := -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm

.PHONY: all clean test cuda lib libdg fucina fucina-calibrate smoke profile model-plan-test allocation-set-test nvfp4-test e4b-test \
        e4b-load-test e4b-gguf-load-test e4b-fwd-test e4b-gen-test e4b-batch-test e4b-nvfp4-test \
        e4b-bench e4b-all e4b-mtp-load-test e4b-spec-test e4b-spec-stream-test \
        go-test go-test-race go-test-cgo vet lint check paged-kv-test paged-prefix-test \
        gpu-gates qwen35-state-test qwen35-chunk-parity-test qwen35-multiseq-prefill-test qwen35-moe-fp8-engine-test \
        qwen35-detect-test qwen35-load-test qwen35-layer-parity-test qwen35-parity-test qwen35-batch-test qwen35-burst-test \
        qwen35-prefill-test qwen35-longctx-test qwen35-fp8-test qwen35-mtp-test qwen35-moe-fp8-test qwen35-moe-fp8-engine-test qwen36-unsloth-nvfp4-test qwen36-ssd-stream-test qwen35-decode-bench qwen35-fp8-bench fp8-block-test \
        paged-kv-device-test packed-kv-test kv-quant-explore bench tool-bench phase-b-test \
        dg dg-dequant-test dg-forward-test dg-generate

# `make` with no arguments builds everything (CUDA archive + Go binary).
.DEFAULT_GOAL := all

all: lib libdg fucina

# lib builds the CUDA static archive explicitly (also a prerequisite of fucina,
# but exposed so `make all` / `make lib` always produce cuda/libfucina.a even
# if the Go link step is skipped or fails).
lib: cuda/libfucina.a

# ─── CUDA Kernel Library ───────────────────────────────────────────────
# Two-step compilation: device code + link
#
# The objects depend on the Makefile itself: an arch/flag change (e.g.
# sm_121 → sm_121a) MUST force a recompile, otherwise `ar rcs` would bundle
# a stale cubin built with the old flags into libfucina.a.

cuda/gemma4_kernels.o: cuda/gemma4_kernels.cu cuda/gemma4_kernels.cuh cuda/gemma4_config.h \
                       cuda/tensor_types.h cuda/model_plan.h cuda/gemma4_detect.h cuda/paged_kv.h cuda/paged_kv_device.cuh \
                       cuda/paged_prefix.h cuda/safetensors.h cuda/nvfp4.h \
                       cuda/nvfp4_loader.h cuda/nvfp4_gemv.cuh cuda/fp8_block.cuh \
                       cuda/qwen35_fp8_loader.h cuda/qwen35_state.cuh cuda/qwen35_kernels.cuh \
                       cuda/qwen35_jspace.cuh cuda/qwen35_runtime.cuh cuda/qwen35_backend.cuh Makefile
	$(NVCC) $(NVCCFLAGS) -dc -o $@ cuda/gemma4_kernels.cu

# The standalone Gemma-4-E4B engine (runtime dims, Per-Layer Embeddings, KV-sharing)
# is bundled into the same archive so the Go cgo bridge (internal/engine/e4b) links it.
# It is a device runtime alongside the dense engine. -std=c++17 for gemma4_e4b.h.
cuda/e4b_engine.o: cuda/e4b_engine.cu cuda/e4b_engine.h cuda/gemma4_e4b.h \
                   cuda/e4b_ple_fp8.cuh cuda/e4b_gguf.cuh cuda/e4b_nvfp4.cuh \
                   cuda/mmvq.cuh cuda/model_arch.h cuda/safetensors.h Makefile
	$(NVCC) $(NVCCFLAGS) -std=c++17 -dc -o $@ cuda/e4b_engine.cu

cuda/gemma4_kernels_link.o: cuda/gemma4_kernels.o cuda/e4b_engine.o Makefile
	$(NVCC) $(NVCCFLAGS) -dlink -o $@ cuda/gemma4_kernels.o cuda/e4b_engine.o

cuda/libfucina.a: cuda/gemma4_kernels.o cuda/e4b_engine.o cuda/gemma4_kernels_link.o
	ar rcs $@ $^
	@arches="$$($(CUDA_HOME)/bin/cuobjdump --list-elf $@ 2>/dev/null | sed -n 's/.*\.\(sm_[0-9a-z]*\)\.cubin/\1/p' | sort -u)"; \
	echo "libfucina.a cubin arch(es): $$arches"; \
	if [ "$$arches" != "$(CUDA_ARCH)" ]; then \
		echo "libfucina.a: ERROR — expected only $(CUDA_ARCH) cubin, got: $$arches (stale object?)"; \
		exit 1; \
	fi

# ─── Go Binary ──────────────────────────────────────────────────────────
# IMPORTANT: cgo does NOT hash the contents of the `-lfucina` static archive, so
# a plain `go build` happily relinks a STALE binary against an updated
# libfucina.a (this caused weights to be read over unified memory → a ~4s cold
# page-fault charged to prefill). Force a relink every time: remove the old
# binary and rebuild the cgo package with -a. Verify with `strings fucina | grep
# uploading` (must print the device-upload banner).
fucina: cuda/libfucina.a cuda/libdg.a
	rm -f $@
	CGO_CFLAGS="$(CGO_CFLAGS)" \
	CGO_LDFLAGS="$(CGO_LDFLAGS)" \
	$(GO) build -a -ldflags="-s -w" -o $@ ./cmd/fucina/
	@strings $@ | grep -q "uploading.*weights to device" \
		&& echo "fucina: OK — device weight-upload path linked" \
		|| { echo "fucina: ERROR — stale link, upload path missing"; exit 1; }

# Calibration profiler: records per-layer expert heat/weight maps into a
# versioned .imatrix-style JSON sidecar. It links the same CUDA engine archive.
fucina-calibrate: cuda/libfucina.a cuda/libdg.a
	CGO_CFLAGS="$(CGO_CFLAGS)" \
	CGO_LDFLAGS="$(CGO_LDFLAGS)" \
	$(GO) build -a -o $@ ./cmd/fucina-calibrate/

# ─── Standalone CUDA Test (without Go) ──────────────────────────────────
cuda/test_engine: cuda/test_engine.cu cuda/libfucina.a
	$(NVCC) $(NVCCFLAGS) -o $@ $< cuda/libfucina.a -lcudart -lcublas -lcublasLt

# ─── Testing ────────────────────────────────────────────────────────────
# Full test suite: pure-Go unit tests, cgo-dependent tests (needs the CUDA
# archive), then the binary's built-in self-tests on the GPU.
test: fucina go-test go-test-cgo paged-kv-test
	./fucina --test-parser
	CUDA_VISIBLE_DEVICES=0 ./fucina --test-cuda

# ─── Aggregate GPU regression gates ─────────────────────────────────────
# The Qwen3-dense / Qwen3-MoE / spec / prefix / suffix / B=2-row-independence
# correctness gates introduced this session. Each is a standalone target, but
# none was a prerequisite of `test`, so all could be silently forgotten. This
# umbrella chains them so a regression in any cannot pass unnoticed. Requires
# the Qwen3.5 FP8/NVFP4 checkpoints and a GPU; run each under the shared GPU flock.
gpu-gates: qwen35-parity-test qwen35-batch-test qwen35-state-test qwen35-chunk-parity-test qwen35-multiseq-prefill-test qwen35-moe-fp8-engine-test
	@echo "gpu-gates: all Qwen3.5 dense+MoE parity/batch/state/chunk/multiseq-prefill/MoE-engine gates passed"


# ─── Paged-KV allocator unit test (host-only, no GPU) ───────────────────
# Pure integer bookkeeping for the continuous-batching paged KV cache; runs on
# the host so it stays fast and CI-portable. See docs/continuous-batching.md.
paged-kv-test:
	g++ -std=c++17 -O2 -Wall -Wextra cuda/paged_kv_test.cc -o /tmp/fucina_paged_kv_test
	/tmp/fucina_paged_kv_test

# ─── Native Q4_K decode GEMV test (GPU) ─────────────────────────────────
# Validates mmvq_q4_k_kernel (Qwen3 quant path) vs a host full-precision Q4_K
# dequant+dot reference; PASS at cosine >= 0.999 (q8_1 activation quant is the
# only error source). Standalone (header-only mmvq.cuh), no libfucina.a needed.
mmvq-q4k-test:
	$(NVCC) -arch=$(CUDA_ARCH) -O3 -I cuda cuda/test_mmvq_q4_k.cu -o /tmp/fucina_mmvq_q4k_test
	/tmp/fucina_mmvq_q4k_test

# ─── Qwen3 ragged spec-in-batch verify (DSpark step) (GPU) ──────────────
# Asserts gemma4_engine_step_batch_spec: anchor-only (d=0) == step_batch greedy;
# correct draft accepted (run grows), wrong draft rejected with target correction.
# ─── Paged-KV device-kernel test (GPU) ──────────────────────────────────
# Proves block-table indirection (paged_kv_device.cuh) is bit-identical to the
# contiguous KV layout on the read path, and numerically correct for attention.
paged-kv-device-test:
	$(NVCC) -arch=$(CUDA_ARCH) -o /tmp/fucina_paged_kv_device_test \
		cuda/paged_kv_device_test.cu -diag-suppress 550
	/tmp/fucina_paged_kv_device_test

# ─── Packed NVFP4 KV storage test (GPU) ─────────────────────────────────
# Proves the real ~4.5-bit packed layout (pkv_pack_row/pkv_unpack) is bit-identical
# to the FP8 NVFP4 fake-quant — so swapping FP8 storage for packed changes only
# memory, not numerics. See docs/kv-quant-exploration.md.
packed-kv-test:
	$(NVCC) -arch=$(CUDA_ARCH) -o /tmp/fucina_packed_kv_test \
		cuda/packed_kv_test.cu -diag-suppress 550
	/tmp/fucina_packed_kv_test

# ─── Qwen3.5 hybrid (qwen35) arch detection (HOST-only, no GPU) ──────────
# Proves gemma4_detect_from_gguf reads the qwen35.* GGUF metadata into the config
# (dims + the period-4 FULL/LINEAR per-layer attention pattern) WITHOUT the CUDA
# engine. CUDA-free (gemma4_detect.h has its own GGUF reader) → plain g++, no flock.
QWEN35_MODEL ?= /opt/spark/models/Qwen3.5-9B-abliterated-Q4_K_M.gguf
QWEN35_FP8_MODEL ?= /opt/spark/models/models--Qwen--Qwen3.5-9B-FP8
QWEN35_MOE_FP8_MODEL ?= /opt/spark/models/models--Qwen--Qwen3.5-35B-A3B-FP8
QWEN36_UNSLOTH_NVFP4_MODEL ?= /opt/spark/models/unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
qwen35-detect-test:
	g++ -std=c++17 -O2 -Wall -Wextra -Icuda cuda/test_qwen35_detect.cc -o /tmp/fucina_qwen35_detect
	/tmp/fucina_qwen35_detect $(QWEN35_MODEL)

# ─── Qwen3.5 hybrid (qwen35) M1 loader gate (GPU) ───────────────────────
# Loads the qwen35 Q4_K_M GGUF through gemma4_engine_create. The loader dumps each layer
# index + KIND (FULL/LINEAR) with its resolved tensor shapes and validates every shape
# against the arch spec; a missing/misshaped tensor → create returns NULL → this gate fails.
# GPU command — wrap in the shared GB10 flock when running by hand.
qwen35-load-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen35_load.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen35_load \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	flock -w 1200 /tmp/fucina_gpu.lock -c "/tmp/fucina_qwen35_load $(QWEN35_MODEL)"

# ─── Qwen3.5 hybrid (qwen35) M2 per-layer kernel parity gate (GPU) ──────
# Dumps a torch reference (one FULL softmax-attn layer + one GDN gated-deltanet layer from
# the GGUF, dequant Q4_K) with cuda/qwen35_layer_ref.py — CPU-only, no flock — then runs the
# fucina M2 mixer kernels and asserts <1e-2 max-abs-rel error vs torch for BOTH kinds and that
# the GDN chunked-scan output equals the single-step recurrence. GPU run wrapped in the flock.
QWEN35_M2_REF ?= /tmp/qwen35_m2_ref.bin
qwen35-layer-parity-test: lib libdg
	python3 cuda/qwen35_layer_ref.py $(QWEN35_MODEL) $(QWEN35_M2_REF)
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen35_layer_parity.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen35_layer_parity \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	flock -w 1200 /tmp/fucina_gpu.lock -c "/tmp/fucina_qwen35_layer_parity $(QWEN35_M2_REF)"

# ─── Qwen3.5 hybrid (qwen35) M3 single-seq forward greedy parity (GPU) ───
# Drives the FULL qwen35 hybrid stack (24 GDN linear + 8 FULL output-gated softmax-GQA layers,
# carrying GDN state + conv ring + per-FULL-layer KV cache) token-by-token through
# qwen35_forward_greedy and asserts the first 8 greedy continuation ids of "The capital of
# France is" match the llama-simple reference [11751,13,198,57590,369,279,6511,314] (8/8).
qwen35-parity-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen35_parity.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen35_parity \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	flock -w 1200 /tmp/fucina_gpu.lock -c "/tmp/fucina_qwen35_parity $(QWEN35_MODEL)"

# ─── Qwen3.5 hybrid (qwen35) M4 paged-batched + CUDA-graph decode gate (GPU) ───
# Loads the qwen35 Q4_K_M GGUF and runs the continuous-batching ABI (seq_add prefill +
# step_batch decode) through qwen35_batch_selftest, asserting: B-row batched decode is
# bit-identical per row to B=1 (row independence), graph-on == graph-off, and the batched
# path reproduces the M3 single-seq France->Paris 8/8 continuation.
qwen35-batch-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen35_batch.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen35_batch \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	flock -w 1200 /tmp/fucina_gpu.lock -c "/tmp/fucina_qwen35_batch $(QWEN35_MODEL)"

# Diverse-prompt burst-admission gate: LONG diverse prompts (wide tc-prefill path) admitted
# back-to-back and decoded in lockstep must be bit-identical per row to solo runs, prefill
# must be deterministic across repeats, and a 16-seq warmup staircase must not poison the
# engine. Guards the conc-N diverse serving corruption (grouped-GEMM nondeterminism) and the
# runtime-NQ flash-partials overflow — identical-prompt benches mask both.
qwen35-burst-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen35_burst.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen35_burst \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	flock -w 1200 /tmp/fucina_gpu.lock -c "/tmp/fucina_qwen35_burst $(QWEN35_MODEL)"

# ─── Qwen3.5 hybrid per-SLOT state snapshot gate (conversation cache) (GPU) ───
# Saves a slot's hybrid state (GDN S + conv rings + FULL fp32 K/V prefix) mid-decode,
# restores it into a DIFFERENT slot, and asserts the restored continuation is
# bit-identical to both the uninterrupted continuation and a cold full re-prefill.
qwen35-state-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen35_state.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen35_state \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	flock -w 1200 /tmp/fucina_gpu.lock -c "/tmp/fucina_qwen35_state $(QWEN35_MODEL)"

# ─── S1a P0: lossless GDN snapshot/rewind/commit gate (GPU) ───
# The DFlash (1+K) verification prerequisite. Asserts that for every accepted length j in 0..K,
# snapshot -> speculatively advance K -> commit(j) leaves the slot's full hybrid state (GDN
# recurrent slab + FULL-layer K/V) BYTE-IDENTICAL to j sequential single-token decodes, and the
# committed next token matches the sequential reference. j=0 (pure rewind) and j=K (full accept)
# are included. Byte-identical, not within-a-bound: commit re-runs the exact sequential path.
qwen35-gdn-rollback-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen35_gdn_rollback.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen35_gdn_rollback \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	flock -w 1200 /tmp/fucina_gpu.lock -c "/tmp/fucina_qwen35_gdn_rollback $(QWEN35_MODEL)"

# ─── Qwen3.5 hybrid (qwen35) P1 BATCHED single-pass prefill gate (GPU) ───
# Drives the continuous-batching ABI (seq_add → qwen35_prefill_batched, then step_batch) and
# asserts: (1) the integrated batched prefill reproduces the qwen35_forward_greedy oracle's
# France->Paris 8/8 continuation; (2) on a 512-token prompt the batched prefill's first token
# is bit-identical to the token-by-token path; (3) prints the prefill TTFT slow-vs-fast (the
# token-by-token 149 s path vs the batched path) — expect a LARGE drop.
qwen35-prefill-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen35_prefill.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen35_prefill \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	flock -w 1800 /tmp/fucina_gpu.lock -c "/tmp/fucina_qwen35_prefill $(QWEN35_MODEL)"

# ─── Qwen3.5 hybrid (qwen35) P3 LONG-CONTEXT argmax parity vs llama.cpp (GPU) ───
# The Gated-DeltaNet fp32 recurrent state accumulates over every position, so long-context argmax
# flips are exactly where state drift surfaces. Drives the INTEGRATED continuous-batching path
# (seq_add → qwen35 batched prefill, then step_batch decode) on a ~1k-token AND a ~4k-token
# natural-text prompt (committed ids in cuda/qwen35_longctx_{1k,4k}.ids) and asserts the greedy
# continuation matches the pinned llama.cpp reference over 40 tokens (40/40, both lengths).
# Regenerate the prompt ids + pinned references with the validated libllama harness:
#   scratchpad/make_text.py | llama-tokenize --ids --no-bos  → cuda/qwen35_longctx_*.ids
#   scratchpad/llama_ref_greedy <gguf> <ids> 40              → REF_1K / REF_4K in the test
qwen35-longctx-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen35_longctx.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen35_longctx \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	flock -w 1800 /tmp/fucina_gpu.lock -c "/tmp/fucina_qwen35_longctx $(QWEN35_MODEL) \
		$(CURDIR)/cuda/qwen35_longctx_1k.ids $(CURDIR)/cuda/qwen35_longctx_4k.ids"

# ─── Qwen3.5 chunked-prefill CONTINUATION (base>0) parity + timing gate (GPU) ───
# One-shot seq_add vs seq_open + seq_prefill_chunk(1024, rest) on a 2400-token natural-text
# prompt (first ids of qwen35_longctx_4k.ids), 24 greedy continuation tokens each — run with
# BOTH the scalar continuation attention (g_fucina_q35_scalar_cont_attn=1) and the default
# tensor-core one. Asserts the TC continuation tracks one-shot at least as well as the scalar
# path (first token + 25-token agreement) and prints the continuation-chunk wall time of both.
qwen35-chunk-parity-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen35_chunk_parity.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen35_chunk_parity \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	flock -w 1800 /tmp/fucina_gpu.lock -c "/tmp/fucina_qwen35_chunk_parity \
		$(QWEN35_MOE_FP8_MODEL) $(CURDIR)/cuda/qwen35_longctx_4k.ids"

# ─── Qwen3.5 FP8 block-quant decode GEMV standalone validation (GPU) ─────
# Validates cuda/fp8_block.cuh (DeepSeek block-fp8 decode GEMV) vs a host dequant+dot reference
# at cosine >= 0.999 — the kernel the M5 FP8 model forward drives for every projection.
fp8-block-test:
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_fp8_block.cu \
		-o /tmp/fucina_fp8_block \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	flock -w 1200 /tmp/fucina_gpu.lock -c "/tmp/fucina_fp8_block"

# ─── Qwen3.5 hybrid (qwen35) M5 FP8 safetensors forward greedy parity (GPU) ───
# Loads the OFFICIAL Qwen3.5-9B FP8 checkpoint (DeepSeek block-fp8 safetensors, text path) and
# drives the hybrid stack token-by-token through qwen35_fp8_forward_greedy (fp8_block decode GEMV
# for the projections), asserting the first 8 greedy continuation ids of "The capital of France is"
# match the torch FP8 oracle [11751,13,198,760,6511,314,9338,369] (8/8). Regenerate the oracle ids
# with: $(PYTHON) cuda/qwen35_fp8_ref.py $(QWEN35_FP8_MODEL)
qwen35-fp8-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen35_fp8.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen35_fp8 \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	flock -w 1200 /tmp/fucina_gpu.lock -c "/tmp/fucina_qwen35_fp8 $(QWEN35_FP8_MODEL)"

# ─── Session persistence restart gate (GPU) ─────────────────────────────────
# Saves a long Qwen3.5 hybrid REPL session, restarts the process, /loads it and
# asserts the continuation re-prefills ONLY the new turn (the saved prefix —
# including the GDN recurrent state — costs zero prefill tokens).
session-restart-test: fucina
	flock -w 1800 /tmp/fucina_gpu.lock -c "scripts/test_session_restart.sh $(QWEN35_FP8_MODEL)"

# ─── Qwen3.5 FP8-9B served through the REAL batched engine (not the B=1 oracle) (GPU) ───
# Loads the official Qwen3.5-9B FP8 checkpoint via gemma4_engine_create (FORMAT_FP8_BLOCK loader:
# every FP8 proj → d_weights + per-128 block-scale table; norms/conv/a_log → f32; in_a/in_b + embed
# + lm_head → Q8_0) and asserts (A) the engine's token-by-token greedy continuation of "The capital
# of France is" matches the torch FP8 oracle 8/8, AND (B) the batched self-test (row independence +
# graph determinism + batched==token-by-token) passes on the same engine.
qwen35-fp8-engine-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen35_fp8_engine.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen35_fp8_engine \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	flock -w 1200 /tmp/fucina_gpu.lock -c "/tmp/fucina_qwen35_fp8_engine $(QWEN35_FP8_MODEL)"

# ─── Qwen3.5-35B-A3B MoE FP8 served through the REAL batched engine (not the B=1 oracle) (GPU) ───
# Same gate as qwen35-fp8-engine-test but for the qwen3_5_moe checkpoint: runtime H=2048/NKV=2,
# per-layer 256-expert FP8 slabs (grouped FP8 GEMM), shared expert, softmax-top8-renorm router.
qwen35-multiseq-prefill-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen35_multiseq_prefill.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen35_multiseq_prefill \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	flock -w 1800 /tmp/fucina_gpu.lock -c "/tmp/fucina_qwen35_multiseq_prefill $(if $(MODEL),$(MODEL),$(QWEN35_MOE_FP8_MODEL)) $(QWEN35_FP8_MODEL)"

qwen35-moe-fp8-engine-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen35_moe_fp8_engine.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen35_moe_fp8_engine \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	flock -w 1800 /tmp/fucina_gpu.lock -c "/tmp/fucina_qwen35_moe_fp8_engine $(QWEN35_MOE_FP8_MODEL)"

# Unsloth compressed-tensors mixed FP8/NVFP4 gate. Reuses the MoE engine oracle because it
# exercises the real continuous-batching prefill, B-row decode, CUDA graph, and B=1 parity paths.
qwen36-unsloth-nvfp4-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen35_moe_fp8_engine.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen36_unsloth_nvfp4 \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	flock -w 1800 /tmp/fucina_gpu.lock -c "/tmp/fucina_qwen36_unsloth_nvfp4 $(QWEN36_UNSLOTH_NVFP4_MODEL)"

# Artificial low-residency gate: only 64 transformed experts may occupy device slots while the
# immutable 16.88-GiB store is read from SSD. The same oracle/batch test proves slot remapping,
# chunk fallback, checksums, and graph-off streaming preserve exact generated tokens.
qwen36-ssd-stream-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen35_moe_fp8_engine.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen36_ssd_stream \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	flock -w 2400 /tmp/fucina_gpu.lock -c 'store=/tmp/fucina-qwen36-experts.$$$$.bin; trap "rm -f $$store" EXIT; FUCINA_EXPERT_STREAM_SSD=$$store FUCINA_EXPERT_STREAM_SLOTS=64 /tmp/fucina_qwen36_ssd_stream $(QWEN36_UNSLOTH_NVFP4_MODEL)'

# ─── Qwen3.5 hybrid (qwen35) M6 single-MTP draft head + LOSSLESS spec (GPU) ───
# Loads the 22 mtp.* tensors (FP8 checkpoint only) and asserts the MTP-drafted speculative decode
# (qwen35_fp8_spec_greedy) emits the IDENTICAL continuation to plain greedy (qwen35_fp8_forward_greedy
# = M5-proven backbone) — i.e. spec is LOSSLESS — with draft accept-rate > 0. Torch oracle for the
# MTP math + accept rate: $(PYTHON) cuda/qwen35_mtp_ref.py $(QWEN35_FP8_MODEL)
qwen35-mtp-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen35_mtp.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen35_mtp \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	flock -w 1200 /tmp/fucina_gpu.lock -c "/tmp/fucina_qwen35_mtp $(QWEN35_FP8_MODEL)"

# ─── Qwen3.5-35B-A3B MoE (qwen3_5_moe) P6 FP8 safetensors forward greedy parity (GPU) ───
# Loads the OFFICIAL Qwen3.5-35B-A3B-FP8 checkpoint (DeepSeek block-fp8 safetensors, text path) and
# drives the hybrid stack token-by-token through qwen35_moe_fp8_forward_greedy (same GDN+FULL mixer
# as the 9B dense path, hidden 2048 / 2 KV heads, dense MLP replaced by the 256-expert top-8
# softmax-renorm mixture + sigmoid-gated shared expert), asserting the first 8 greedy continuation
# ids of "The capital of France is" match the torch MoE oracle [11751,13,198,760,6511,314,9338,369]
# (8/8). Regenerate the oracle ids with: $(PYTHON) cuda/qwen35_moe_fp8_ref.py $(QWEN35_MOE_FP8_MODEL)
qwen35-moe-fp8-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen35_moe_fp8.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen35_moe_fp8 \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	flock -w 1800 /tmp/fucina_gpu.lock -c "/tmp/fucina_qwen35_moe_fp8 $(QWEN35_MOE_FP8_MODEL)"

# ─── Qwen3.5 hybrid (qwen35) P5 served-path decode tok/s bench (GPU, not a gate) ───
# Times NSTEP single-token decode steps through the SERVED step_batch path
# (qwen35_decode_multiseq_body / CUDA-graph). Used to measure the P5 in_qkv Q5_K→Q8_0
# native-GEMV win (8.0 -> 30.5 tok/s on Qwen3.5-9B Q4_K_M). Override NSTEP=N on the CLI.
QWEN35_BENCH_NSTEP ?= 128
qwen35-decode-bench: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen35_decode_bench.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen35_decode_bench \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	flock -w 1800 /tmp/fucina_gpu.lock -c "/tmp/fucina_qwen35_decode_bench $(QWEN35_MODEL) $(QWEN35_BENCH_NSTEP)"

# ─── Qwen3.5 FP8 same-checkpoint single-stream bench (GPU) ───────────────
# P7 apples-to-apples anchor: single-stream decode tok/s + prefill latency of the FP8
# reference path on the SAME official Qwen3.5-9B-FP8 checkpoint vLLM serves. Isolates
# fucina's per-token FP8 compute from the Q4_K_M-vs-FP8 quant gap of the served comparison.
# NOTE: the FP8 path is the token-by-token reference oracle (no CUDA-graph/batching); the
# optimized fucina server runs the GGUF. Pair with scripts/bench_serving.py (HTTP harness).
qwen35-fp8-bench: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen35_fp8_bench.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen35_fp8_bench \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	flock -w 1800 /tmp/fucina_gpu.lock -c "/tmp/fucina_qwen35_fp8_bench $(QWEN35_FP8_MODEL)"

# ─── Qwen3 dense numeric parity vs llama.cpp (GPU) ──────────────────────
# Feeds llama.cpp's input ids for "The capital of France is" through fucina's
# arch-driven multiseq path and asserts the greedy continuation matches
# llama.cpp's [12095,13,576,6722,315,15344,374,21718] (Qwen3-8B Q4_K_M).
# ─── Qwen3-MoE (qwen3moe) numeric parity vs llama.cpp (GPU) ─────────────
# Loads Qwen3-30B-A3B (sparse MoE: 128 experts, top-8) and asserts fucina's greedy
# continuation of "The capital of France is" matches llama.cpp's raw-completion reference
# [12095,13,576,6722,315,279,3639,4180] token-for-token (8/8). Override the model with MODEL=.
# ─── Qwen3-MoE DSpark spec losslessness on the MoE FFN (GPU) ────────────
# Drives Qwen3-30B-A3B through the spec-decode verify path and asserts: (A) depth-0 spec is
# byte-identical to plain step_batch (lossless), (B) a correct draft is accepted and grows the
# run, (C) a wrong draft is rejected and corrected. Guards the deterministic per-token MoE
# expert reduce (no atomicAdd scatter) that makes the batched MoE FFN bit-reproducible.
# ─── Qwen3-MoE B=2 cross-sequence row-independence (GPU) ────────────────
# The atomicAdd-scatter nondeterminism bug only surfaced at B>=2 with DIVERGENT rows;
# single-seq parity missed it entirely. This batches two sequences at different
# trajectory positions in ONE step_batch(B=2) and asserts each row continues
# independently (row0==single-seq next, row1==trajectory[3], rows differ) — the
# direct gate for the deterministic dg_moe_route_inv/dg_moe_reduce combine.
# ─── Qwen3 cross-request prefix cache losslessness (GPU) ────────────────
# Proves cache-served requests (shared-prefix reuse) produce a greedy token stream
# bit-identical to a cold request, sequentially and concurrently. See paged_prefix.h.
# ─── Stage 9: base-offset compute-bound suffix prefill losslessness (GPU) ─
# Proves the GEMM suffix prefill (paged_prefill_qwen3 base>0) is lossless: a sequence built
# prefix-then-suffix (cross-request prefix-cache adoption) yields a greedy stream bit-identical
# to a one-shot prefill, on BOTH Qwen3-8B dense and Qwen3-30B-A3B MoE. Args: <dense> [<moe>].
# ─── Stage 18: FUSED prefill+decode losslessness (GPU) ──────────────────
# Proves gemma4_engine_step_batch_fused is lossless on BOTH halves: a sequence prefilled via the
# FUSED path while co-batched with N>=2 unrelated DECODE rows yields a first token + >=16-token
# greedy continuation byte-identical to a STANDALONE seq_open+seq_prefill_chunk (LOSSLESS-PREFILL),
# AND the co-batched decode rows are byte-identical to a plain step_batch of the same rows without
# the prefill (LOSSLESS-DECODE). Runs on Qwen3-8B dense and Qwen3-30B-A3B MoE. Args: <dense> [<moe>].
# ─── Qwen3 decode throughput (GPU) ──────────────────────────────────────
# ─── KV-quant exploration (host, Phase 6) ───────────────────────────────
# Offline comparison of KV-cache quant codecs (FP8 / per-token FP8 / NVFP4 /
# TurboQuant-MSE). Decides whether to move KV off flat FP8. Host-only, no engine
# link. Optional args: `make kv-quant-explore ARGS="<n_outlier> <outlier_std>"`.
# See docs/kv-quant-exploration.md.
kv-quant-explore:
	g++ -std=c++17 -O2 -Wall -Wextra cuda/kv_quant_explore.cc -o /tmp/fucina_kv_quant_explore -lm
	/tmp/fucina_kv_quant_explore $(ARGS)

# ─── Correctness + performance smoke (GB10) ─────────────────────────────
# Runs the engine self-tests (batch==single, sampling) + greedy byte-identity,
# then reports prefill/decode throughput. Correctness gates are hard (non-zero
# exit on failure); perf is reported. Override the model with MODEL=/path.gguf.
bench: fucina
	MODEL=$(if $(MODEL),$(MODEL),model.gguf) scripts/bench.sh

test-vectors: fucina
	./fucina --test-vectors tests/vectors/official.vec

# ─── Quick smoke test ───────────────────────────────────────────────────
smoke: fucina
	./fucina --prompt "Hello, world!" --predict 32 --temp 0

# ─── LoRA smoke test ────────────────────────────────────────────────────
lora-smoke: fucina
	./fucina --model model.gguf --lora-scaled lora.gguf \
		--prompt "Test prompt" --predict 32 --temp 0

# ─── Profiling ──────────────────────────────────────────────────────────
profile: fucina
	nsys profile -o fucina_profile -t cuda,nvtx ./fucina \
		--prompt "Write a haiku about CUDA." --predict 128 --temp 0

# ─── Go quality / unit tests (no CUDA required) ───────────────────────────
# NOTE: ./internal/engine/cuda and ./cmd/... are intentionally excluded from
# the pure-Go targets below — the cgo engine package links against
# cuda/libfucina.a, so `go test`/`go vet` there fails to build/link unless the
# CUDA archive has been compiled with nvcc on a GB10 box. The server,
# tokenizer, sampler and chat packages are pure Go and run anywhere.
GO_TEST_PKGS := ./internal/server/ ./internal/server/batch/ ./internal/tokenizer/ ./internal/sampler/ ./internal/chat/

# cgo-dependent Go tests (cmd/fucina: CLI parsing tests). Requires
# cuda/libfucina.a to link, hence the `lib` prerequisite.
GO_TEST_CGO_PKGS := ./cmd/...

go-test-cgo: lib libdg
	CGO_CFLAGS="$(CGO_CFLAGS)" \
	CGO_LDFLAGS="$(CGO_LDFLAGS)" \
	$(GO) test $(GO_TEST_CGO_PKGS) -count=1

go-test:
	$(GO) test $(GO_TEST_PKGS) -count=1

phase-b-test:
	PYTHONPATH=scripts python3 -m unittest scripts/test_phase_b.py

go-test-race:
	$(GO) test $(GO_TEST_PKGS) -race -count=1

# vet: restricted to non-cgo packages. `go vet ./cmd/...` is avoided because
# cmd/fucina pulls in the cgo engine package which cannot link without
# libfucina.a; vetting it here would fail on a CUDA-less machine.
vet:
	$(GO) vet $(GO_TEST_PKGS)

lint:
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run $(GO_TEST_PKGS); \
	else \
		echo "lint: golangci-lint not installed — skipping"; \
	fi

# Convenience: the full pure-Go correctness bar — static analysis, lint, and the
# unit tests under the race detector. This is the gate CI mirrors; keep them in
# sync. `lint` soft-skips if golangci-lint is absent locally, but CI installs it
# so a missing-tool skip can never hide a real finding on main.
check: vet lint go-test-race

# ─── DiffusionGemma (26B-A4B text-diffusion MoE) engine ───────────────────
# Separate from the autoregressive gemma4 engine: own kernels, own forward, own
# static archive (cuda/libdg.a) consumed by the internal/engine/diffusion cgo
# package. DG_NVCCFLAGS omits --use_fast_math (the forward/generation were
# validated vs llama.cpp / coherent text with default-math transcendentals).
DG_GGUF ?= ./models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf
DG_NVCCFLAGS := -arch=$(CUDA_ARCH) -O3 -lineinfo -Xcompiler -O3 -Xcompiler -pthread --threads 8
# CUTLASS (vendored under flashinfer) for the grouped NVFP4 expert GEMM (cuda/dg_fp4_moe.cu).
# Auto-detected: import flashinfer if it's on the default python, else glob the venv. Override
# with `make all CUTLASS_DIR=/path/to/cutlass` for a checkout elsewhere.
CUTLASS_DIR ?= $(shell \
	python3 -c "import flashinfer,os;print(os.path.join(os.path.dirname(flashinfer.__file__),'data','cutlass'))" 2>/dev/null \
	|| ls -d $(HOME)/.venv/lib/python*/site-packages/flashinfer/data/cutlass 2>/dev/null | head -1)
DG_FP4_NVCCFLAGS := -arch=$(CUDA_ARCH) -std=c++17 -O3 -lineinfo --expt-relaxed-constexpr \
	--expt-extended-lambda -DCUTLASS_ARCH_MMA_SM120_SUPPORTED=1 \
	-I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include -Xcompiler -O3 -Xcompiler -pthread

cuda/diffusion_gemma_kernels.o: cuda/diffusion_gemma_kernels.cu cuda/diffusion_gemma_kernels.cuh Makefile
	$(NVCC) $(DG_NVCCFLAGS) -dc -o $@ cuda/diffusion_gemma_kernels.cu

cuda/diffusion_gemma_engine.o: cuda/diffusion_gemma_engine.cu cuda/diffusion_gemma_engine.h cuda/diffusion_gemma_kernels.cuh Makefile
	$(NVCC) $(DG_NVCCFLAGS) -dc -o $@ cuda/diffusion_gemma_engine.cu

cuda/dg_fp4_moe.o: cuda/dg_fp4_moe.cu Makefile
	@test -f "$(CUTLASS_DIR)/include/cutlass/cutlass.h" || { \
	  echo "ERROR: CUTLASS not found (CUTLASS_DIR='$(CUTLASS_DIR)'). The DiffusionGemma NVFP4"; \
	  echo "  MoE needs CUTLASS. Install flashinfer (pip install flashinfer) or pass a checkout:"; \
	  echo "  make all CUTLASS_DIR=/path/to/cutlass"; exit 1; }
	$(NVCC) $(DG_FP4_NVCCFLAGS) -dc -o $@ cuda/dg_fp4_moe.cu

cuda/libdg_link.o: cuda/diffusion_gemma_kernels.o cuda/diffusion_gemma_engine.o cuda/dg_fp4_moe.o Makefile
	$(NVCC) $(DG_NVCCFLAGS) -dlink -o $@ cuda/diffusion_gemma_kernels.o cuda/diffusion_gemma_engine.o cuda/dg_fp4_moe.o

cuda/libdg.a: cuda/diffusion_gemma_kernels.o cuda/diffusion_gemma_engine.o cuda/dg_fp4_moe.o cuda/libdg_link.o
	ar rcs $@ $^

libdg: cuda/libdg.a

# Standalone DiffusionGemma test/dev binaries (no Go).
dg-dequant-test:
	$(NVCC) -O2 -arch=$(CUDA_ARCH) cuda/diffusion_gemma_kernels.cu cuda/test_diffusion_dequant.cu -o /tmp/dg_dequant_test
dg-forward-test:
	$(NVCC) -O2 -arch=$(CUDA_ARCH) cuda/diffusion_gemma_kernels.cu cuda/test_diffusion_forward.cu -lcublas -o /tmp/dg_forward_test
dg-generate:
	$(NVCC) -O2 -arch=$(CUDA_ARCH) cuda/diffusion_gemma_kernels.cu cuda/dg_generate.cu -lcublas -o /tmp/dg_gen
dg-matmul-test:
	$(NVCC) -O2 -arch=$(CUDA_ARCH) cuda/diffusion_gemma_kernels.cu cuda/test_diffusion_matmul.cu -lcublas -o /tmp/dg_matmul_test
dg-moe-stream:
	$(NVCC) -O2 -arch=$(CUDA_ARCH) cuda/diffusion_gemma_kernels.cu cuda/test_diffusion_moe_stream.cu -lcublas -o /tmp/dg_moe_stream
dg-moe-grouped:
	$(NVCC) -O2 -arch=$(CUDA_ARCH) cuda/diffusion_gemma_kernels.cu cuda/test_diffusion_moe_grouped.cu -lcublas -o /tmp/dg_moe_grouped
dg-bf16-test:
	$(NVCC) -O2 -arch=$(CUDA_ARCH) cuda/test_diffusion_bf16.cu -lcublas -o /tmp/dg_bf16
dg-sampler-test:
	$(NVCC) -O2 -arch=$(CUDA_ARCH) cuda/diffusion_gemma_kernels.cu cuda/test_diffusion_sampler.cu -o /tmp/dg_samp
dg: dg-dequant-test dg-forward-test dg-generate dg-matmul-test dg-moe-stream dg-moe-grouped dg-bf16-test dg-sampler-test

# NVFP4 FP4-tensor-core probes (FUCINA_FP4 / DG_FP4 development harnesses).
fp4-probe:        # cuBLASLt FP4 support matrix (NVFP4 vec16 vs MXFP4 vec32, FP8/BF16 controls)
	$(NVCC) -O3 -arch=$(CUDA_ARCH) cuda/test_fp4_probe2.cu -lcublasLt -o /tmp/fp4_probe2
fp4-gemm-test:    # NVFP4 scale-swizzle validation + speed/accuracy vs BF16 on real shapes
	$(NVCC) -O3 -arch=$(CUDA_ARCH) cuda/test_fp4_gemm.cu -lcublasLt -lcublas -o /tmp/fp4_gemm
# Grouped NVFP4 expert-GEMM (DiffusionGemma MoE) — CUTLASS sm120, needs the libdg objects.
dg-fp4-grouped:   # CUTLASS grouped FP4 microbench (speed vs dp4a); run: /tmp/dg_fp4_grouped bench
	$(NVCC) -std=c++17 -O3 -arch=$(CUDA_ARCH) --expt-relaxed-constexpr --expt-extended-lambda \
		-DCUTLASS_ARCH_MMA_SM120_SUPPORTED=1 -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include \
		cuda/test_dg_fp4_grouped.cu -o /tmp/dg_fp4_grouped
dg-fp4-parity: cuda/diffusion_gemma_kernels.o cuda/dg_fp4_moe.o   # FP4 grouped vs dequant ref on real weights
	$(NVCC) -std=c++17 -O3 -arch=$(CUDA_ARCH) -dc --expt-relaxed-constexpr --expt-extended-lambda \
		-DCUTLASS_ARCH_MMA_SM120_SUPPORTED=1 -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include \
		cuda/test_dg_fp4_parity.cu -o /tmp/parity.o
	$(NVCC) -arch=$(CUDA_ARCH) -dlink /tmp/parity.o cuda/diffusion_gemma_kernels.o cuda/dg_fp4_moe.o -o /tmp/parity_link.o
	$(NVCC) -arch=$(CUDA_ARCH) /tmp/parity.o cuda/diffusion_gemma_kernels.o cuda/dg_fp4_moe.o /tmp/parity_link.o -lcublas -lcublasLt -o /tmp/dg_fp4_parity
	@echo "run: /tmp/dg_fp4_parity $(DG_GGUF)"
fp4: fp4-probe fp4-gemm-test

# Host-only immutable model-plan validation and deterministic serialization.
model-plan-test:
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Icuda cuda/model_plan_test.cc -o /tmp/model_plan_test && /tmp/model_plan_test

# Host-only S2b graph-key helpers (shape triple + dominance dispatch + decode-first ordering).
qwen35-graph-key-test:
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Icuda -x c++ cuda/qwen35_graph_key_test.cc -o /tmp/q35_graph_key_test && /tmp/q35_graph_key_test

# Host-only DFlash counter-RNG + rejection-sampler oracle (P1 of S1a). Self-contained, no model.
# Determinism, domain/position independence, uniform range, greedy + probabilistic rejection math.
qwen35-dflash-rng-test:
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Icuda cuda/qwen35_dflash_rng_test.cc -o /tmp/dflash_rng && /tmp/dflash_rng

# Host-only DFlash draft loader schema (P2 of S1a): config geometry + tensor validation and
# hostile-input rejection (mismatched shapes/dtypes/vocab) BEFORE any CUDA allocation. No model.
qwen35-dflash-loader-test:
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Icuda cuda/qwen35_dflash_loader_test.cc -o /tmp/dflash_ld && /tmp/dflash_ld

# Host-only DFlash shape/lookahead planner + enable/concurrency gate (S1a): (1+K) verify shapes,
# S2 spec graph key, N+1 KV lookahead, default-off + conservative concurrency gating. No model.
qwen35-dflash-plan-test:
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Icuda cuda/qwen35_dflash_plan_test.cc -o /tmp/dflash_plan && /tmp/dflash_plan

# Host-only DFlash verify->commit assembly (P4 orchestration): maps a rejection result to the exact
# P0 commit token sequence + next-step input token + emitted count. Weights-free, deterministic.
qwen35-dflash-commit-test:
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Icuda cuda/qwen35_dflash_commit_test.cc -o /tmp/dflash_commit && /tmp/dflash_commit

# Host-only DFlash verify-pipeline integration (S1a): composes planner -> shared-key draft sampling
# -> rejection -> commit assembly end to end on synthetic logits (the seams around the two device
# forwards). Weights-free; proves the deterministic glue before the draft forward exists.
qwen35-dflash-pipeline-test:
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Icuda cuda/qwen35_dflash_pipeline_test.cc -o /tmp/dflash_pipe && /tmp/dflash_pipe

# CUDA<->CPU parity for the DFlash RNG + rejection sampler (P1 of S1a). Self-contained, no model;
# runs the shared __host__ __device__ header on-GPU and asserts bit-identical results vs the host.
qwen35-dflash-parity-test:
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen35_dflash_parity.cu -o /tmp/dflash_parity \
		-lcudart -lcuda
	flock -w 600 /tmp/fucina_gpu.lock -c "/tmp/dflash_parity"

allocation-set-test:
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Icuda cuda/device_allocation_set_test.cc -o /tmp/allocation_set_test && /tmp/allocation_set_test

# NVFP4 safetensors loader unit tests (host + decode-kernel parity). Self-contained, no model.
nvfp4-test: model-plan-test allocation-set-test
	g++ -std=c++17 -O2 -Wall -Wextra cuda/safetensors_test.cc   -o /tmp/st_test     && /tmp/st_test
	g++ -std=c++17 -O2 -Wall -Wextra cuda/nvfp4_test.cc          -o /tmp/nvfp4_test  && /tmp/nvfp4_test
	g++ -std=c++17 -O2 -Wall -Wextra cuda/nvfp4_loader_test.cc   -o /tmp/nvfp4_ld    && /tmp/nvfp4_ld
	g++ -std=c++17 -O2 -Wall -Wextra cuda/qwen35_fp8_loader_test.cc -o /tmp/q35fp8_ld && /tmp/q35fp8_ld
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 cuda/test_nvfp4_gemv.cu -o /tmp/nvfp4_gemv && /tmp/nvfp4_gemv

# Gemma-4-E4B foundation: runtime config detection + FP8 Per-Layer-Embedding
# "index" codec, validated against the real BF16 checkpoint (cosine gate).
# Pass MODEL_DIR=... to point at a different E4B snapshot.
e4b-test:
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_e4b_ple_fp8.cu -o /tmp/e4b_ple_test && /tmp/e4b_ple_test $(MODEL_DIR)

# E4B engine BF16 weight loader: load the real checkpoint, report residency.
e4b-load-test:
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_e4b_load.cu cuda/e4b_engine.cu -o /tmp/e4b_load -lcudart -lcublas && /tmp/e4b_load $(MODEL_DIR)

# E4B GGUF (Q4_0-QAT/Q6_K) loader: load the real GGUF through the same engine
# (dequant → BF16 + FP8 PLE), check dims/residency, run a forward sanity
# (finite logits, in-range argmax, no illegal access) + optional BF16 parity.
# Pass GGUF=... to point at a different GGUF; MODEL_DIR=... for the BF16 reference.
e4b-gguf-load-test:
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_e4b_gguf_load.cu cuda/e4b_engine.cu -o /tmp/e4b_gguf_load -lcudart -lcublas && /tmp/e4b_gguf_load $(GGUF) $(MODEL_DIR)

# E4B forward pass validated against an HF reference dump (/tmp/e4b_ref.bin).
e4b-fwd-test:
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_e4b_forward.cu cuda/e4b_engine.cu -o /tmp/e4b_fwd -lcudart -lcublas && /tmp/e4b_fwd $(MODEL_DIR)

# E4B incremental decode (KV cache) validated against HF greedy (/tmp/e4b_gen_ref.bin).
e4b-gen-test:
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_e4b_generate.cu cuda/e4b_engine.cu -o /tmp/e4b_gen -lcudart -lcublas && /tmp/e4b_gen $(MODEL_DIR)

# E4B continuous batching: B concurrent sequences == independent decode.
e4b-batch-test:
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_e4b_batch.cu cuda/e4b_engine.cu -o /tmp/e4b_batch -lcudart -lcublas && /tmp/e4b_batch $(MODEL_DIR)

# E4B MTP increment 1: load the gemma4-assistant draft head + verify residency/dims (no
# drafter forward yet). GGUF=<e4b base q4_0 gguf> MTP=<assistant gguf>. See docs/e4b-mtp-plan.md.
e4b-mtp-load-test: lib
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_e4b_mtp_load.cu cuda/libfucina.a -o /tmp/e4b_mtp_load -lcudart -lcublas -lcublasLt -lcuda -lstdc++ -lm && /tmp/e4b_mtp_load $(GGUF) $(MTP)

# E4B MTP increments 3+4: greedy speculative decode via the draft head. DECISIVE GATE —
# compares e4b_engine_generate_greedy (baseline) vs e4b_engine_generate_spec_greedy (assistant
# loaded) for the same prompt and asserts BYTE-IDENTICAL token ids (greedy spec is lossless).
# GGUF=<e4b base q4_0 gguf> MTP=<assistant gguf>. See docs/e4b-mtp-plan.md.
e4b-spec-test: lib
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_e4b_spec.cu cuda/libfucina.a -o /tmp/e4b_spec -lcudart -lcublas -lcublasLt -lcuda -lstdc++ -lm && /tmp/e4b_spec $(GGUF) $(MTP)

# E4B MTP increment 5: the SERVER continue/streaming spec path. Prefills then drives
# e4b_engine_spec_stream (continue from live KV, h0 re-derived from the last history token,
# per-token emit callback) and asserts BYTE-IDENTICAL to plain greedy + that the callback saw
# exactly the returned tokens in order. GGUF=<base> MTP=<assistant>. See docs/e4b-mtp-plan.md.
e4b-spec-stream-test: lib
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_e4b_spec_stream.cu cuda/libfucina.a -o /tmp/e4b_spec_stream -lcudart -lcublas -lcublasLt -lcuda -lstdc++ -lm && /tmp/e4b_spec_stream $(GGUF) $(MTP)

# E4B NVFP4 weight-path foundation: quantizer + tuned decode GEMV, validated vs the
# host dequant oracle (kernel correctness) and full precision (FP4 SNR), with bandwidth.
e4b-nvfp4-test:
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_e4b_nvfp4.cu -o /tmp/e4b_nvfp4_test -lcudart && /tmp/e4b_nvfp4_test

# E4B throughput baseline (prefill + decode tok/s), not a correctness test.
e4b-bench:
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_e4b_bench.cu cuda/e4b_engine.cu -o /tmp/e4b_bench -lcudart -lcublas && /tmp/e4b_bench $(MODEL_DIR)

# All E4B tests.
e4b-all: e4b-test e4b-load-test e4b-gguf-load-test e4b-fwd-test e4b-gen-test e4b-batch-test e4b-nvfp4-test

# ─── Clean ──────────────────────────────────────────────────────────────
clean:
	rm -f fucina cuda/*.o cuda/*.a cuda/test_engine tests/test_parser cuda/libdg.a
	$(GO) clean
	rm -rf tests/vectors/tmp/
