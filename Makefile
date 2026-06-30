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

.PHONY: all clean test cuda lib libdg fucina smoke profile nvfp4-test \
        go-test go-test-race go-test-cgo vet lint check paged-kv-test paged-prefix-test qwen3-prefix-test \
        qwen3-parity-test qwen3moe-parity-test qwen3moe-spec-test qwen3moe-one-test qwen3-suffix-test gpu-gates \
        qwen35-detect-test \
        paged-kv-device-test packed-kv-test kv-quant-explore bench tool-bench \
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

cuda/gemma4_kernels.o: cuda/gemma4_kernels.cu cuda/gemma4_kernels.cuh \
                       cuda/gemma4_config.h cuda/gemma4_detect.h \
                       cuda/paged_kv.h cuda/paged_kv_device.cuh cuda/paged_prefix.h Makefile
	$(NVCC) $(NVCCFLAGS) -dc -o $@ cuda/gemma4_kernels.cu

cuda/gemma4_kernels_link.o: cuda/gemma4_kernels.o Makefile
	$(NVCC) $(NVCCFLAGS) -dlink -o $@ $<

cuda/libfucina.a: cuda/gemma4_kernels.o cuda/gemma4_kernels_link.o
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

# ─── Standalone CUDA Test (without Go) ──────────────────────────────────
cuda/test_engine: cuda/test_engine.cu cuda/libfucina.a
	$(NVCC) $(NVCCFLAGS) -o $@ $< cuda/libfucina.a -lcudart -lcublas -lcublasLt

# ─── Testing ────────────────────────────────────────────────────────────
# Full test suite: pure-Go unit tests, cgo-dependent tests (needs the CUDA
# archive), the binary's built-in self-tests, then the session's CUDA
# parity/determinism/lossless regression gates (gpu-gates) — all on the GPU.
test: fucina go-test go-test-cgo paged-kv-test paged-prefix-test gpu-gates
	./fucina --test-parser
	CUDA_VISIBLE_DEVICES=0 ./fucina --test-cuda

# ─── Aggregate GPU regression gates ─────────────────────────────────────
# The Qwen3-dense / Qwen3-MoE / spec / prefix / suffix / B=2-row-independence
# correctness gates introduced this session. Each is a standalone target, but
# none was a prerequisite of `test`, so all could be silently forgotten. This
# umbrella chains them so a regression in any cannot pass unnoticed. Requires
# the Qwen3 dense + MoE GGUFs (override with MODEL= / QWEN3MOE_MODEL=) and a
# GPU; run each invocation under the shared GPU flock.
gpu-gates: qwen3-parity-test qwen3moe-parity-test qwen3moe-spec-test qwen3moe-one-test qwen3-prefix-test qwen3-suffix-test
	@echo "gpu-gates: all Qwen3-dense/MoE/spec/prefix/suffix/B=2 regression gates passed"

# ─── Paged-KV allocator unit test (host-only, no GPU) ───────────────────
# Pure integer bookkeeping for the continuous-batching paged KV cache; runs on
# the host so it stays fast and CI-portable. See docs/continuous-batching.md.
paged-kv-test:
	g++ -std=c++17 -O2 -Wall -Wextra cuda/paged_kv_test.cc -o /tmp/fucina_paged_kv_test
	/tmp/fucina_paged_kv_test

# Cross-request prefix cache (RadixAttention): radix tree + refcount + LRU over
# the paged-KV pool. Pure host integer logic; see docs/continuous-batching.md.
paged-prefix-test:
	g++ -std=c++17 -O2 -Wall -Wextra cuda/paged_prefix_test.cc -o /tmp/fucina_paged_prefix_test
	/tmp/fucina_paged_prefix_test

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
qwen35-detect-test:
	g++ -std=c++17 -O2 -Wall -Wextra -Icuda cuda/test_qwen35_detect.cc -o /tmp/fucina_qwen35_detect
	/tmp/fucina_qwen35_detect $(QWEN35_MODEL)

# ─── Qwen3 dense numeric parity vs llama.cpp (GPU) ──────────────────────
# Feeds llama.cpp's input ids for "The capital of France is" through fucina's
# arch-driven multiseq path and asserts the greedy continuation matches
# llama.cpp's [12095,13,576,6722,315,15344,374,21718] (Qwen3-8B Q4_K_M).
qwen3-parity-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen3_parity.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen3_parity \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	/tmp/fucina_qwen3_parity

# ─── Qwen3-MoE (qwen3moe) numeric parity vs llama.cpp (GPU) ─────────────
# Loads Qwen3-30B-A3B (sparse MoE: 128 experts, top-8) and asserts fucina's greedy
# continuation of "The capital of France is" matches llama.cpp's raw-completion reference
# [12095,13,576,6722,315,279,3639,4180] token-for-token (8/8). Override the model with MODEL=.
QWEN3MOE_MODEL ?= /opt/spark/models/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf
qwen3moe-parity-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen3moe_parity.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen3moe_parity \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	/tmp/fucina_qwen3moe_parity $(QWEN3MOE_MODEL)

# ─── Qwen3-MoE DSpark spec losslessness on the MoE FFN (GPU) ────────────
# Drives Qwen3-30B-A3B through the spec-decode verify path and asserts: (A) depth-0 spec is
# byte-identical to plain step_batch (lossless), (B) a correct draft is accepted and grows the
# run, (C) a wrong draft is rejected and corrected. Guards the deterministic per-token MoE
# expert reduce (no atomicAdd scatter) that makes the batched MoE FFN bit-reproducible.
qwen3moe-spec-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen3moe_spec.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen3moe_spec \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	/tmp/fucina_qwen3moe_spec $(QWEN3MOE_MODEL)

# ─── Qwen3-MoE B=2 cross-sequence row-independence (GPU) ────────────────
# The atomicAdd-scatter nondeterminism bug only surfaced at B>=2 with DIVERGENT rows;
# single-seq parity missed it entirely. This batches two sequences at different
# trajectory positions in ONE step_batch(B=2) and asserts each row continues
# independently (row0==single-seq next, row1==trajectory[3], rows differ) — the
# direct gate for the deterministic dg_moe_route_inv/dg_moe_reduce combine.
qwen3moe-one-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen3moe_one.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen3moe_one \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	/tmp/fucina_qwen3moe_one $(QWEN3MOE_MODEL)

# ─── Qwen3 cross-request prefix cache losslessness (GPU) ────────────────
# Proves cache-served requests (shared-prefix reuse) produce a greedy token stream
# bit-identical to a cold request, sequentially and concurrently. See paged_prefix.h.
qwen3-prefix-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen3_prefix.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen3_prefix \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	/tmp/fucina_qwen3_prefix $(MODEL)

# ─── Stage 9: base-offset compute-bound suffix prefill losslessness (GPU) ─
# Proves the GEMM suffix prefill (paged_prefill_qwen3 base>0) is lossless: a sequence built
# prefix-then-suffix (cross-request prefix-cache adoption) yields a greedy stream bit-identical
# to a one-shot prefill, on BOTH Qwen3-8B dense and Qwen3-30B-A3B MoE. Args: <dense> [<moe>].
QWEN3_DENSE_MODEL ?= /opt/spark/models/Qwen3-8B-abliterated.Q4_K_M.gguf
qwen3-suffix-test: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen3_suffix_prefill.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen3_suffix \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	/tmp/fucina_qwen3_suffix $(if $(MODEL),$(MODEL),$(QWEN3_DENSE_MODEL)) $(QWEN3MOE_MODEL)

# ─── Qwen3 decode throughput (GPU) ──────────────────────────────────────
qwen3-bench: lib libdg
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -Icuda cuda/test_qwen3_bench.cu \
		cuda/libfucina.a cuda/libdg.a -o /tmp/fucina_qwen3_bench \
		-lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
	/tmp/fucina_qwen3_bench $(MODEL)

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

# tool-bench launches fucina and runs the agentic tool-call benchmark with the
# corrected --max-turns 12 (the upstream default 8 truncates TC-46/TC-62 mid-chain).
# Pass-through args via ARGS=, e.g.: make tool-bench ARGS="--perf"
# Continuous batching for the concurrency columns: make tool-bench BATCH=1 ARGS="--perf"
tool-bench: fucina
	MODEL=$(if $(MODEL),$(MODEL),model.gguf) BATCH=$(BATCH) scripts/tool_eval_bench.sh $(ARGS)

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

go-test-cgo: lib
	CGO_CFLAGS="$(CGO_CFLAGS)" \
	CGO_LDFLAGS="$(CGO_LDFLAGS)" \
	$(GO) test $(GO_TEST_CGO_PKGS) -count=1

go-test:
	$(GO) test $(GO_TEST_PKGS) -count=1

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

# NVFP4 safetensors loader unit tests (host + decode-kernel parity). Self-contained, no model.
nvfp4-test:
	g++ -std=c++17 -O2 -Wall -Wextra cuda/safetensors_test.cc   -o /tmp/st_test     && /tmp/st_test
	g++ -std=c++17 -O2 -Wall -Wextra cuda/nvfp4_test.cc          -o /tmp/nvfp4_test  && /tmp/nvfp4_test
	g++ -std=c++17 -O2 -Wall -Wextra cuda/nvfp4_loader_test.cc   -o /tmp/nvfp4_ld    && /tmp/nvfp4_ld
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 cuda/test_nvfp4_gemv.cu -o /tmp/nvfp4_gemv && /tmp/nvfp4_gemv

# Native Q4_K LM-head matvec unit test (kernel vs host reference decode).
q4k-test:
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 cuda/test_q4k_gemv.cu -o /tmp/q4k_gemv && /tmp/q4k_gemv

# ─── Clean ──────────────────────────────────────────────────────────────
clean:
	rm -f fucina cuda/*.o cuda/*.a cuda/test_engine tests/test_parser cuda/libdg.a
	$(GO) clean
	rm -rf tests/vectors/tmp/
