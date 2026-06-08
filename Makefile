# gem4d - Gemma 4 12B inference engine for DGX Spark GB10
# Only FP8 and Q8_0 formats, only CUDA (sm_121 Blackwell GB10)
# LoRA adapter support via --lora-scaled

# ─── Toolchain ─────────────────────────────────────────────────────────
# DGX Spark GB10: CUDA 13.0, CUDA arch sm_121, /usr/local/cuda-13
# Go 1.26.4 installed at /usr/local/go/bin
NVCC   := /usr/local/cuda-13/bin/nvcc
CUDA_HOME := /usr/local/cuda-13
GO     := /usr/local/go/bin/go

# ─── Architecture ──────────────────────────────────────────────────────
# sm_121 = Blackwell GB10 (DGX Spark specific compute capability)
CUDA_ARCH := sm_121
NVCCFLAGS := -arch=$(CUDA_ARCH) -O3 -lineinfo --use_fast_math \
             -Xcompiler -O3 -Xcompiler -pthread \
             --threads 8

CGO_CFLAGS   := -I$(CUDA_HOME)/include
CGO_LDFLAGS  := -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcuda -lpthread -lstdc++ -lm

.PHONY: all clean test cuda lib gem4d smoke profile

all: lib gem4d

# ─── CUDA Kernel Library ───────────────────────────────────────────────
# Two-step compilation: device code + link

cuda/gemma4_kernels.o: cuda/gemma4_kernels.cu cuda/gemma4_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -dc -o $@ cuda/gemma4_kernels.cu

cuda/gemma4_kernels_link.o: cuda/gemma4_kernels.o
	$(NVCC) $(NVCCFLAGS) -dlink -o $@ $<

cuda/libgem4d.a: cuda/gemma4_kernels.o cuda/gemma4_kernels_link.o
	ar rcs $@ $^

# ─── Go Binary ──────────────────────────────────────────────────────────
# IMPORTANT: cgo does NOT hash the contents of the `-lgem4d` static archive, so
# a plain `go build` happily relinks a STALE binary against an updated
# libgem4d.a (this caused weights to be read over unified memory → a ~4s cold
# page-fault charged to prefill). Force a relink every time: remove the old
# binary and rebuild the cgo package with -a. Verify with `strings gem4d | grep
# uploading` (must print the device-upload banner).
gem4d: cuda/libgem4d.a
	rm -f $@
	CGO_CFLAGS="$(CGO_CFLAGS)" \
	CGO_LDFLAGS="$(CGO_LDFLAGS)" \
	$(GO) build -a -ldflags="-s -w" -o $@ ./cmd/gem4d/
	@strings $@ | grep -q "uploading.*weights to device" \
		&& echo "gem4d: OK — device weight-upload path linked" \
		|| { echo "gem4d: ERROR — stale link, upload path missing"; exit 1; }

# ─── Standalone CUDA Test (without Go) ──────────────────────────────────
cuda/test_engine: cuda/test_engine.cu cuda/libgem4d.a
	$(NVCC) $(NVCCFLAGS) -o $@ $< cuda/libgem4d.a -lcudart -lcublas

# ─── Testing ────────────────────────────────────────────────────────────
test: gem4d
	./gem4d --test-parser
	CUDA_VISIBLE_DEVICES=0 ./gem4d --test-cuda

test-vectors: gem4d
	./gem4d --test-vectors tests/vectors/official.vec

# ─── Quick smoke test ───────────────────────────────────────────────────
smoke: gem4d
	./gem4d --prompt "Hello, world!" --predict 32 --temp 0

# ─── LoRA smoke test ────────────────────────────────────────────────────
lora-smoke: gem4d
	./gem4d --model model.gguf --lora-scaled lora.gguf \
		--prompt "Test prompt" --predict 32 --temp 0

# ─── Profiling ──────────────────────────────────────────────────────────
profile: gem4d
	nsys profile -o gem4d_profile -t cuda,nvtx ./gem4d \
		--prompt "Write a haiku about CUDA." --predict 128 --temp 0

# ─── Clean ──────────────────────────────────────────────────────────────
clean:
	rm -f gem4d cuda/*.o cuda/*.a cuda/test_engine tests/test_parser
	$(GO) clean
	rm -rf tests/vectors/tmp/
