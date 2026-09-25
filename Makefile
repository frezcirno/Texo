.DEFAULT_GOAL := all
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
NVCC_ARCH ?= sm_80
NVCCFLAGS ?= -O3 -std=c++14 -arch=$(NVCC_ARCH) -Xcompiler -Wall
# Separate architectures to avoid accidentally running an old binary on T4.
BIN_DIR ?= build/$(NVCC_ARCH)
SRC_DIR := src
TEST_DIR := tests

PROGRAMS := reduce_bench max_bench softmax_bench attention_bench conv2d_bench \
            conv3d_bench mat_vec_mul_bench gemm_bench cat_ce_test mse_test gauss_blur_test top_k_test
ELEMENTWISE_TESTS := relu_test leaky_relu_test silu_test swiglu_test clip_test geglu_test
BASIC_TESTS := mat_add_test mat_copy_test reverse_test conv1d_test rainbow_test interleave_test sigmoid_test rgb2grayscale_test batched_mm_test
PROGRAMS += $(ELEMENTWISE_TESTS) $(BASIC_TESTS)
GEMM_BENCHES := gemm_bench gemm_tile_bench gemm_wmma_bench gemm_wmma_tiled_bench \
                gemm_wmma_tiled_pipeline_bench gemm_wmma_tiled_pipeline_aligned_bench \
                gemm_wmma_tiled_pipeline_aligned_swizzled_bench gemm_wmma_tiled_pipeline_multistage_bench \
                gemm_wmma_tiled_pipeline_mainloop_bench gemm_wmma_tiled_pipeline_reuse_bench \
                gemm_wmma_tiled_pipeline_epilogue_bench gemm_wmma_tiled_pipeline_large_bench \
                gemm_wmma_tiled_pipeline_schedule_bench \
                gemm_cublas_bench
PROGRAMS += $(filter-out gemm_bench,$(GEMM_BENCHES))
GEMM_DYNAMIC_TEST := $(BIN_DIR)/gemm_wmma_tiled_pipeline_large_dynamic_test
GEMM_SCHEDULE_TESTS := $(BIN_DIR)/gemm_wmma_tiled_pipeline_schedule_fixed_test \
                       $(BIN_DIR)/gemm_wmma_tiled_pipeline_schedule_four_test \
                       $(BIN_DIR)/gemm_wmma_tiled_pipeline_schedule_dynamic_test
GEMM_ARGS ?= 1024 1024 1024 100
NSYS ?= $(CUDA_HOME)/bin/nsys
NSYS_DIR ?= $(BIN_DIR)/nsys
NSYS_FLAGS ?= --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
NSYS_REPORTS ?= cuda_gpu_kern_sum,cuda_kern_exec_sum
NSYS_GEMMS := gemm_wmma gemm_wmma_tiled gemm_wmma_tiled_pipeline gemm_wmma_tiled_pipeline_aligned \
              gemm_wmma_tiled_pipeline_aligned_swizzled gemm_wmma_tiled_pipeline_multistage \
              gemm_wmma_tiled_pipeline_mainloop gemm_wmma_tiled_pipeline_reuse \
              gemm_wmma_tiled_pipeline_epilogue gemm_wmma_tiled_pipeline_large \
              gemm_wmma_tiled_pipeline_schedule gemm_cublas
NCU ?= $(CUDA_HOME)/bin/ncu
NCU_RUN ?=
NCU_DIR ?= $(BIN_DIR)/ncu
NCU_BIN_DIR ?= $(BIN_DIR)/ncu-bin
NCU_GEMMS ?= $(patsubst %_bench,%,$(GEMM_BENCHES))
NCU_SET ?= full
NCU_LAUNCH_SKIP ?= 7
NCU_LAUNCH_COUNT ?= 1
NCU_FLAGS ?= --config-file off --replay-mode kernel --clock-control none --cache-control all --import-source yes --target-processes application-only
PYTHON ?= python3
# Python dependencies remain optional for the standalone CUDA build.
WITH_TRITON ?= 0
TRITON_CUDA_GEMM ?= gemm_wmma_tiled_pipeline_schedule
TRITON_SOURCE ?= src/gemm.triton.py
TRITON_TEST_ARGS ?= --all-configs
TRITON_BENCH_ARGS ?=
BINARIES := $(addprefix $(BIN_DIR)/,$(PROGRAMS))
KERNEL_OBJECTS := $(patsubst src/%.cu,$(BIN_DIR)/kernels/%.o,$(wildcard src/*.cu))
SUM_OBJECTS := $(BIN_DIR)/sum_manual.o $(BIN_DIR)/sum_cg.o $(BIN_DIR)/sum_cub.o
SOFTMAX_OBJECTS := $(BIN_DIR)/softmax_3kernel.o $(BIN_DIR)/softmax_4kernel.o

.PHONY: all help compile-kernels check check-full sanitize clean bench \
        run-reduce run-max run-softmax run-attention run-conv2d run-conv3d \
        run-mat-vec run-gemm run-cat-ce run-mse run-gauss-blur run-max-compare run-top-k \
        run-relu run-leaky-relu run-silu run-swiglu run-clip run-mat-add \
        run-mat-copy run-reverse run-conv1d run-rainbow run-interleave \
        run-sigmoid run-geglu run-rgb2grayscale run-batched-mm \
        run-gemm-tile run-gemm-wmma run-gemm-wmma-tiled run-gemm-cublas run-gemm-compare check-gemm \
        run-gemm-wmma-tiled-pipeline run-gemm-wmma-tiled-pipeline-aligned \
        run-gemm-wmma-tiled-pipeline-aligned-swizzled run-gemm-wmma-tiled-pipeline-multistage \
        run-gemm-wmma-tiled-pipeline-mainloop run-gemm-wmma-tiled-pipeline-reuse \
        run-gemm-wmma-tiled-pipeline-epilogue run-gemm-wmma-tiled-pipeline-large \
        run-gemm-wmma-tiled-pipeline-schedule \
        nsys-gemm nsys-gemm-stats ncu-gemm-build ncu-gemm ncu-gemm-stats \
        check-gemm-triton bench-gemm-triton-compare sanitize-gemm-triton

all: $(BINARIES)
compile-kernels: $(KERNEL_OBJECTS)

$(BIN_DIR) $(BIN_DIR)/kernels:
	mkdir -p $@

$(BIN_DIR)/kernels/%.o: src/%.cu | $(BIN_DIR)/kernels
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BIN_DIR)/sum_manual.o: src/sum.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -Dsolve=reduce_manual -c $< -o $@
$(BIN_DIR)/sum_cg.o: src/sum_cg.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -Dsolve=reduce_cg -c $< -o $@
$(BIN_DIR)/sum_cub.o: src/sum_cub.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
$(BIN_DIR)/reduce_bench: tests/sum.cpp $(SUM_OBJECTS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Compile each standalone solve under a unique name for comparisons.
$(BIN_DIR)/softmax_%kernel.o: src/softmax_%kernel.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -Dsolve=softmax_$*kernel \
	  -Dwarp_max=softmax_$*kernel_warp_max -Dblock_max=softmax_$*kernel_block_max \
	  -Datomic_max_float=softmax_$*kernel_atomic_max_float \
	  -Dmax_kernel=softmax_$*kernel_max_kernel -Dexp_kernel=softmax_$*kernel_exp_kernel \
	  -Dexp_sum_kernel=softmax_$*kernel_exp_sum_kernel \
	  -Dsum_kernel=softmax_$*kernel_sum_kernel \
	  -Dnormalize_kernel=softmax_$*kernel_normalize_kernel -c $< -o $@
$(BIN_DIR)/softmax_bench: tests/softmax.cpp $(SOFTMAX_OBJECTS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BIN_DIR)/max_bench: tests/max.cpp src/max.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/attention_bench: tests/attention.cpp src/mha.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/conv2d_bench: tests/conv2d.cpp src/conv2d.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/conv3d_bench: tests/conv3d.cpp src/conv3d.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
# This test includes the implementation to instantiate both kernel variants.
$(BIN_DIR)/mat_vec_mul_bench: tests/mat_vec_mul.cu src/mat_vec_mul.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@
$(BIN_DIR)/gemm_bench: tests/gemm.cu src/gemm.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/gemm_tile_bench: tests/gemm.cu src/gemm_tile.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/gemm_wmma_bench: tests/gemm.cu src/gemm_wmma.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/gemm_wmma_tiled_bench: tests/gemm.cu src/gemm_wmma_tiled.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/gemm_wmma_tiled_pipeline_bench: tests/gemm.cu src/gemm_wmma_tiled_pipeline.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/gemm_wmma_tiled_pipeline_aligned_bench: tests/gemm.cu src/gemm_wmma_tiled_pipeline_aligned.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/gemm_wmma_tiled_pipeline_aligned_swizzled_bench: tests/gemm.cu src/gemm_wmma_tiled_pipeline_aligned_swizzled.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/gemm_wmma_tiled_pipeline_multistage_bench: tests/gemm.cu src/gemm_wmma_tiled_pipeline_multistage.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/gemm_wmma_tiled_pipeline_mainloop_bench: tests/gemm.cu src/gemm_wmma_tiled_pipeline_mainloop.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/gemm_wmma_tiled_pipeline_reuse_bench: tests/gemm.cu src/gemm_wmma_tiled_pipeline_reuse.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/gemm_wmma_tiled_pipeline_epilogue_bench: tests/gemm.cu src/gemm_wmma_tiled_pipeline_epilogue.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/gemm_wmma_tiled_pipeline_large_bench: tests/gemm.cu src/gemm_wmma_tiled_pipeline_large.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/gemm_wmma_tiled_pipeline_schedule_bench: tests/gemm.cu src/gemm_wmma_tiled_pipeline_schedule.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
# Small CPU-reference cases must exercise the new mainloop even when automatic
# dispatch reserves it for larger grids. Cover both two- and four-K-group loops.
$(BIN_DIR)/gemm_wmma_tiled_pipeline_schedule_fixed_test: tests/gemm.cu src/gemm_wmma_tiled_pipeline_schedule.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -DGEMM_AUTO_TILE=0 -DGEMM_BM=128 -DGEMM_BN=128 \
	  -DGEMM_BK=32 -DGEMM_WM=64 -DGEMM_WN=64 -DGEMM_STAGES=3 \
	  -DGEMM_MULTISTAGE_MIN_BLOCKS=2 $^ -o $@
$(BIN_DIR)/gemm_wmma_tiled_pipeline_schedule_four_test: tests/gemm.cu src/gemm_wmma_tiled_pipeline_schedule.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -DGEMM_AUTO_TILE=0 -DGEMM_BM=128 -DGEMM_BN=128 \
	  -DGEMM_BK=32 -DGEMM_WM=64 -DGEMM_WN=64 -DGEMM_STAGES=4 \
	  -DGEMM_MULTISTAGE_MIN_BLOCKS=2 $^ -o $@
$(BIN_DIR)/gemm_wmma_tiled_pipeline_schedule_dynamic_test: tests/gemm.cu src/gemm_wmma_tiled_pipeline_schedule.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -DGEMM_AUTO_TILE=0 -DGEMM_BM=128 -DGEMM_BN=256 \
	  -DGEMM_BK=64 -DGEMM_WM=32 -DGEMM_WN=64 -DGEMM_STAGES=2 \
	  -DGEMM_MULTISTAGE_MIN_BLOCKS=1 $^ -o $@
# Exercise dynamic storage and a different warp partition even when automatic
# dispatch retains smaller static tiles. This is a check target, not a baseline.
$(GEMM_DYNAMIC_TEST): tests/gemm.cu src/gemm_wmma_tiled_pipeline_large.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -DGEMM_AUTO_TILE=0 -DGEMM_BM=128 -DGEMM_BN=256 \
	  -DGEMM_BK=32 -DGEMM_WM=32 -DGEMM_WN=64 -DGEMM_STAGES=4 \
	  -DGEMM_MULTISTAGE_MIN_BLOCKS=1 $^ -o $@
$(BIN_DIR)/gemm_cublas_bench: tests/gemm.cu src/gemm_cublas.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lcublas
# Optional large-shape timings avoid the cubic CPU reference in *_bench.
# Keep CPU correctness in make check; these validate against cuBLAS first.
$(BIN_DIR)/gemm%_perf: benchmarks/gemm.cu src/gemm%.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lcublas
# Optional ctypes adapters retain each standalone solve and its default stream.
$(BIN_DIR)/gemm%.so: src/gemm%.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler -fPIC -cudart shared $< -o $@ -lcublas
$(BIN_DIR)/cat_ce_test: tests/cat_ce.cpp src/cat_ce.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/mse_test: tests/mse.cpp src/mse.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/gauss_blur_test: tests/gauss_blur.cpp src/gauss_blur.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/top_k_test: tests/top_k.cpp src/top_k.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Share the elementwise CPU checks while linking each solve independently.
$(BIN_DIR)/relu_test: TEST_DEFINE := TEST_RELU
$(BIN_DIR)/leaky_relu_test: TEST_DEFINE := TEST_LEAKY_RELU
$(BIN_DIR)/silu_test: TEST_DEFINE := TEST_SILU
$(BIN_DIR)/swiglu_test: TEST_DEFINE := TEST_SWIGLU
$(BIN_DIR)/clip_test: TEST_DEFINE := TEST_CLIP
$(BIN_DIR)/geglu_test: TEST_DEFINE := TEST_GEGLU
$(addprefix $(BIN_DIR)/,$(ELEMENTWISE_TESTS)): $(BIN_DIR)/%_test: tests/elementwise.cpp src/%.cu tests/test_utils.h | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -D$(TEST_DEFINE) $(filter %.cpp %.cu,$^) -o $@
$(addprefix $(BIN_DIR)/,$(BASIC_TESTS)): $(BIN_DIR)/%_test: tests/%.cpp src/%.cu tests/test_utils.h | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(filter %.cpp %.cu,$^) -o $@

$(BIN_DIR)/reduce_max_compare: benchmarks/reduce_max.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run-reduce bench: $(BIN_DIR)/reduce_bench
	$<
run-max: $(BIN_DIR)/max_bench
	$<
run-softmax: $(BIN_DIR)/softmax_bench
	$<
run-attention: $(BIN_DIR)/attention_bench
	$<
run-conv2d: $(BIN_DIR)/conv2d_bench
	$<
run-conv3d: $(BIN_DIR)/conv3d_bench
	$<
run-mat-vec: $(BIN_DIR)/mat_vec_mul_bench
	$<
run-gemm: $(BIN_DIR)/gemm_bench
	$<
run-gemm-tile: $(BIN_DIR)/gemm_tile_bench
	$<
run-gemm-wmma: $(BIN_DIR)/gemm_wmma_bench
	$<
run-gemm-wmma-tiled: $(BIN_DIR)/gemm_wmma_tiled_bench
	$<
run-gemm-wmma-tiled-pipeline: $(BIN_DIR)/gemm_wmma_tiled_pipeline_bench
	$< $(GEMM_ARGS)
run-gemm-wmma-tiled-pipeline-aligned: $(BIN_DIR)/gemm_wmma_tiled_pipeline_aligned_bench
	$< $(GEMM_ARGS)
run-gemm-wmma-tiled-pipeline-aligned-swizzled: $(BIN_DIR)/gemm_wmma_tiled_pipeline_aligned_swizzled_bench
	$< $(GEMM_ARGS)
run-gemm-wmma-tiled-pipeline-multistage: $(BIN_DIR)/gemm_wmma_tiled_pipeline_multistage_bench
	$< $(GEMM_ARGS)
run-gemm-wmma-tiled-pipeline-mainloop: $(BIN_DIR)/gemm_wmma_tiled_pipeline_mainloop_bench
	$< $(GEMM_ARGS)
run-gemm-wmma-tiled-pipeline-reuse: $(BIN_DIR)/gemm_wmma_tiled_pipeline_reuse_bench
	$< $(GEMM_ARGS)
run-gemm-wmma-tiled-pipeline-epilogue: $(BIN_DIR)/gemm_wmma_tiled_pipeline_epilogue_bench
	$< $(GEMM_ARGS)
run-gemm-wmma-tiled-pipeline-large: $(BIN_DIR)/gemm_wmma_tiled_pipeline_large_bench
	$< $(GEMM_ARGS)
run-gemm-wmma-tiled-pipeline-schedule: $(BIN_DIR)/gemm_wmma_tiled_pipeline_schedule_bench
	$< $(GEMM_ARGS)
run-gemm-cublas: $(BIN_DIR)/gemm_cublas_bench
	$<
run-gemm-compare: $(addprefix $(BIN_DIR)/,$(GEMM_BENCHES))
	@set -e; for binary in $^; do \
	  echo "$$binary"; "$$binary" $(GEMM_ARGS); \
	done
check-gemm: $(addprefix $(BIN_DIR)/,$(GEMM_BENCHES)) $(GEMM_DYNAMIC_TEST) $(GEMM_SCHEDULE_TESTS)
	@set -e; for binary in $^; do \
	  echo "$$binary"; "$$binary" --check-only; \
	done
ifeq ($(WITH_TRITON),1)
	$(MAKE) check-gemm-triton
endif
check-gemm-triton: $(BIN_DIR)/gemm_cublas_bench
	$(PYTHON) tests/gemm_triton.py --source "$(TRITON_SOURCE)" --cases-binary $< $(TRITON_TEST_ARGS)
bench-gemm-triton-compare: $(BIN_DIR)/$(TRITON_CUDA_GEMM).so $(BIN_DIR)/gemm_cublas.so
	$(PYTHON) benchmarks/gemm_triton.py --bin-dir "$(BIN_DIR)" --cuda "$(TRITON_CUDA_GEMM)" \
	  --source "$(TRITON_SOURCE)" --output "$(BIN_DIR)/triton-comparison.csv" $(TRITON_BENCH_ARGS)
sanitize-gemm-triton: $(BIN_DIR)/gemm_cublas_bench
	@set -e; for tool in memcheck racecheck synccheck; do \
	  $(COMPUTE_SANITIZER) --tool $$tool --error-exitcode 1 \
	    $(PYTHON) tests/gemm_triton.py --source "$(TRITON_SOURCE)" --cases-binary $< --all-configs --fixed-only \
	    --case multi-tile-tail --case unaligned-A-B --case ab10-nan-K0; \
	done
# Keep GPU profiling serial even with make -j; compilation may run in parallel.
nsys-gemm: $(addprefix $(BIN_DIR)/,$(addsuffix _bench,$(NSYS_GEMMS)))
	mkdir -p "$(NSYS_DIR)"
	@set -e; for impl in $(NSYS_GEMMS); do \
	  "$(NSYS)" profile $(NSYS_FLAGS) --force-overwrite=true \
	    -o "$(NSYS_DIR)/$$impl" "$(BIN_DIR)/$${impl}_bench" $(GEMM_ARGS); \
	done
	$(MAKE) nsys-gemm-stats
nsys-gemm-stats:
	@set -e; for impl in $(NSYS_GEMMS); do \
	  "$(NSYS)" stats --force-export=true --timeunit us --report $(NSYS_REPORTS) \
	    "$(NSYS_DIR)/$$impl.nsys-rep"; \
	done
# Build as the current user. NCU_RUN only prefixes the profiler, e.g. sudo.
ncu-gemm-build:
	$(MAKE) BIN_DIR="$(NCU_BIN_DIR)" NVCCFLAGS="$(NVCCFLAGS) -lineinfo" \
	  $(addprefix $(NCU_BIN_DIR)/,$(addsuffix _bench,$(NCU_GEMMS)))
ncu-gemm: ncu-gemm-build
	mkdir -p "$(NCU_DIR)"
	@set -e; for impl in $(NCU_GEMMS); do \
	  $(NCU_RUN) "$(NCU)" $(NCU_FLAGS) --set $(NCU_SET) \
	    --launch-skip $(NCU_LAUNCH_SKIP) --launch-count $(NCU_LAUNCH_COUNT) \
	    --force-overwrite --export "$(NCU_DIR)/$$impl" \
	    "$(NCU_BIN_DIR)/$${impl}_bench" $(GEMM_ARGS); \
	done
	$(MAKE) ncu-gemm-stats
ncu-gemm-stats:
	@set -e; for impl in $(NCU_GEMMS); do \
	  "$(NCU)" --import "$(NCU_DIR)/$$impl.ncu-rep" --page details \
	    > "$(NCU_DIR)/$$impl.txt"; \
	  "$(NCU)" --import "$(NCU_DIR)/$$impl.ncu-rep" --page raw --csv --print-units base \
	    > "$(NCU_DIR)/$$impl.raw.csv"; \
	done
	$(PYTHON) scripts/ncu_summary.py "$(NCU_DIR)" $(NCU_GEMMS)
run-cat-ce: $(BIN_DIR)/cat_ce_test
	$<
run-mse: $(BIN_DIR)/mse_test
	$<
run-gauss-blur: $(BIN_DIR)/gauss_blur_test
	$<
run-max-compare: $(BIN_DIR)/reduce_max_compare
	$<
run-top-k: $(BIN_DIR)/top_k_test
	$<
run-relu: $(BIN_DIR)/relu_test
	$<
run-leaky-relu: $(BIN_DIR)/leaky_relu_test
	$<
run-silu: $(BIN_DIR)/silu_test
	$<
run-swiglu: $(BIN_DIR)/swiglu_test
	$<
run-clip: $(BIN_DIR)/clip_test
	$<
run-mat-add: $(BIN_DIR)/mat_add_test
	$<
run-mat-copy: $(BIN_DIR)/mat_copy_test
	$<
run-reverse: $(BIN_DIR)/reverse_test
	$<
run-conv1d: $(BIN_DIR)/conv1d_test
	$<
run-rainbow: $(BIN_DIR)/rainbow_test
	$<
run-interleave: $(BIN_DIR)/interleave_test
	$<
run-sigmoid: $(BIN_DIR)/sigmoid_test
	$<
run-geglu: $(BIN_DIR)/geglu_test
	$<
run-rgb2grayscale: $(BIN_DIR)/rgb2grayscale_test
	$<
run-batched-mm: $(BIN_DIR)/batched_mm_test
	$<

# Small reproducible GPU checks. Every executable returns nonzero on failure.
check: all $(GEMM_DYNAMIC_TEST) $(GEMM_SCHEDULE_TESTS)
	$(BIN_DIR)/reduce_bench 1025 2
	$(BIN_DIR)/max_bench 1025 2
	$(BIN_DIR)/softmax_bench 1025 2 1
	$(BIN_DIR)/attention_bench 17 33 16 2 1
	$(BIN_DIR)/conv2d_bench 17 35 3 5 2 1
	$(BIN_DIR)/conv3d_bench 9 11 13 3 3 3 2 1
	$(BIN_DIR)/mat_vec_mul_bench --check-only
	$(BIN_DIR)/gemm_bench --check-only
	$(BIN_DIR)/gemm_tile_bench --check-only
	$(BIN_DIR)/gemm_wmma_bench --check-only
	$(BIN_DIR)/gemm_wmma_tiled_bench --check-only
	$(BIN_DIR)/gemm_wmma_tiled_pipeline_bench --check-only
	$(BIN_DIR)/gemm_wmma_tiled_pipeline_aligned_bench --check-only
	$(BIN_DIR)/gemm_wmma_tiled_pipeline_aligned_swizzled_bench --check-only
	$(BIN_DIR)/gemm_wmma_tiled_pipeline_multistage_bench --check-only
	$(BIN_DIR)/gemm_wmma_tiled_pipeline_mainloop_bench --check-only
	$(BIN_DIR)/gemm_wmma_tiled_pipeline_reuse_bench --check-only
	$(BIN_DIR)/gemm_wmma_tiled_pipeline_epilogue_bench --check-only
	$(BIN_DIR)/gemm_wmma_tiled_pipeline_large_bench --check-only
	$(GEMM_DYNAMIC_TEST) --check-only
	$(BIN_DIR)/gemm_wmma_tiled_pipeline_schedule_bench --check-only
	@set -e; for binary in $(GEMM_SCHEDULE_TESTS); do "$$binary" --check-only; done
	$(BIN_DIR)/gemm_cublas_bench --check-only
	$(BIN_DIR)/cat_ce_test
	$(BIN_DIR)/mse_test 1025
	$(BIN_DIR)/gauss_blur_test
	$(BIN_DIR)/top_k_test
	$(BIN_DIR)/relu_test
	$(BIN_DIR)/leaky_relu_test
	$(BIN_DIR)/silu_test
	$(BIN_DIR)/swiglu_test
	$(BIN_DIR)/clip_test
	$(BIN_DIR)/mat_add_test
	$(BIN_DIR)/mat_copy_test
	$(BIN_DIR)/reverse_test
	$(BIN_DIR)/conv1d_test
	$(BIN_DIR)/rainbow_test
	$(BIN_DIR)/interleave_test
	$(BIN_DIR)/sigmoid_test
	$(BIN_DIR)/geglu_test
	$(BIN_DIR)/rgb2grayscale_test
	$(BIN_DIR)/batched_mm_test
ifeq ($(WITH_TRITON),1)
	$(MAKE) check-gemm-triton
endif

# Large reduction regressions, including the LeetGPU top-k performance shape.
check-full: check
	$(BIN_DIR)/mse_test
	$(BIN_DIR)/top_k_test 50000000 100

COMPUTE_SANITIZER ?= compute-sanitizer
sanitize: all $(GEMM_DYNAMIC_TEST) $(GEMM_SCHEDULE_TESTS)
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/gemm_bench 17 33 19 0
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_bench 65 129 67 0
	$(COMPUTE_SANITIZER) --tool racecheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_bench 65 129 67 0
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_bench --check-only
	$(COMPUTE_SANITIZER) --tool racecheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_bench --check-only
	$(COMPUTE_SANITIZER) --tool synccheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_bench --check-only
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_aligned_bench --check-only
	$(COMPUTE_SANITIZER) --tool racecheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_aligned_bench --check-only
	$(COMPUTE_SANITIZER) --tool synccheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_aligned_bench --check-only
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_aligned_swizzled_bench --check-only
	$(COMPUTE_SANITIZER) --tool racecheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_aligned_swizzled_bench --check-only
	$(COMPUTE_SANITIZER) --tool synccheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_aligned_swizzled_bench --check-only
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_multistage_bench --check-only
	$(COMPUTE_SANITIZER) --tool racecheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_multistage_bench --check-only
	$(COMPUTE_SANITIZER) --tool synccheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_multistage_bench --check-only
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_mainloop_bench --check-only
	$(COMPUTE_SANITIZER) --tool racecheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_mainloop_bench --check-only
	$(COMPUTE_SANITIZER) --tool synccheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_mainloop_bench --check-only
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_reuse_bench --check-only
	$(COMPUTE_SANITIZER) --tool racecheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_reuse_bench --check-only
	$(COMPUTE_SANITIZER) --tool synccheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_reuse_bench --check-only
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_epilogue_bench --check-only
	$(COMPUTE_SANITIZER) --tool racecheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_epilogue_bench --check-only
	$(COMPUTE_SANITIZER) --tool synccheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_epilogue_bench --check-only
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_large_bench --check-only
	$(COMPUTE_SANITIZER) --tool racecheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_large_bench --check-only
	$(COMPUTE_SANITIZER) --tool synccheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_large_bench --check-only
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(GEMM_DYNAMIC_TEST) --check-only
	$(COMPUTE_SANITIZER) --tool racecheck --error-exitcode 1 $(GEMM_DYNAMIC_TEST) --check-only
	$(COMPUTE_SANITIZER) --tool synccheck --error-exitcode 1 $(GEMM_DYNAMIC_TEST) --check-only
	@set -e; for binary in $(BIN_DIR)/gemm_wmma_tiled_pipeline_schedule_bench $(GEMM_SCHEDULE_TESTS); do \
	  for tool in memcheck racecheck synccheck; do \
	    $(COMPUTE_SANITIZER) --tool $$tool --error-exitcode 1 "$$binary" --check-only; \
	  done; \
	done
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/gemm_cublas_bench --check-only
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/gauss_blur_test
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/cat_ce_test 257 65
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/mse_test 1025
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/top_k_test 4097 2049
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/interleave_test
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/sigmoid_test
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/batched_mm_test
	$(COMPUTE_SANITIZER) --tool synccheck --error-exitcode 1 $(BIN_DIR)/cat_ce_test 257 65
	$(COMPUTE_SANITIZER) --tool synccheck --error-exitcode 1 $(BIN_DIR)/mse_test 257
	$(COMPUTE_SANITIZER) --tool synccheck --error-exitcode 1 $(BIN_DIR)/top_k_test 4097 2049
ifeq ($(WITH_TRITON),1)
	$(MAKE) sanitize-gemm-triton
endif

help:
	@echo 'make [-j2] [NVCC=/path/to/nvcc] [NVCC_ARCH=sm_80|sm_75]'
	@echo 'all             Build tests and benchmarks (no GPU needed)'
	@echo 'compile-kernels Compile every src/*.cu independently'
	@echo 'check           Run small GPU correctness checks'
	@echo 'check-full      Also run large MSE and top-k regressions'
	@echo 'sanitize        Run selected memory/synchronization checks'
	@echo 'run-<operator>  Run one test/benchmark with default arguments'
	@echo 'check-gemm      Check scalar, tiled, all WMMA variants, and cuBLAS GEMM'
	@echo 'run-gemm-compare Compare all fourteen GEMMs; GEMM_ARGS="M N K repeats"'
	@echo 'run-gemm-wmma-tiled-pipeline Run async/double-buffered WMMA; uses GEMM_ARGS'
	@echo 'run-gemm-wmma-tiled-pipeline-aligned Run WMMA with a full/aligned fast path; uses GEMM_ARGS'
	@echo 'run-gemm-wmma-tiled-pipeline-aligned-swizzled Run XOR shared-layout GEMM; uses GEMM_ARGS'
	@echo 'run-gemm-wmma-tiled-pipeline-multistage Run GEMM with deeper input/operand buffering; uses GEMM_ARGS'
	@echo 'run-gemm-wmma-tiled-pipeline-mainloop Run GEMM mainloop scheduling experiment; uses GEMM_ARGS'
	@echo 'run-gemm-wmma-tiled-pipeline-reuse Run GEMM with the 96x128 reuse tile; uses GEMM_ARGS'
	@echo 'run-gemm-wmma-tiled-pipeline-epilogue Run GEMM with alpha=1/beta=0 specialization; uses GEMM_ARGS'
	@echo 'run-gemm-wmma-tiled-pipeline-large Run large-tile/dynamic-shared GEMM experiments; uses GEMM_ARGS'
	@echo 'run-gemm-wmma-tiled-pipeline-schedule Run distributed GEMM mainloop; uses GEMM_ARGS'
	@echo 'gemm*_perf      Optional large-shape binaries in BIN_DIR; M N K [iterations|--profile]'
	@echo 'check-gemm-triton Reuse CUDA GEMM cases for Triton, plus all autotune candidates'
	@echo '                 Set PYTHON to a CUDA PyTorch/Triton environment; WITH_TRITON=1 includes it in check/check-gemm'
	@echo '                 TRITON_SOURCE=src/gemm.triton.v2.py (or v3.py) selects another version'
	@echo 'bench-gemm-triton-compare Compare Triton, CUDA schedule, and cuBLAS on identical buffers'
	@echo '                 TRITON_BENCH_ARGS="--shape M N K [--verify-only]"; TRITON_CUDA_GEMM selects the CUDA source'
	@echo '                 Repeat --reference-source src/gemm.triton.py to compare multiple Triton versions together'
	@echo 'sanitize-gemm-triton Check Triton tails/unaligned pointers with all three sanitizer tools'
	@echo 'nsys-gemm       Profile all WMMA variants and cuBLAS serially, then print stats'
	@echo '                Set CUDA_VISIBLE_DEVICES, GEMM_ARGS, NSYS_DIR, NSYS_FLAGS as needed'
	@echo 'nsys-gemm-stats  Print existing reports in NSYS_DIR (default: $(BIN_DIR)/nsys)'
	@echo 'ncu-gemm-build  Build all fourteen GEMMs with -lineinfo in NCU_BIN_DIR'
	@echo 'ncu-gemm        Profile all fourteen serially, then export text/CSV and print a summary'
	@echo '                Set CUDA_VISIBLE_DEVICES, GEMM_ARGS, NCU_DIR, NCU_SET, NCU_RUN as needed'
	@echo 'ncu-gemm-stats  Export existing NCU reports and regenerate comparison.csv (no GPU needed)'
	@echo 'clean           Remove current architecture build directory'

clean:
	rm -rf $(BIN_DIR)
