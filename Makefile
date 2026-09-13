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
ELEMENTWISE_TESTS := relu_test leaky_relu_test silu_test swiglu_test clip_test sigmoid_test geglu_test
BASIC_TESTS := mat_add_test mat_copy_test reverse_test conv1d_test rainbow_test interleave_test rgb2grayscale_test batched_mm_test
PROGRAMS += $(ELEMENTWISE_TESTS) $(BASIC_TESTS)
GEMM_BENCHES := gemm_bench gemm_tile_bench gemm_wmma_bench gemm_wmma_tiled_bench gemm_cublas_bench
PROGRAMS += $(filter-out gemm_bench,$(GEMM_BENCHES))
GEMM_ARGS ?= 1024 1024 1024 100
NSYS ?= $(CUDA_HOME)/bin/nsys
NSYS_DIR ?= $(BIN_DIR)/nsys
NSYS_FLAGS ?= --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
NSYS_REPORTS ?= cuda_gpu_kern_sum,cuda_kern_exec_sum
NSYS_GEMMS := gemm_wmma gemm_wmma_tiled gemm_cublas
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
        nsys-gemm nsys-gemm-stats ncu-gemm-build ncu-gemm ncu-gemm-stats

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
$(BIN_DIR)/gemm_cublas_bench: tests/gemm.cu src/gemm_cublas.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lcublas
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
$(BIN_DIR)/sigmoid_test: TEST_DEFINE := TEST_SIGMOID
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
run-gemm-cublas: $(BIN_DIR)/gemm_cublas_bench
	$<
run-gemm-compare: $(addprefix $(BIN_DIR)/,$(GEMM_BENCHES))
	@set -e; for binary in $^; do \
	  echo "$$binary"; "$$binary" $(GEMM_ARGS); \
	done
check-gemm: $(addprefix $(BIN_DIR)/,$(GEMM_BENCHES))
	@set -e; for binary in $^; do \
	  echo "$$binary"; "$$binary" --check-only; \
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
check: all
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

# Large reduction regressions, including the LeetGPU top-k performance shape.
check-full: check
	$(BIN_DIR)/mse_test
	$(BIN_DIR)/top_k_test 50000000 100

COMPUTE_SANITIZER ?= compute-sanitizer
sanitize: all
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/gemm_bench 17 33 19 0
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_bench 65 129 67 0
	$(COMPUTE_SANITIZER) --tool racecheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_bench 65 129 67 0
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/gemm_cublas_bench --check-only
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/gauss_blur_test
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/cat_ce_test 257 65
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/mse_test 1025
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/top_k_test 4097 2049
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/interleave_test
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/batched_mm_test
	$(COMPUTE_SANITIZER) --tool synccheck --error-exitcode 1 $(BIN_DIR)/cat_ce_test 257 65
	$(COMPUTE_SANITIZER) --tool synccheck --error-exitcode 1 $(BIN_DIR)/mse_test 257
	$(COMPUTE_SANITIZER) --tool synccheck --error-exitcode 1 $(BIN_DIR)/top_k_test 4097 2049

help:
	@echo 'make [-j2] [NVCC=/path/to/nvcc] [NVCC_ARCH=sm_80|sm_75]'
	@echo 'all             Build tests and benchmarks (no GPU needed)'
	@echo 'compile-kernels Compile every src/*.cu independently'
	@echo 'check           Run small GPU correctness checks'
	@echo 'check-full      Also run large MSE and top-k regressions'
	@echo 'sanitize        Run selected memory/synchronization checks'
	@echo 'run-<operator>  Run one test/benchmark with default arguments'
	@echo 'check-gemm      Check scalar, tiled, both WMMA variants, and cuBLAS GEMM'
	@echo 'run-gemm-compare Compare all five GEMMs; GEMM_ARGS="M N K repeats"'
	@echo 'nsys-gemm       Profile WMMA, tiled WMMA, cuBLAS serially, then print stats'
	@echo '                Set CUDA_VISIBLE_DEVICES, GEMM_ARGS, NSYS_DIR, NSYS_FLAGS as needed'
	@echo 'nsys-gemm-stats  Print existing reports in NSYS_DIR (default: $(BIN_DIR)/nsys)'
	@echo 'ncu-gemm-build  Build all five GEMMs with -lineinfo in NCU_BIN_DIR'
	@echo 'ncu-gemm        Profile all five serially, then export text/CSV and print a summary'
	@echo '                Set CUDA_VISIBLE_DEVICES, GEMM_ARGS, NCU_DIR, NCU_SET, NCU_RUN as needed'
	@echo 'ncu-gemm-stats  Export existing NCU reports and regenerate comparison.csv (no GPU needed)'
	@echo 'clean           Remove current architecture build directory'

clean:
	rm -rf $(BIN_DIR)
