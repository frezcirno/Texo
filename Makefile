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
BINARIES := $(addprefix $(BIN_DIR)/,$(PROGRAMS))
KERNEL_OBJECTS := $(patsubst src/%.cu,$(BIN_DIR)/kernels/%.o,$(wildcard src/*.cu))
SUM_OBJECTS := $(BIN_DIR)/sum_manual.o $(BIN_DIR)/sum_cg.o $(BIN_DIR)/sum_cub.o
SOFTMAX_OBJECTS := $(BIN_DIR)/softmax_3kernel.o $(BIN_DIR)/softmax_4kernel.o

.PHONY: all help compile-kernels check check-full sanitize clean bench \
        run-reduce run-max run-softmax run-attention run-conv2d run-conv3d \
        run-mat-vec run-gemm run-cat-ce run-mse run-gauss-blur run-max-compare run-top-k

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
$(BIN_DIR)/cat_ce_test: tests/cat_ce.cpp src/cat_ce.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/mse_test: tests/mse.cpp src/mse.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/gauss_blur_test: tests/gauss_blur.cpp src/gauss_blur.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/top_k_test: tests/top_k.cpp src/top_k.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
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
	$(BIN_DIR)/cat_ce_test
	$(BIN_DIR)/mse_test 1025
	$(BIN_DIR)/gauss_blur_test
	$(BIN_DIR)/top_k_test

# Large reduction regressions, including the LeetGPU top-k performance shape.
check-full: check
	$(BIN_DIR)/mse_test
	$(BIN_DIR)/top_k_test 50000000 100

COMPUTE_SANITIZER ?= compute-sanitizer
sanitize: all
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/gemm_bench 17 33 19 0
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/gauss_blur_test
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/cat_ce_test 257 65
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/mse_test 1025
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/top_k_test 4097 2049
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
	@echo 'clean           Remove current architecture build directory'

clean:
	rm -rf $(BIN_DIR)
