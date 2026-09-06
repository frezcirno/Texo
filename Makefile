CUDA_HOME ?= /usr/local/cuda-11.7
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CXX       ?= g++

NVCC_ARCH ?= sm_80
NVCCFLAGS ?= -O3 -std=c++14 -arch=$(NVCC_ARCH) -Xcompiler -Wall
CXXFLAGS  ?= -O3 -std=c++14 -Wall -I$(CUDA_HOME)/include
LDFLAGS   ?= -L$(CUDA_HOME)/lib64 -lcudart

SRC_DIR  := src
TEST_DIR := test
BIN_DIR  := build

REDUCE_BENCH := $(BIN_DIR)/reduce_bench
MAX_BENCH    := $(BIN_DIR)/max_bench
SOFTMAX_BENCH := $(BIN_DIR)/softmax_bench
ATTENTION_BENCH := $(BIN_DIR)/attention_bench
CONV2D_BENCH := $(BIN_DIR)/conv2d_bench
CONV3D_BENCH := $(BIN_DIR)/conv3d_bench
MAT_VEC_BENCH := $(BIN_DIR)/mat_vec_mul_bench
GEMM_BENCH := $(BIN_DIR)/gemm_bench
CAT_CE_TEST := $(BIN_DIR)/cat_ce_test
MSE_TEST := $(BIN_DIR)/mse_test
GAUSS_BLUR_TEST := $(BIN_DIR)/gauss_blur_test


.PHONY: all clean run-reduce bench run-max run-softmax run-attention run-conv2d run-conv3d run-mat-vec run-gemm run-cat-ce run-mse run-gauss-blur

all: $(REDUCE_BENCH) $(MAX_BENCH) $(SOFTMAX_BENCH) $(ATTENTION_BENCH) $(CONV2D_BENCH) $(CONV3D_BENCH) $(MAT_VEC_BENCH) $(GEMM_BENCH) $(CAT_CE_TEST) $(MSE_TEST) $(GAUSS_BLUR_TEST)

$(BIN_DIR):
	@mkdir -p $@

REDUCE_SRCS := $(SRC_DIR)/sum_my.cu $(SRC_DIR)/sum_my_v2.cu $(SRC_DIR)/sum_stellar.cu $(SRC_DIR)/sum_cub.cu $(TEST_DIR)/sum.cpp
MAX_SRCS    := $(SRC_DIR)/max.cu $(TEST_DIR)/max.cpp
SOFTMAX_3_OBJ := $(BIN_DIR)/softmax_3kernel.o
SOFTMAX_4_OBJ := $(BIN_DIR)/softmax_4kernel.o

$(REDUCE_BENCH): $(REDUCE_SRCS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(REDUCE_SRCS) -o $@

$(MAX_BENCH): $(MAX_SRCS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(MAX_SRCS) -o $@

$(SOFTMAX_3_OBJ): $(SRC_DIR)/softmax_3kernel.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) \
		-Dsolve=softmax_3kernel \
		-Dwarp_max=softmax_3kernel_warp_max \
		-Dblock_max=softmax_3kernel_block_max \
		-Datomic_max_float=softmax_3kernel_atomic_max_float \
		-Dmax_kernel=softmax_3kernel_max_kernel \
		-Dexp_sum_kernel=softmax_3kernel_exp_sum_kernel \
		-Dnormalize_kernel=softmax_3kernel_normalize_kernel \
		-c $< -o $@

$(SOFTMAX_4_OBJ): $(SRC_DIR)/softmax_4kernal.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) \
		-Dsolve=softmax_4kernel \
		-Dwarp_max=softmax_4kernel_warp_max \
		-Dblock_max=softmax_4kernel_block_max \
		-Datomic_max_float=softmax_4kernel_atomic_max_float \
		-Dmax_kernel=softmax_4kernel_max_kernel \
		-Dexp_kernel=softmax_4kernel_exp_kernel \
		-Dsum_kernel=softmax_4kernel_sum_kernel \
		-Dnormalize_kernel=softmax_4kernel_normalize_kernel \
		-c $< -o $@

$(SOFTMAX_BENCH): $(SOFTMAX_3_OBJ) $(SOFTMAX_4_OBJ) $(TEST_DIR)/softmax.cpp | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(ATTENTION_BENCH): $(SRC_DIR)/attention.cu $(TEST_DIR)/attention.cpp | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(CONV2D_BENCH): $(SRC_DIR)/conv2d.cu $(TEST_DIR)/conv2d.cpp | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(CONV3D_BENCH): $(SRC_DIR)/conv3d.cu $(TEST_DIR)/conv3d.cpp | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(MAT_VEC_BENCH): $(TEST_DIR)/mat_vec_mul.cu $(SRC_DIR)/mat-vec-mul.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run-mat-vec: $(MAT_VEC_BENCH)
	$(MAT_VEC_BENCH)

$(GEMM_BENCH): $(TEST_DIR)/gemm.cu $(SRC_DIR)/gemm.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

run-gemm: $(GEMM_BENCH)
	$(GEMM_BENCH)

$(CAT_CE_TEST): $(TEST_DIR)/cat_ce.cpp $(SRC_DIR)/cat_ce.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

run-cat-ce: $(CAT_CE_TEST)
	$(CAT_CE_TEST)

$(MSE_TEST): $(TEST_DIR)/mse.cpp $(SRC_DIR)/mse.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

run-mse: $(MSE_TEST)
	$(MSE_TEST)

$(GAUSS_BLUR_TEST): $(TEST_DIR)/gauss_blur.cpp $(SRC_DIR)/gauss_blur.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

run-gauss-blur: $(GAUSS_BLUR_TEST)
	$(GAUSS_BLUR_TEST)

run-reduce bench: $(REDUCE_BENCH)
	$(REDUCE_BENCH)

run-max: $(MAX_BENCH)
	$(MAX_BENCH)

run-softmax: $(SOFTMAX_BENCH)
	$(SOFTMAX_BENCH)

run-attention: $(ATTENTION_BENCH)
	$(ATTENTION_BENCH)

run-conv2d: $(CONV2D_BENCH)
	$(CONV2D_BENCH)

run-conv3d: $(CONV3D_BENCH)
	$(CONV3D_BENCH)

clean:
	rm -rf $(BIN_DIR)
