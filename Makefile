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

.PHONY: all clean run-reduce bench run-max run-softmax

all: $(REDUCE_BENCH) $(MAX_BENCH) $(SOFTMAX_BENCH)

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

run-reduce bench: $(REDUCE_BENCH)
	$(REDUCE_BENCH)

run-max: $(MAX_BENCH)
	$(MAX_BENCH)

run-softmax: $(SOFTMAX_BENCH)
	$(SOFTMAX_BENCH)

clean:
	rm -rf $(BIN_DIR)
