.DEFAULT_GOAL := all
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
NVCC_ARCH ?= sm_80
NVCCFLAGS ?= -O3 -std=c++14 -arch=$(NVCC_ARCH) -Xcompiler -Wall
CUDNN_INCLUDE ?= $(firstword $(wildcard $(CUDA_HOME)/include/cudnn.h /usr/include/cudnn.h))
CUDNN_LIBRARY ?= $(firstword $(wildcard $(CUDA_HOME)/lib64/libcudnn.so \
	$(CUDA_HOME)/targets/x86_64-linux/lib/libcudnn.so \
	/usr/lib/x86_64-linux-gnu/libcudnn.so))
# Separate architectures to avoid accidentally running an old binary on T4.
BIN_DIR ?= build/$(NVCC_ARCH)
SRC_DIR := src
TEST_DIR := tests

PROGRAMS := reduce_bench max_bench softmax_bench attention_bench conv2d_bench \
            conv3d_bench mv_bench gemm_bench cat_ce_test mse_test gauss_blur_test top_k_test
ELEMENTWISE_TESTS := relu_test leaky_relu_test silu_test swiglu_test clip_test geglu_test
DOT_TESTS := dot_test dot_fp16_test
BASIC_TESTS := mat_add_test mat_copy_test reverse_test conv1d_test rainbow_test interleave_test sigmoid_test rgb2grayscale_test batched_mm_test batched_mm_fp16_test alibi_test mm_int8_test lr_test mc_int_test mat_pow_test nn_test batch_norm_test rms_norm_test group_norm_test layer_norm_test max_pooling_2d_test count_test count3d_test slice_sum_test slice_sum2d_test slice_sum3d_test max_subarray_sum_test 2d_jacobi_stencil_test dequantization_test rope_test sparse_mm_test stream_compaction_test segmented_scan_test fft2d_test adderboard_test
PROGRAMS += $(ELEMENTWISE_TESTS) $(BASIC_TESTS) $(DOT_TESTS) lr_newton_test
GEMM_BENCHES := gemm_bench gemm_tiled_bench gemm_wmma_bench gemm_wmma_tiled_bench \
                gemm_wmma_tiled_pipeline_bench gemm_wmma_tiled_pipeline_schedule_bench \
                gemm_cublas_bench
PROGRAMS += $(filter-out gemm_bench,$(GEMM_BENCHES)) gemm_fp32_bench \
            gemm_fp32_split_128_test gemm_fp32_split_64_test
GEMM_SCHEDULE_TESTS := $(BIN_DIR)/gemm_wmma_tiled_pipeline_schedule_fixed_test \
                       $(BIN_DIR)/gemm_wmma_tiled_pipeline_schedule_four_test \
                       $(BIN_DIR)/gemm_wmma_tiled_pipeline_schedule_dynamic_test
GEMM_ARGS ?= 1024 1024 1024 100
BATCHED_MM_FP16_ARGS ?=
ALIBI_ARGS ?=
DOT_ARGS ?=
DOT_FP16_ARGS ?=
SOFTMAX_ARGS ?=
MM_INT8_ARGS ?=
MC_INT_ARGS ?=
MAT_POW_ARGS ?=
NN_ARGS ?=
BN_ARGS ?=
RMS_ARGS ?=
GN_ARGS ?=
LN_ARGS ?=
POOL_ARGS ?=
COUNT_ARGS ?=
COUNT3D_ARGS ?=
SLICE_SUM_ARGS ?=
SLICE_SUM2D_ARGS ?=
SLICE_SUM3D_ARGS ?=
MAX_SUBARRAY_SUM_ARGS ?=
FFT2D_ARGS ?=
NSYS ?= $(CUDA_HOME)/bin/nsys
NSYS_DIR ?= $(BIN_DIR)/nsys
NSYS_FLAGS ?= --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
NSYS_REPORTS ?= cuda_gpu_kern_sum,cuda_kern_exec_sum
NSYS_GEMMS := gemm_wmma gemm_wmma_tiled gemm_wmma_tiled_pipeline \
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
KERNEL_SOURCES := $(filter-out src/softmax_cudnn.cu,$(wildcard src/*.cu))
KERNEL_OBJECTS := $(patsubst src/%.cu,$(BIN_DIR)/kernels/%.o,$(KERNEL_SOURCES))
SUM_OBJECTS := $(BIN_DIR)/sum_manual.o $(BIN_DIR)/sum_cg.o $(BIN_DIR)/sum_cub.o
SOFTMAX_OBJECTS := $(BIN_DIR)/softmax_3kernel.o $(BIN_DIR)/softmax_4kernel.o \
	$(BIN_DIR)/softmax_online.o
SOFTMAX_CUDNN_FLAGS :=
SOFTMAX_CUDNN_LIBS :=
ifneq ($(and $(CUDNN_INCLUDE),$(CUDNN_LIBRARY)),)
SOFTMAX_OBJECTS += $(BIN_DIR)/softmax_cudnn.o
SOFTMAX_CUDNN_FLAGS := -DSOFTMAX_HAS_CUDNN -I$(dir $(CUDNN_INCLUDE))
SOFTMAX_CUDNN_LIBS := $(CUDNN_LIBRARY) -Wl,-rpath,$(dir $(CUDNN_LIBRARY))
endif

.PHONY: all help compile-kernels check check-full sanitize clean bench \
        run-reduce run-max run-softmax run-attention run-conv2d run-conv3d \
        run-mv run-gemm run-cat-ce run-mse run-gauss-blur run-max-compare run-top-k \
        run-relu run-leaky-relu run-silu run-swiglu run-clip run-mat-add \
        run-mat-copy run-reverse run-conv1d run-rainbow run-interleave \
        run-sigmoid run-geglu run-rgb2grayscale run-batched-mm run-mm-int8 run-lr run-mc-int \
        run-batched-mm-fp16 sanitize-batched-mm-fp16 \
        run-alibi sanitize-alibi \
        run-dot run-dot-fp16 sanitize-dot sanitize-dot-fp16 \
        sanitize-mc-int run-mat-pow sanitize-mat-pow run-nn sanitize-nn \
        run-batch-norm sanitize-batch-norm \
        run-rms-norm sanitize-rms-norm \
        run-group-norm sanitize-group-norm \
        run-layer-norm sanitize-layer-norm \
        run-max-pooling-2d sanitize-max-pooling-2d \
        run-count sanitize-count \
        run-count3d sanitize-count3d \
        run-slice-sum sanitize-slice-sum \
        run-slice-sum2d sanitize-slice-sum2d \
        run-slice-sum3d sanitize-slice-sum3d \
        run-max-subarray-sum sanitize-max-subarray-sum \
        run-2d-jacobi-stencil run-dequantization run-rope run-sparse-mm run-stream-compaction run-segmented-scan run-adderboard \
        run-gemm-tiled run-gemm-wmma run-gemm-wmma-tiled run-gemm-wmma-tiled-pipeline \
        run-gemm-cublas run-gemm-compare check-gemm \
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
$(BIN_DIR)/softmax_online.o: src/softmax_online.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -Dsolve=softmax_online -c $< -o $@
$(BIN_DIR)/softmax_cudnn.o: src/softmax_cudnn.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(dir $(CUDNN_INCLUDE)) -c $< -o $@
$(BIN_DIR)/softmax_bench: tests/softmax.cpp $(SOFTMAX_OBJECTS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(SOFTMAX_CUDNN_FLAGS) $^ $(SOFTMAX_CUDNN_LIBS) -o $@

$(BIN_DIR)/max_bench: tests/max.cpp src/max.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/attention_bench: tests/attention.cpp src/attention.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/conv2d_bench: tests/conv2d.cpp src/conv2d.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/conv3d_bench: tests/conv3d.cpp src/conv3d.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
# This test includes the implementation to instantiate both kernel variants.
$(BIN_DIR)/mv_bench: tests/mv.cu src/mv.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@
$(BIN_DIR)/gemm_bench: tests/gemm.cu src/gemm.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/gemm_tiled_bench: tests/gemm.cu src/gemm_tiled.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/gemm_wmma_bench: tests/gemm.cu src/gemm_wmma.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/gemm_wmma_tiled_bench: tests/gemm.cu src/gemm_wmma_tiled.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
$(BIN_DIR)/gemm_wmma_tiled_pipeline_bench: tests/gemm.cu src/gemm_wmma_tiled_pipeline.cu | $(BIN_DIR)
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
$(BIN_DIR)/gemm_cublas_bench: tests/gemm.cu src/gemm_cublas.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lcublas
$(BIN_DIR)/gemm_fp32_bench: tests/gemm_fp32.cu src/gemm_fp32.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
# Automatic dispatch uses 128x64 tiles. Force the other tiles with split-K
# so small CPU-reference cases cover their atomic epilogues too.
$(BIN_DIR)/gemm_fp32_split_128_test: tests/gemm_fp32.cu src/gemm_fp32.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -DGEMM_FP32_TILE=1 -DGEMM_FP32_SPLIT=3 $^ -o $@
$(BIN_DIR)/gemm_fp32_split_64_test: tests/gemm_fp32.cu src/gemm_fp32.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -DGEMM_FP32_TILE=3 -DGEMM_FP32_SPLIT=7 $^ -o $@
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
# One reference harness, separate executables and solve signatures for FP32/FP16.
$(BIN_DIR)/dot_test: TEST_DEFINE := DOT_FP32
$(BIN_DIR)/dot_fp16_test: TEST_DEFINE := DOT_FP16
$(addprefix $(BIN_DIR)/,$(DOT_TESTS)): $(BIN_DIR)/%_test: tests/dot.cpp src/%.cu tests/test_utils.h | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -D$(TEST_DEFINE) $(filter %.cpp %.cu,$^) -o $@
# Exercise both optimizers even if the source's default selection is changed.
$(BIN_DIR)/lr_test: NVCCFLAGS += -DLR_OPTIMIZER='"GD"'
$(addprefix $(BIN_DIR)/,$(BASIC_TESTS)): $(BIN_DIR)/%_test: tests/%.cpp src/%.cu tests/test_utils.h | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(filter %.cpp %.cu,$^) -o $@
$(BIN_DIR)/lr_newton_test: tests/lr.cpp src/lr.cu tests/test_utils.h | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -DLR_OPTIMIZER='"Newton"' $(filter %.cpp %.cu,$^) -o $@

$(BIN_DIR)/reduce_max_compare: benchmarks/reduce_max.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run-reduce bench: $(BIN_DIR)/reduce_bench
	$<
run-max: $(BIN_DIR)/max_bench
	$<
run-softmax: $(BIN_DIR)/softmax_bench
	$< $(SOFTMAX_ARGS)
run-attention: $(BIN_DIR)/attention_bench
	$<
run-conv2d: $(BIN_DIR)/conv2d_bench
	$<
run-conv3d: $(BIN_DIR)/conv3d_bench
	$<
run-mv: $(BIN_DIR)/mv_bench
	$<
run-gemm: $(BIN_DIR)/gemm_bench
	$<
run-gemm-tiled: $(BIN_DIR)/gemm_tiled_bench
	$<
run-gemm-wmma: $(BIN_DIR)/gemm_wmma_bench
	$<
run-gemm-wmma-tiled: $(BIN_DIR)/gemm_wmma_tiled_bench
	$<
run-gemm-wmma-tiled-pipeline: $(BIN_DIR)/gemm_wmma_tiled_pipeline_bench
	$< $(GEMM_ARGS)
run-gemm-wmma-tiled-pipeline-schedule: $(BIN_DIR)/gemm_wmma_tiled_pipeline_schedule_bench
	$< $(GEMM_ARGS)
run-gemm-cublas: $(BIN_DIR)/gemm_cublas_bench
	$<
run-gemm-compare: $(addprefix $(BIN_DIR)/,$(GEMM_BENCHES))
	@set -e; for binary in $^; do \
	  echo "$$binary"; "$$binary" $(GEMM_ARGS); \
	done
check-gemm: $(addprefix $(BIN_DIR)/,$(GEMM_BENCHES)) $(GEMM_SCHEDULE_TESTS)
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
run-batched-mm-fp16: $(BIN_DIR)/batched_mm_fp16_test
	$< $(BATCHED_MM_FP16_ARGS)
run-alibi: $(BIN_DIR)/alibi_test
	$< $(ALIBI_ARGS)
run-dot: $(BIN_DIR)/dot_test
	$< $(DOT_ARGS)
run-dot-fp16: $(BIN_DIR)/dot_fp16_test
	$< $(DOT_FP16_ARGS)
run-mm-int8: $(BIN_DIR)/mm_int8_test
	$< $(MM_INT8_ARGS)
run-lr: $(BIN_DIR)/lr_test $(BIN_DIR)/lr_newton_test
	$(BIN_DIR)/lr_test
	$(BIN_DIR)/lr_newton_test
run-mc-int: $(BIN_DIR)/mc_int_test
	$< $(MC_INT_ARGS)
run-mat-pow: $(BIN_DIR)/mat_pow_test
	$< $(MAT_POW_ARGS)
run-nn: $(BIN_DIR)/nn_test
	$< $(NN_ARGS)
run-batch-norm: $(BIN_DIR)/batch_norm_test
	$< $(BN_ARGS)
run-rms-norm: $(BIN_DIR)/rms_norm_test
	$< $(RMS_ARGS)
run-group-norm: $(BIN_DIR)/group_norm_test
	$< $(GN_ARGS)
run-layer-norm: $(BIN_DIR)/layer_norm_test
	$< $(LN_ARGS)
run-max-pooling-2d: $(BIN_DIR)/max_pooling_2d_test
	$< $(POOL_ARGS)
run-count: $(BIN_DIR)/count_test
	$< $(COUNT_ARGS)
run-count3d: $(BIN_DIR)/count3d_test
	$< $(COUNT3D_ARGS)
run-slice-sum: $(BIN_DIR)/slice_sum_test
	$< $(SLICE_SUM_ARGS)
run-slice-sum2d: $(BIN_DIR)/slice_sum2d_test
	$< $(SLICE_SUM2D_ARGS)
run-slice-sum3d: $(BIN_DIR)/slice_sum3d_test
	$< $(SLICE_SUM3D_ARGS)
run-max-subarray-sum: $(BIN_DIR)/max_subarray_sum_test
	$< $(MAX_SUBARRAY_SUM_ARGS)
run-2d-jacobi-stencil: $(BIN_DIR)/2d_jacobi_stencil_test
	$<
run-dequantization: $(BIN_DIR)/dequantization_test
	$<
run-rope: $(BIN_DIR)/rope_test
	$<
run-sparse-mm: $(BIN_DIR)/sparse_mm_test
	$<
run-stream-compaction: $(BIN_DIR)/stream_compaction_test
	$<
run-segmented-scan: $(BIN_DIR)/segmented_scan_test
	$<
run-fft2d: $(BIN_DIR)/fft2d_test
	$< $(FFT2D_ARGS)
run-adderboard: $(BIN_DIR)/adderboard_test
	$<

# Small reproducible GPU checks. Every executable returns nonzero on failure.
check: all $(GEMM_SCHEDULE_TESTS)
	$(BIN_DIR)/reduce_bench 1025 2
	$(BIN_DIR)/max_bench 1025 2
	$(BIN_DIR)/softmax_bench 1025 2 1
	$(BIN_DIR)/softmax_bench 4097 2 1
	$(BIN_DIR)/attention_bench 17 33 16 2 1
	$(BIN_DIR)/conv2d_bench 17 35 3 5 2 1
	$(BIN_DIR)/conv3d_bench 9 11 13 3 3 3 2 1
	$(BIN_DIR)/mv_bench --check-only
	$(BIN_DIR)/gemm_bench --check-only
	$(BIN_DIR)/gemm_tiled_bench --check-only
	$(BIN_DIR)/gemm_wmma_bench --check-only
	$(BIN_DIR)/gemm_wmma_tiled_bench --check-only
	$(BIN_DIR)/gemm_wmma_tiled_pipeline_bench --check-only
	$(BIN_DIR)/gemm_wmma_tiled_pipeline_schedule_bench --check-only
	@set -e; for binary in $(GEMM_SCHEDULE_TESTS); do "$$binary" --check-only; done
	$(BIN_DIR)/gemm_cublas_bench --check-only
	$(BIN_DIR)/gemm_fp32_bench --check-only
	$(BIN_DIR)/gemm_fp32_split_128_test --check-only
	$(BIN_DIR)/gemm_fp32_split_64_test --check-only
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
	$(BIN_DIR)/batched_mm_fp16_test
	$(BIN_DIR)/alibi_test
	$(BIN_DIR)/dot_test
	$(BIN_DIR)/dot_fp16_test
	$(BIN_DIR)/mm_int8_test
	$(BIN_DIR)/lr_test
	$(BIN_DIR)/lr_newton_test
	$(BIN_DIR)/mc_int_test
	$(BIN_DIR)/mat_pow_test
	$(BIN_DIR)/nn_test
	$(BIN_DIR)/batch_norm_test
	$(BIN_DIR)/rms_norm_test
	$(BIN_DIR)/group_norm_test
	$(BIN_DIR)/layer_norm_test
	$(BIN_DIR)/max_pooling_2d_test
	$(BIN_DIR)/count_test
	$(BIN_DIR)/count3d_test
	$(BIN_DIR)/slice_sum_test
	$(BIN_DIR)/slice_sum2d_test
	$(BIN_DIR)/slice_sum3d_test
	$(BIN_DIR)/max_subarray_sum_test
	$(BIN_DIR)/2d_jacobi_stencil_test
	$(BIN_DIR)/dequantization_test
	$(BIN_DIR)/rope_test
	$(BIN_DIR)/sparse_mm_test
	$(BIN_DIR)/stream_compaction_test
	$(BIN_DIR)/segmented_scan_test
	$(BIN_DIR)/fft2d_test
	$(BIN_DIR)/adderboard_test
ifeq ($(WITH_TRITON),1)
	$(MAKE) check-gemm-triton
endif

# Large regression suites.
check-full: check
	$(BIN_DIR)/mse_test
	$(BIN_DIR)/top_k_test 50000000 100
	$(BIN_DIR)/batched_mm_fp16_test --large
	$(BIN_DIR)/alibi_test --large
	$(BIN_DIR)/dot_test --large
	$(BIN_DIR)/dot_fp16_test --large
	$(BIN_DIR)/mm_int8_test --large
	$(BIN_DIR)/mc_int_test --large
	$(BIN_DIR)/mat_pow_test --large
	$(BIN_DIR)/nn_test --large
	$(BIN_DIR)/batch_norm_test --large
	$(BIN_DIR)/rms_norm_test --large
	$(BIN_DIR)/group_norm_test --large
	$(BIN_DIR)/layer_norm_test --large
	$(BIN_DIR)/max_pooling_2d_test --large
	$(BIN_DIR)/count_test --large
	$(BIN_DIR)/count3d_test --large
	$(BIN_DIR)/slice_sum_test --large
	$(BIN_DIR)/slice_sum2d_test --large
	$(BIN_DIR)/slice_sum3d_test --large
	$(BIN_DIR)/max_subarray_sum_test --large
	$(BIN_DIR)/adderboard_test --large

COMPUTE_SANITIZER ?= compute-sanitizer
sanitize-dot: $(BIN_DIR)/dot_test
	@dot_status=0; \
	$(COMPUTE_SANITIZER) --tool memcheck --leak-check full --print-limit 20 --error-exitcode 1 $< $(DOT_ARGS) || dot_status=1; \
	for tool in initcheck racecheck synccheck; do \
	  $(COMPUTE_SANITIZER) --tool $$tool --print-limit 20 --error-exitcode 1 $< $(DOT_ARGS) || dot_status=1; \
	done; exit $$dot_status
sanitize-dot-fp16: $(BIN_DIR)/dot_fp16_test
	@dot_fp16_status=0; \
	$(COMPUTE_SANITIZER) --tool memcheck --leak-check full --print-limit 20 --error-exitcode 1 $< $(DOT_FP16_ARGS) || dot_fp16_status=1; \
	for tool in initcheck racecheck synccheck; do \
	  $(COMPUTE_SANITIZER) --tool $$tool --print-limit 20 --error-exitcode 1 $< $(DOT_FP16_ARGS) || dot_fp16_status=1; \
	done; exit $$dot_fp16_status
sanitize-alibi: $(BIN_DIR)/alibi_test
	@alibi_status=0; \
	$(COMPUTE_SANITIZER) --tool memcheck --leak-check full --print-limit 20 --error-exitcode 1 $< $(ALIBI_ARGS) || alibi_status=1; \
	for tool in initcheck racecheck synccheck; do \
	  $(COMPUTE_SANITIZER) --tool $$tool --print-limit 20 --error-exitcode 1 $< $(ALIBI_ARGS) || alibi_status=1; \
	done; exit $$alibi_status
sanitize-batched-mm-fp16: $(BIN_DIR)/batched_mm_fp16_test
	@fp16_status=0; \
	$(COMPUTE_SANITIZER) --tool memcheck --leak-check full --print-limit 20 --error-exitcode 1 $< $(BATCHED_MM_FP16_ARGS) || fp16_status=1; \
	for tool in initcheck racecheck synccheck; do \
	  $(COMPUTE_SANITIZER) --tool $$tool --print-limit 20 --error-exitcode 1 $< $(BATCHED_MM_FP16_ARGS) || fp16_status=1; \
	done; exit $$fp16_status
sanitize-mc-int: $(BIN_DIR)/mc_int_test
	@status=0; for tool in memcheck initcheck racecheck synccheck; do \
	  $(COMPUTE_SANITIZER) --tool $$tool --error-exitcode 1 $< || status=1; \
	done; exit $$status
sanitize-mat-pow: $(BIN_DIR)/mat_pow_test
	@mat_pow_status=0; \
	$(COMPUTE_SANITIZER) --tool memcheck --leak-check full --error-exitcode 1 $< || mat_pow_status=1; \
	$(COMPUTE_SANITIZER) --tool initcheck --error-exitcode 1 $< || mat_pow_status=1; \
	exit $$mat_pow_status
sanitize-nn: $(BIN_DIR)/nn_test
	@nn_status=0; \
	$(COMPUTE_SANITIZER) --tool memcheck --leak-check full --print-limit 20 --error-exitcode 1 $< $(NN_ARGS) || nn_status=1; \
	$(COMPUTE_SANITIZER) --tool initcheck --print-limit 20 --error-exitcode 1 $< $(NN_ARGS) || nn_status=1; \
	exit $$nn_status
sanitize-batch-norm: $(BIN_DIR)/batch_norm_test
	@bn_status=0; \
	$(COMPUTE_SANITIZER) --tool memcheck --leak-check full --error-exitcode 1 $< $(BN_ARGS) || bn_status=1; \
	$(COMPUTE_SANITIZER) --tool initcheck --error-exitcode 1 $< $(BN_ARGS) || bn_status=1; \
	exit $$bn_status
sanitize-rms-norm: $(BIN_DIR)/rms_norm_test
	@rms_status=0; \
	$(COMPUTE_SANITIZER) --tool memcheck --leak-check full --print-limit 20 --error-exitcode 1 $< $(RMS_ARGS) || rms_status=1; \
	for tool in initcheck racecheck synccheck; do \
	  $(COMPUTE_SANITIZER) --tool $$tool --print-limit 20 --error-exitcode 1 $< $(RMS_ARGS) || rms_status=1; \
	done; exit $$rms_status
sanitize-group-norm: $(BIN_DIR)/group_norm_test
	$(COMPUTE_SANITIZER) --tool memcheck --leak-check full --print-limit 20 --error-exitcode 1 $< $(GN_ARGS)
	@gn_status=0; for tool in initcheck racecheck synccheck; do \
	  $(COMPUTE_SANITIZER) --tool $$tool --print-limit 20 --error-exitcode 1 $< $(GN_ARGS) || gn_status=1; \
	done; exit $$gn_status
sanitize-layer-norm: $(BIN_DIR)/layer_norm_test
	@ln_status=0; \
	$(COMPUTE_SANITIZER) --tool memcheck --leak-check full --print-limit 20 --error-exitcode 1 $< $(LN_ARGS) || ln_status=1; \
	for tool in initcheck racecheck synccheck; do \
	  $(COMPUTE_SANITIZER) --tool $$tool --print-limit 20 --error-exitcode 1 $< $(LN_ARGS) || ln_status=1; \
	done; exit $$ln_status
sanitize-max-pooling-2d: $(BIN_DIR)/max_pooling_2d_test
	@pool_status=0; \
	$(COMPUTE_SANITIZER) --tool memcheck --leak-check full --print-limit 20 --error-exitcode 1 $< $(POOL_ARGS) || pool_status=1; \
	$(COMPUTE_SANITIZER) --tool initcheck --print-limit 20 --error-exitcode 1 $< $(POOL_ARGS) || pool_status=1; \
	exit $$pool_status
sanitize-count: $(BIN_DIR)/count_test
	@count_status=0; for tool in memcheck initcheck racecheck synccheck; do \
	  $(COMPUTE_SANITIZER) --tool $$tool --print-limit 20 --error-exitcode 1 $< $(COUNT_ARGS) || count_status=1; \
	done; exit $$count_status
sanitize-count3d: $(BIN_DIR)/count3d_test
	@count3d_status=0; for tool in memcheck initcheck racecheck synccheck; do \
	  $(COMPUTE_SANITIZER) --tool $$tool --print-limit 20 --error-exitcode 1 $< $(COUNT3D_ARGS) || count3d_status=1; \
	done; exit $$count3d_status
sanitize-slice-sum: $(BIN_DIR)/slice_sum_test
	@slice_sum_status=0; for tool in memcheck initcheck racecheck synccheck; do \
	  $(COMPUTE_SANITIZER) --tool $$tool --print-limit 20 --error-exitcode 1 $< $(SLICE_SUM_ARGS) || slice_sum_status=1; \
	done; exit $$slice_sum_status
sanitize-slice-sum2d: $(BIN_DIR)/slice_sum2d_test
	@slice_sum2d_status=0; for tool in memcheck initcheck racecheck synccheck; do \
	  $(COMPUTE_SANITIZER) --tool $$tool --print-limit 20 --error-exitcode 1 $< $(SLICE_SUM2D_ARGS) || slice_sum2d_status=1; \
	done; exit $$slice_sum2d_status
sanitize-slice-sum3d: $(BIN_DIR)/slice_sum3d_test
	@slice_sum3d_status=0; for tool in memcheck initcheck racecheck synccheck; do \
	  $(COMPUTE_SANITIZER) --tool $$tool --print-limit 20 --error-exitcode 1 $< $(SLICE_SUM3D_ARGS) || slice_sum3d_status=1; \
	done; exit $$slice_sum3d_status
sanitize-max-subarray-sum: $(BIN_DIR)/max_subarray_sum_test
	@max_subarray_sum_status=0; \
	$(COMPUTE_SANITIZER) --tool memcheck --leak-check full --print-limit 20 --error-exitcode 1 $< $(MAX_SUBARRAY_SUM_ARGS) || max_subarray_sum_status=1; \
	for tool in initcheck racecheck synccheck; do \
	  $(COMPUTE_SANITIZER) --tool $$tool --print-limit 20 --error-exitcode 1 $< $(MAX_SUBARRAY_SUM_ARGS) || max_subarray_sum_status=1; \
	done; exit $$max_subarray_sum_status
sanitize: all $(GEMM_SCHEDULE_TESTS)
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/gemm_bench 17 33 19 0
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_bench 65 129 67 0
	$(COMPUTE_SANITIZER) --tool racecheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_bench 65 129 67 0
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_bench --check-only
	$(COMPUTE_SANITIZER) --tool racecheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_bench --check-only
	$(COMPUTE_SANITIZER) --tool synccheck --error-exitcode 1 $(BIN_DIR)/gemm_wmma_tiled_pipeline_bench --check-only
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
	$(MAKE) sanitize-batched-mm-fp16
	$(MAKE) sanitize-alibi
	$(MAKE) sanitize-dot
	$(MAKE) sanitize-dot-fp16
	$(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 1 $(BIN_DIR)/mm_int8_test
	$(COMPUTE_SANITIZER) --tool memcheck --leak-check full --error-exitcode 1 $(BIN_DIR)/lr_test
	$(COMPUTE_SANITIZER) --tool memcheck --leak-check full --error-exitcode 1 $(BIN_DIR)/lr_newton_test
	$(MAKE) sanitize-mc-int
	$(MAKE) sanitize-mat-pow
	$(MAKE) sanitize-nn
	$(MAKE) sanitize-batch-norm
	$(MAKE) sanitize-rms-norm
	$(MAKE) sanitize-group-norm
	$(MAKE) sanitize-layer-norm
	$(MAKE) sanitize-max-pooling-2d
	$(MAKE) sanitize-count
	$(MAKE) sanitize-count3d
	$(MAKE) sanitize-slice-sum
	$(MAKE) sanitize-slice-sum2d
	$(MAKE) sanitize-slice-sum3d
	$(MAKE) sanitize-max-subarray-sum
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
	@echo 'check-full      Also run large MSE, top-k, FP16 batched/INT8 matmul, ALiBi, FP32/FP16 dot, Monte Carlo, matrix power, nearest-neighbor, batch-norm, RMS-norm, group-norm, layer-norm, max-pooling, count, subarray-sum and max-subarray-sum regressions'
	@echo 'sanitize        Run selected memory/synchronization checks'
	@echo 'run-<operator>  Run one test/benchmark with default arguments'
	@echo 'run-batched-mm-fp16 Check FP16 batched MM; BATCHED_MM_FP16_ARGS="--case NAME" isolates a case'
	@echo '               --large checks 1024 dimensions and B=128,M=N=K=256; --list-cases lists cases'
	@echo 'sanitize-batched-mm-fp16 Run memcheck/initcheck/racecheck/synccheck; accepts BATCHED_MM_FP16_ARGS'
	@echo 'run-alibi      Check ALiBi; ALIBI_ARGS="--case NAME" isolates a case'
	@echo '               --large checks M/N up to 2048,d up to 1024; --list-cases lists cases'
	@echo 'sanitize-alibi Run ALiBi memcheck/leak/initcheck/racecheck/synccheck; accepts ALIBI_ARGS'
	@echo 'run-dot        Check FP32 dot; DOT_ARGS="--case NAME" isolates a case'
	@echo 'run-dot-fp16   Check FP16 dot; DOT_FP16_ARGS="--case NAME" isolates a case'
	@echo '               --large checks N=99999999/100000000 and decimal precision; --list-cases lists cases'
	@echo 'sanitize-dot / sanitize-dot-fp16 Run all four sanitizer tools; accept the respective args'
	@echo 'run-mm-int8     Check INT8 quantization; MM_INT8_ARGS=--large checks 8192x4096x2048'
	@echo 'run-mc-int      Check Monte Carlo integration; MC_INT_ARGS=--large checks 10M/100M samples'
	@echo 'sanitize-mc-int Run Monte Carlo memcheck/initcheck/racecheck/synccheck checks'
	@echo 'run-mat-pow     Check matrix powers; MAT_POW_ARGS=--large checks N=511/512/1023/1024'
	@echo 'sanitize-mat-pow Run matrix power memory/leak/initialization checks'
	@echo 'run-nn         Check nearest neighbors; NN_ARGS="--case NAME" isolates a case'
	@echo '               NN_ARGS=--large checks 10K/100K points; --list-cases lists quick cases'
	@echo 'sanitize-nn    Run nearest-neighbor memory/leak/initialization checks; accepts NN_ARGS'
	@echo 'run-batch-norm Check batch normalization; BN_ARGS="--case NAME" isolates a case'
	@echo '               BN_ARGS=--large checks N=5000/10000,C=1024; --list-cases lists quick cases'
	@echo 'sanitize-batch-norm Run batch-norm memory/leak/initialization checks; accepts BN_ARGS'
	@echo 'run-rms-norm   Check RMS normalization; RMS_ARGS="--case NAME" isolates a case'
	@echo '               RMS_ARGS=--large checks N=99999/100000; --list-cases lists cases'
	@echo 'sanitize-rms-norm Run RMS-norm memcheck/initcheck/racecheck/synccheck; accepts RMS_ARGS'
	@echo 'run-group-norm Check NCHW group normalization; GN_ARGS="--case NAME" isolates a case'
	@echo '               GN_ARGS=--large checks challenge/spatial sizes; --list-cases lists cases'
	@echo 'sanitize-group-norm Run group-norm memcheck, then initcheck/racecheck/synccheck if it passes; accepts GN_ARGS'
	@echo 'run-layer-norm Check per-row layer normalization; LN_ARGS="--case NAME" isolates a case'
	@echo '               LN_ARGS=--large checks N/C=65536/512 and 1024/4096; --list-cases lists cases'
	@echo 'sanitize-layer-norm Run layer-norm memcheck/initcheck/racecheck/synccheck; accepts LN_ARGS'
	@echo 'run-max-pooling-2d Check NCHW max pooling; POOL_ARGS="--case NAME" isolates a case'
	@echo '               POOL_ARGS=--large checks N=4,k=3,s=2; --list-cases lists cases'
	@echo 'sanitize-max-pooling-2d Run max-pooling memory/leak/initialization checks; accepts POOL_ARGS'
	@echo 'run-count      Check integer occurrence counts; COUNT_ARGS="--case NAME" isolates a case'
	@echo '               COUNT_ARGS=--large checks 16,777,217/100M elements; --list-cases lists cases'
	@echo 'sanitize-count Run count memcheck/initcheck/racecheck/synccheck checks; accepts COUNT_ARGS'
	@echo 'run-count3d    Check 3D occurrence counts; COUNT3D_ARGS="--case NAME" isolates a case'
	@echo '               COUNT3D_ARGS=--large checks 500^3/1000^3 elements; --list-cases lists cases'
	@echo 'sanitize-count3d Run 3D count memcheck/initcheck/racecheck/synccheck; accepts COUNT3D_ARGS'
	@echo 'run-slice-sum  Check inclusive subarray sums; SLICE_SUM_ARGS="--case NAME" isolates a case'
	@echo '               SLICE_SUM_ARGS=--large checks 100M elements; --list-cases lists cases'
	@echo 'sanitize-slice-sum Run subarray-sum memory/race/synchronization checks; accepts SLICE_SUM_ARGS'
	@echo 'run-slice-sum2d Check rectangular sums; SLICE_SUM2D_ARGS="--case NAME" isolates a case'
	@echo '               SLICE_SUM2D_ARGS=--large checks 10000x10000; --list-cases lists cases'
	@echo 'sanitize-slice-sum2d Run 2D subarray-sum memory/race/synchronization checks; accepts SLICE_SUM2D_ARGS'
	@echo 'run-slice-sum3d Check 3D region sums; SLICE_SUM3D_ARGS="--case NAME" isolates a case'
	@echo '               SLICE_SUM3D_ARGS=--large checks 500^3; --list-cases lists cases'
	@echo 'sanitize-slice-sum3d Run 3D subarray-sum memory/race/synchronization checks; accepts SLICE_SUM3D_ARGS'
	@echo 'run-max-subarray-sum Check fixed-length maximum window sums; MAX_SUBARRAY_SUM_ARGS="--case NAME" isolates a case'
	@echo '               MAX_SUBARRAY_SUM_ARGS=--large checks N=50000; --list-cases lists cases'
	@echo 'sanitize-max-subarray-sum Run memory/leak/initialization/race/synchronization checks; accepts MAX_SUBARRAY_SUM_ARGS'
	@echo 'check-gemm      Check scalar, tiled, representative WMMA stages, and cuBLAS GEMM'
	@echo 'run-gemm-compare Compare seven GEMM implementations; GEMM_ARGS="M N K repeats"'
	@echo 'run-gemm-wmma-tiled-pipeline Run async/double-buffered WMMA; uses GEMM_ARGS'
	@echo 'run-gemm-wmma-tiled-pipeline-schedule Run distributed GEMM mainloop; uses GEMM_ARGS'
	@echo 'gemm*_perf      Optional large-shape binaries in BIN_DIR; M N K [iterations|--profile]'
	@echo 'check-gemm-triton Reuse CUDA GEMM cases for Triton, plus all autotune candidates'
	@echo '                 Set PYTHON to a CUDA PyTorch/Triton environment; WITH_TRITON=1 includes it in check/check-gemm'
	@echo '                 TRITON_SOURCE=src/gemm.triton.v2.py (or v3.py) selects another version'
	@echo 'bench-gemm-triton-compare Compare Triton, CUDA schedule, and cuBLAS on identical buffers'
	@echo '                 TRITON_BENCH_ARGS="--shape M N K [--verify-only]"; TRITON_CUDA_GEMM selects the CUDA source'
	@echo '                 Repeat --reference-source src/gemm.triton.py to compare multiple Triton versions together'
	@echo 'sanitize-gemm-triton Check Triton tails/unaligned pointers with all three sanitizer tools'
	@echo 'nsys-gemm       Profile the WMMA stages and cuBLAS serially, then print stats'
	@echo '                Set CUDA_VISIBLE_DEVICES, GEMM_ARGS, NSYS_DIR, NSYS_FLAGS as needed'
	@echo 'nsys-gemm-stats  Print existing reports in NSYS_DIR (default: $(BIN_DIR)/nsys)'
	@echo 'ncu-gemm-build  Build seven GEMM implementations with -lineinfo in NCU_BIN_DIR'
	@echo 'ncu-gemm        Profile seven implementations serially, then export text/CSV and print a summary'
	@echo '                Set CUDA_VISIBLE_DEVICES, GEMM_ARGS, NCU_DIR, NCU_SET, NCU_RUN as needed'
	@echo 'ncu-gemm-stats  Export existing NCU reports and regenerate comparison.csv (no GPU needed)'
	@echo 'clean           Remove current architecture build directory'

clean:
	rm -rf $(BIN_DIR)
