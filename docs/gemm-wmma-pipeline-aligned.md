# WMMA pipeline with a full, aligned specialization

This page records the initial specialization at commit `5eed6d5`. The separate
shared-layout successor and its measurements are described in the
[shared-layout experiment](gemm-wmma-shared-layout.md).

`src/gemm_wmma_tiled_pipeline_aligned.cu` is an independent successor to
`gemm_wmma_tiled_pipeline.cu`. It implements the first experiment proposed in
the [pipeline NCU analysis](gemm-wmma-pipeline-ncu.md): remove input checks and
repeated address calculations when the entire matrix satisfies the tile layout.
The original pipeline source remains the comparison baseline.

The follow-up [cuBLAS gap analysis](gemm-wmma-pipeline-aligned-ncu.md) profiles
1024 and 2048 cubed, quantifies remaining data movement and shared-memory costs,
and ranks the next experiments.

## Dispatch and implementation

The contract remains contiguous, row-major FP16 A[M,K], B[K,N], C[M,N], FP32
accumulation, and `C = alpha * A * B + beta * C`. Dimensions are nonnegative;
M/N=0 is a no-op, K=0 applies beta to C, and beta=0 does not read C.

With the default BM=64, BN=64, BK=32, WM=WN=32, SKEW=16, `solve` selects
`gemm_wmma_tiled_pipeline_aligned<true, ...>` when:

- M, N, K are positive and divisible by BM, BN, BK, respectively.
- Both A and B base addresses are divisible by 16 bytes.

The complete tiles guarantee that every eight-half copy is in bounds. Because
BN and BK are multiples of 16 half elements, the input row strides also preserve
16-byte alignment. C needs only half alignment: the epilogue uses scalar stores.
The host checks the conditions once per call, then launches one specialization.
Partial tiles, unaligned A/B pointers, and K=0 select `<false, ...>`, which retains
the original generic staging and output bounds checks. No matrix padding, extra
allocation, or extra kernel is introduced.

Each thread in the fast path owns two 16-byte A copies and two B copies per
K chunk. It computes their source pointers and shared-memory offsets once per
output tile. Between chunks, A pointers advance by BK half elements and B
pointers by BK*N. There is no advance or prefetch beyond the last valid chunk.
Fixed copy loops are unrolled; no per-copy matrix bounds or alignment checks are
needed. The fast epilogue also omits output bounds checks. Compile-time tile
overrides remain available through the existing `GEMM_*` macros.

Both paths retain the original two-stage input buffers and synchronization:
prefetch/commit/wait/barrier for the first chunk, then prefetch the next chunk,
compute the current chunk, wait, and synchronize the block before reuse.
Warp reconvergence before commit is retained, including for tile overrides where
some threads have no copies. On sm_75 the same specialization uses synchronous
staging; sm_80+ uses `__pipeline_memcpy_async(..., 16)`.

## Validation

Validated on 2026-09-13 with NVIDIA A800 80GB PCIe, CUDA 12.6.20, driver
535.247.01, and `nvcc -O3 -std=c++14 -arch=sm_80 -Xcompiler -Wall`:

- `CUDA_VISIBLE_DEVICES=4 make -j2 check` passed, including all seven GEMMs.
- `CUDA_VISIBLE_DEVICES=4 make sanitize` passed. Both pipeline versions passed
  all 28 correctness cases under memcheck, racecheck, and synccheck, with zero
  errors or hazards.
- The CPU reference checks two calls per case, output guards, nontrivial
  alpha/beta, one/two/three K chunks, multiple complete output tiles, independently
  partial M/N/K dimensions, K=0, and empty output dimensions. Misaligned A/B cases
  now use otherwise complete tile dimensions, isolating pointer dispatch. A
  complete-tile case with unaligned C checks that C alignment is not required.
- sm_75 compiled successfully. Native T4 execution was unavailable. A separate
  `-gencode arch=compute_75,code=sm_80` build exercised the synchronous source
  branch on A800: all 28 cases passed, and SASS contained no LDGSTS instructions.
- A `GEMM_WM=16, GEMM_WN=16` build passed all 28 cases under synccheck. This
  configuration has 512 threads and 256 copy groups per input tile, exercising
  the copy-work guard for threads without input copies.

The default sm_80 binary contains both specializations, with LDGSTS and HMMA
instructions. `cuobjdump --dump-resource-usage` reports 26,624 bytes of static
shared memory per block for both, no stack or local memory, 62 registers/thread
for the generic specialization and 92 for the full/aligned specialization.

## Ordinary benchmark results

Measured serially on physical GPU 5, A800 80GB PCIe. All variants used identical
aligned buffers, alpha=1, beta=0, five warmups, and the median of five batches of
100 calls, and passed CPU-reference validation for every shape. CUDA-event timing
excludes allocations, transfers, and CPU validation, and includes GPU execution
and any host submission gaps. Clocks were not locked.

| M x N x K | Pipeline (us) | Aligned pipeline (us) | Speedup | cuBLAS (us) |
|---|---:|---:|---:|---:|
| 128 x 128 x 128 | 8.458 | 5.386 | 1.57x | 7.322 |
| 512 x 512 x 512 | 18.924 | 10.476 | 1.81x | 7.393 |
| 1024 x 1024 x 1024 | 43.960 | 29.082 | 1.51x | 15.329 |
| 128 x 192 x 96 | 7.516 | 5.018 | 1.50x | 4.884 |
| 1009 x 513 x 257 | 27.761 | 27.873 | 1.00x | 19.220 |

The tail shape uses the generic path and remains approximately unchanged. At
1024 cubed, the new version takes 1.90x cuBLAS time, down from 2.87x for the
original pipeline. Small-shape results include host submission overhead and do
not imply a general advantage over cuBLAS.

## NCU confirmation

A fresh Nsight Compute 2024.3.0 collection compared all three variants serially
on GPU 5 at M=N=K=1024. It used `--set full`, kernel replay, cache flushing,
unmodified clocks, `--launch-skip 7 --launch-count 1`, and binaries with
`-lineinfo`. The recorded custom kernel is explicitly the `<1, 64, 64, 32, 32,
32, 16>` specialization. These are captured kernel durations; the benchmark's
own event timing while under NCU is affected by profiling and is not used.

| Metric | Pipeline | Aligned pipeline | cuBLAS |
|---|---:|---:|---:|
| Kernel duration (us) | 45.984 | 30.656 | 17.248 |
| Executed warp instructions | 9,336,832 | 3,326,976 | 1,701,216 |
| HMMA warp instructions | 524,288 | 524,288 | 540,672 |
| HMMA share of instructions | 5.62% | 15.76% | 31.78% |
| Tensor active (% of active cycles) | 16.39 | 25.24 | 58.99 |
| Registers/thread | 62 | 92 | 196 |
| Theoretical occupancy (%) | 37.50 | 31.25 | 12.50 |
| Achieved occupancy (%) | 14.72 | 14.79 | 6.16 |
| Shared-load bank conflicts | 1,048,576 | 1,048,576 | 0 |
| DRAM throughput (% of peak) | 4.75 | 7.12 | 12.74 |
| Long-scoreboard cycles per issued instruction | 0.97 | 5.20 | 0.42 |

Total executed instructions fell 64.4% while the HMMA count stayed constant.
The integer/address/logic opcode group used in the previous analysis fell from
6,175,744 to 1,812,480 instructions. BRA/BSSY/BSYNC counts fell from
537,600/364,544/233,472 to 65,536/4,096/4,096. Together with the timings, this
confirms that removing control and address work improves the complete-tile case.

The extra persistent addresses and changed scheduling increase register use;
the theoretical block limit becomes five per SM rather than six. Actual
occupancy stays near 14.8% for this 256-block grid on 108 SMs. The fast path has
no local-memory spills, but the register cost should be checked on larger grids.

Shared-load bank conflicts are unchanged. Long-scoreboard stalls become more
prominent after removing other instructions: 1,629 of 1,808 sampled occurrences
are at two HMMA PCs, so investigate their preceding input dependencies. The
5.20 value is normalized per issued instruction, whose count also decreased;
it is not a wall-time slowdown. Shared-memory layout and scheduling the copy
waits remain useful next experiments. These metrics do not establish that a
third buffer alone will fix the remaining gap. PM sampling dropped three samples
per report and was too coarse for reliable phase-level conclusions.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=4 make check
CUDA_VISIBLE_DEVICES=4 make sanitize
CUDA_VISIBLE_DEVICES=5 make run-gemm-wmma-tiled-pipeline-aligned GEMM_ARGS="1024 1024 1024 100"
CUDA_VISIBLE_DEVICES=5 build/sm_80/gemm_wmma_tiled_pipeline_bench 1024 1024 1024 100
CUDA_VISIBLE_DEVICES=5 build/sm_80/gemm_cublas_bench 1024 1024 1024 100

make -j2 ncu-gemm-build NCU_GEMMS="gemm_wmma_tiled_pipeline gemm_wmma_tiled_pipeline_aligned gemm_cublas"
# Requires counter access; see the README for temporary sudo execution.
CUDA_VISIBLE_DEVICES=5 make ncu-gemm \
  NCU_GEMMS="gemm_wmma_tiled_pipeline gemm_wmma_tiled_pipeline_aligned gemm_cublas" \
  GEMM_ARGS="1024 1024 1024 2" NCU_DIR=build/ncu-aligned-1024
```

The new version is also included in `check-gemm`, `run-gemm-compare`, and
`nsys-gemm`. Generated captures are excluded from version control.
