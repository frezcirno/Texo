# Multi-warp WMMA tiling experiment

The new `src/gemm_wmma_tiled.cu` keeps the single-warp example intact and adds
cooperative block loads and multiple accumulator fragments per warp. It uses
ordinary synchronous loads, with no asynchronous copy or double buffering.

## Default mapping

- Block output: 64x64; K staging chunk: 32.
- Four warps (128 threads), arranged as a 2x2 grid of 32x32 output regions.
- Each warp holds four 16x16 FP32 accumulator fragments in a 2x2 arrangement.
- Each loaded A fragment is reused across two B fragments, and vice versa.
- All warps load the shared A[64][32] and B[32][64] tiles cooperatively.
- Shared rows have 16 extra half elements; these are padding, not input data.
- Two block barriers protect each K chunk. Each warp uses its own 16x16 FP32
  scratch tile and warp barriers to store and convert its four output fragments.
- Partial M/N/K tiles are zero-padded; only valid C elements are stored.

| Output rows / columns within block | 0..31 | 32..63 |
|---|---|---|
| 0..31 | warp 0 | warp 1 |
| 32..63 | warp 2 | warp 3 |

At a fixed K chunk, increasing the output block from 16x16 to 64x64 increases
logical input reuse from 8 to 32 FLOP per byte loaded from global addresses.
This is a source-level traffic calculation, not measured DRAM traffic: caches
can satisfy repeated loads. Increasing BK alone does not change this ratio.

## Measurements

Measured on 2026-09-13, NVIDIA A800 80GB PCIe, physical GPU 3, CUDA 12.6,
`nvcc -O3 -lineinfo -std=c++14 -arch=sm_80`. FP16 inputs/output, FP32 accumulation,
alpha=1, beta=0. Buffers are 256-byte aligned. Five warmups precede five batches
of 100 calls; the table reports median batch time per call, in microseconds.
CUDA events exclude allocation, copies, and CPU validation, but include gaps
between submissions. cuBLAS handle initialization precedes timing.

The temporary sweep harness reused `tests/gemm.cu` with a cache-friendly CPU
reference loop: preconvert half to double, iterate row/K/column. Each output
still accumulates in increasing K order. GPU calls and timing are unchanged.
The selected default was also tested with the unmodified CPU-reference loop in
the repository test: 119.214 us at 1024 cubed, consistent with the sweep.

### Configuration sweep at M=N=K=1024

BM/BN describe a block's output, BK its K chunk, WM/WN a warp's output.
Both no-padding and 16-half-padding variants passed the then-current 14-case
correctness suite and the 1024-cubed random case.

| Block output | Warp output | Warps/block | BK | No padding (us) | Padding 16 (us) |
|---|---|---:|---:|---:|---:|
| 32x32 | 16x16 | 4 | 16 | 195.676 | 198.236 |
| 64x64 | 32x32 | 4 | 16 | 146.790 | 131.707 |
| 64x64 | 32x32 | 4 | 32 | 150.948 | 118.600 |
| 64x64 | 32x32 | 4 | 64 | 173.414 | 121.508 |
| 64x128 | 32x32 | 8 | 32 | 164.598 | 119.163 |
| 128x128 | 32x64 | 8 | 32 | 172.001 | 128.922 |

The 64x64/BK32/four-warp configuration is the default. Its timing is close to
64x128, while using a smaller block and fewer warps. Larger blocks are not
uniformly faster: 128x128 creates only 64 blocks at 1024 cubed, fewer than the
A800's 108 SMs. Padding improved several configurations; hardware counters were
unavailable, so the bank-conflict reduction itself was not measured. Padding
also changes resource allocation and compiler-generated code.

### Selected default versus original WMMA and cuBLAS

All entries passed CPU-reference correctness checks.

| M x N x K | Original WMMA (us) | Multi-warp WMMA (us) | cuBLAS (us) | Speedup over original |
|---|---:|---:|---:|---:|
| 128 x 128 x 128 | 17.971 | 15.268 | 7.127 | 1.18x |
| 256 x 256 x 256 | 33.178 | 26.675 | 7.270 | 1.24x |
| 512 x 512 x 512 | 65.137 | 49.091 | 7.332 | 1.33x |
| 1024 x 1024 x 1024 | 266.414 | 118.610 | 15.309 | 2.25x |
| 2048 x 2048 x 2048 | 1807.165 | 698.440 | 74.035 | 2.59x |
| 256 x 2048 x 512 | 71.567 | 53.463 | 8.468 | 1.34x |
| 2048 x 256 x 512 | 71.649 | 56.003 | 10.107 | 1.28x |
| 1009 x 513 x 257 | 40.428 | 33.106 | 19.364 | 1.22x |
| 1 x 4096 x 4096 | 867.727 | 453.018 | 21.688 | 1.92x |

The default is a learning baseline, not a per-shape dispatcher. It still trails
cuBLAS and wastes output work on thin matrices. No universal optimum or T4
runtime speedup is claimed.

### Resource tradeoff

CUDA function attributes and the occupancy API report 96 registers/thread,
15 KiB static shared memory/block, zero local memory, and at most five resident
128-thread blocks/SM: theoretical occupancy 31.25%. Original single-warp WMMA
had a theoretical ceiling of 50%. Higher arithmetic/data reuse improved time
despite a lower occupancy ceiling. These are theoretical limits, not achieved
occupancy or a measured stall breakdown. Nsight Compute counters remain blocked
by ERR_NVGPUCTRPERM in this environment.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=3 make check-gemm
CUDA_VISIBLE_DEVICES=3 make run-gemm-compare GEMM_ARGS="1024 1024 1024 100"

# A separate binary for one configuration; no changes to the default are needed.
/usr/local/cuda/bin/nvcc -O3 -lineinfo -std=c++14 -arch=sm_80 \
  -DGEMM_BM=64 -DGEMM_BN=64 -DGEMM_BK=32 \
  -DGEMM_WM=32 -DGEMM_WN=32 -DGEMM_SKEW=16 \
  tests/gemm.cu src/gemm_wmma_tiled.cu -o /tmp/gemm_wmma_config
CUDA_VISIBLE_DEVICES=3 /tmp/gemm_wmma_config --check-only
CUDA_VISIBLE_DEVICES=3 /tmp/gemm_wmma_config 1024 1024 1024 100
```

BM must be divisible by WM, BN by WN; WM, WN and BK must be positive multiples
of 16. SKEW is a nonnegative multiple of 16 half elements. The warp count is
(BM/WM)*(BN/WN). The complete shared-memory allocation must fit the target GPU.
Test a configuration before timing it. The original single-warp WMMA and cuBLAS
remain independent source files and comparison targets.

The new 65x129x67 alpha/beta case in `tests/gemm.cu` exercises block, warp, and
K-chunk boundaries together. The selected default passes all 15 cases on A800.
It compiles for sm_80 and sm_75; only A800 runtime was tested.
Compute Sanitizer memcheck, racecheck, and synccheck also passed the 65x129x67
case with zero reported errors or hazards. `make sanitize` includes the memory
and race checks for this variant.

For background on row padding, NVIDIA's
[cudaTensorCoreGemm sample](https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu)
uses a 16-half skew to alter shared-memory bank mapping while preserving WMMA
alignment. This experiment uses row-major A and B and measures its own layouts.
