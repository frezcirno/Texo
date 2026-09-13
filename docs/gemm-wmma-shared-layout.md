# Shared-layout and larger-tile experiment

The full/aligned path in `src/gemm_wmma_tiled_pipeline_aligned.cu` now uses
permuted shared storage and explicit Tensor Core instructions. On the measured
A800, ordinary 1024/2048 cubed benchmarks improve by 1.31x/1.34x over the initial
aligned specialization at `5eed6d5`. Larger block and warp tiles were tested;
the default remains BM=BN=64, BK=32, WM=WN=32, with four warps per block.

## Implementation and contract

The public contract remains contiguous row-major FP16 A/B/C, FP32 accumulation,
and `C = alpha * A * B + beta * C`. Empty output dimensions are a no-op; K=0
scales C by beta. The fast path requires positive M/N/K divisible by BM/BN/BK
and 16-byte-aligned A/B. Tails, unaligned A/B, and K=0 retain the generic WMMA
pipeline. C still needs only half alignment.

Each shared input tile stores a logical `(row, col)` at this half-element offset:

```cpp
(row * COLS + col) ^ ((row & 7) * 8)
```

COLS is BK for A and BN for B. It must be a power of two, at least 16; other
legal tile configurations retain the original full/aligned WMMA path. The XOR
permutes eight-half packs while preserving all values within a 16-byte copy.
It is bijective within the allocated tile: source bits are above the bits they
toggle, and tile dimensions are multiples of 16. For narrow rows, the permutation
can also change row bits; both copies and matrix loads use the same offset map.
Aligned 8x8 matrix rows then span all eight 16-byte shared-bank groups.

The implementation uses `ldmatrix.x4` for A and `ldmatrix.x4.trans` for B, followed
by `mma.sync.m16n8k16` on sm_80+. This follows the documented PTX register mapping;
it does not reinterpret the opaque C++ WMMA fragment representation. On sm_75,
two `mma.sync.m16n8k8` operations cover each K=16 fragment and input staging is
synchronous. The instruction layouts are specified in the
[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions).

The two-stage pipeline still prefetches the next K chunk while computing the
current one. Per-thread copy waits and a block barrier protect buffer reuse.
Copy pointers and permuted destinations are computed once per output tile; no
prefetch or pointer advance occurs after the final valid chunk.

The epilogue packs two adjacent FP16 results, uses warp shuffles to arrange
complete output rows, then stores `half2`. A two-byte-offset C pointer uses two
scalar stores instead. Alpha/beta are applied in FP32 before conversion; beta=0
does not read C. This avoids the original shared-memory output scratch. The
default kernel uses 16 KiB shared storage versus 26 KiB before, and 94 registers
per thread versus 92, with no local allocation or stack frame in the sm_80 build.

The retained WMMA path also accepts `GEMM_SKEW=8`: half strides need multiples of
eight elements, while its fragment origins remain 32-byte aligned. This follows
the [CUDA 12.6 WMMA alignment requirements](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#warp-matrix-functions).
The default SKEW remains 16 and does not affect the XOR path.

## Ordinary benchmark results

Collected on 2026-09-13, NVIDIA A800 80GB PCIe (108 SMs), CUDA 12.6.20,
driver 535.247.01, with `-O3 -std=c++14 -arch=sm_80 -Xcompiler -Wall -lineinfo`.
The before/after/cuBLAS runs were serial on physical GPU 4, with no competing
compute work on that GPU observed. Other GPUs on the machine were in use.
Clocks were not locked and system configuration was not changed.

These are `tests/gemm.cu` random-data benchmarks with aligned buffers, alpha=1,
beta=0, five warmup calls, and the median of five batches of 100 calls. CUDA-event
times exclude allocation, transfers, and CPU validation, but include any host
submission gaps. Each run passed the CPU reference; maximum absolute error on
the large cases was 0.03125 within `atol=0.01, rtol=0.01`.

| M x N x K | Before (us) | XOR layout (us) | Speedup | cuBLAS (us) |
|---|---:|---:|---:|---:|
| 128 x 128 x 128 | 5.356 | 5.396 | 0.99x | 7.086 |
| 512 x 512 x 512 | 10.435 | 10.588 | 0.99x | 7.301 |
| 1024 x 1024 x 1024 | 29.184 | 22.282 | 1.31x | 15.309 |
| 2048 x 2048 x 2048 | 167.711 | 125.256 | 1.34x | 74.598 |
| 1009 x 513 x 257 | 27.791 | 27.904 | 1.00x | 19.272 |

The tail case uses the generic path. Small differences near one percent should
not be treated as a stable gain or loss on this machine. The new default still
takes 1.46x/1.68x cuBLAS time at 1024/2048 cubed in these ordinary benchmarks.

## NCU before/after

Fresh captures were serial on physical GPU 5 with Nsight Compute 2024.3.0:
`--set full --replay-mode kernel --cache-control all --clock-control none
--launch-skip 7 --launch-count 1 --import-source yes`. Each benchmark used two
repeats; the selected launch follows two correctness calls and five warmups.
The six reports required 47–49 replay passes. Profiling used temporary
administrator access without changing driver settings. cuBLAS source was
unavailable. The comparison below uses aggregate counters, not PM/PC sampling
to infer a detailed execution timeline.

NCU durations measure individual kernels with replay/cache flushing; they are
separate from the warmed ordinary benchmark above. The captured 2048 default
kernel's SASS instructions and encodings were checked against the final build
and are identical, following removal of an unsuccessful automatic wider-tile
dispatch experiment.

| Metric | 1024 before | 1024 XOR | 1024 cuBLAS | 2048 before | 2048 XOR | 2048 cuBLAS |
|---|---:|---:|---:|---:|---:|---:|
| Kernel duration (us) | 30.400 | 29.536 | 17.408 | 164.320 | 118.240 | 73.248 |
| Tensor active (% active cycles) | 25.55 | 25.80 | 58.85 | 36.58 | 49.27 | 85.16 |
| Achieved occupancy (%) | 14.90 | 14.80 | 6.14 | 27.41 | 29.06 | 10.68 |
| Shared-load bank conflicts | 1,048,576 | 0 | 0 | 8,434,096 | 265 | 18,394 |
| Shared-store bank conflicts | 33,352 | 0 | 0 | 166,952 | 0 | 6,561 |
| LDSM actual / ideal wavefronts | 2.00 | 1.00 | 1.00 | 2.00 | 1.00 | 1.00 |
| Async-copy actual / ideal wavefronts | 2.125 | 1.500 | 1.380 | 2.125 | 1.500 | 1.522 |
| L2 read traffic (MiB) | 64 | 64 | 38 | 512 | 512 | 232 |
| Executed warp instructions (millions) | 3.327 | 3.699 | 1.701 | 23.400 | 27.378 | 10.044 |

The 2048 gain is 1.39x. Shared matrix loads now need the ideal number of
wavefronts. Short-scoreboard stalls fall from 3.38 to 0.74 cycles per issued
instruction; MIO throttle falls from 3.78 to 0.20 and barrier stalls from 2.44
to 0.51. These counters support reduced shared-memory pressure. Wavefronts
attributed to async-copy source lines and hardware bank-conflict totals describe
different work; zero conflicts do not imply that async-copy wavefronts are ideal.
See the [NCU metrics reference](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference).

There are remaining costs. Default L2 traffic is unchanged, explicit mapping and
the shuffled epilogue increase instruction count, and long-scoreboard stalls
remain high (5.23 cycles per issued instruction at 2048). cuBLAS uses less L2
traffic and sustains more Tensor work despite lower occupancy. At 1024, NCU shows
only a 1.03x improvement, unlike the 1.31x warmed benchmark; cache state and
measurement scope materially affect the result. The remaining NCU time ratios
to cuBLAS are 1.70x at 1024 and 1.61x at 2048.

## Larger block and warp tiles

The final four configurations below all use BK=32 and four warps per block.
Each independently passed the maintained 31-case CPU-reference suite. Screening
times are from a separate all-ones harness on GPU 5: five warmups, median of five
batches of 100 calls, CUDA events, checking every output against K and checking
output guards. They compare tile choices, and should not be mixed with the
random-data results from GPU 4 above.

| Block BM x BN | Warp WM x WN | 128 cubed (us) | 512 cubed (us) | 1024 cubed (us) | 2048 cubed (us) |
|---|---|---:|---:|---:|---:|
| 64 x 64 | 32 x 32 | 5.396 | 10.650 | 22.252 | 117.903 |
| 128 x 64 | 64 x 32 | 6.697 | 13.158 | 25.027 | 183.357 |
| 64 x 128 | 32 x 64 | 6.728 | 12.954 | 25.549 | 123.884 |
| 128 x 128 | 64 x 64 | 9.636 | 18.012 | 29.655 | 139.766 |

Larger warp tiles increase accumulator and operand register use, and larger
blocks reduce the number of output blocks. Reuse alone did not yield a faster
final configuration in this sweep. The 64x128 option uses 166 registers/thread
and 24 KiB shared storage; it remains explicitly selectable but is not enabled
automatically. This limited sweep does not establish the best tile for every
matrix shape or architecture.

Register budgeting mattered: the default `__launch_bounds__` target of five
blocks reduced an earlier 120-register build to 94 without a stack frame.
Targeting six introduced a 32-byte stack frame and was slower. A preliminary
automatic wide path reached 188 registers and regressed at 2048; it was removed.
The final wide option targets three blocks. These are compiler register budgets,
not guarantees of achieved residency or occupancy.

Simply changing WMMA padding from 16 to eight half elements was also insufficient:
an initial 1024 NCU probe removed LDSM conflicts but increased async-copy
wavefronts from 1,114,112 to 1,671,168 and duration from 30.752 to 33.248 us.
The final implementation changes both storage layout and matrix-load mapping.

## Reproduce and validate

From the repository root, select an available GPU and run the default version:

```bash
CUDA_VISIBLE_DEVICES=4 make run-gemm-wmma-tiled-pipeline-aligned GEMM_ARGS="2048 2048 2048 100"
CUDA_VISIBLE_DEVICES=4 make check
CUDA_VISIBLE_DEVICES=4 make sanitize
```

Build the wider option in a separate directory so compiler flags cannot reuse a
stale binary:

```bash
make BIN_DIR=build/sm_80/wide \
  NVCCFLAGS='-O3 -std=c++14 -arch=sm_80 -Xcompiler -Wall -lineinfo -DGEMM_BN=128 -DGEMM_WN=64' \
  build/sm_80/wide/gemm_wmma_tiled_pipeline_aligned_bench
CUDA_VISIBLE_DEVICES=4 build/sm_80/wide/gemm_wmma_tiled_pipeline_aligned_bench --check-only
CUDA_VISIBLE_DEVICES=4 build/sm_80/wide/gemm_wmma_tiled_pipeline_aligned_bench 2048 2048 2048 100
```

Other knobs remain `GEMM_BM/BN/BK/WM/WN`; set `GEMM_USE_SWIZZLE=0` to compare the
retained full/aligned WMMA path. `GEMM_SWIZZLE_MIN_BLOCKS=0` chooses the register
budget above; positive values override it for experiments. Inspect register,
stack, and shared usage after changing tiles or compiler versions.

Validation completed on A800:

- `make -j2 check` and `make sanitize` passed. Both pipelined variants passed all
  31 cases under memcheck, racecheck, and synccheck, with zero errors or hazards.
- All four final block/warp configurations passed the 31-case suite. The 64x128
  build additionally passed all three sanitizer tools over the complete suite.
- Three added cases use 1024x2048 output, K=32/96, random inputs, nontrivial
  alpha/beta, unaligned C, and unaligned A. Existing cases retain independent
  M/N/K tails, K=0, repeated calls, guards, and one/two/three K chunks.
- sm_75 compiled. Compiling with `-gencode arch=compute_75,code=sm_80` and running
  all 31 cases on A800 exercised the synchronous-copy/two-K=8-MMA source branch.
  This is not a T4 hardware runtime test. A BK=48 build also passed all 31 cases,
  exercising fallback for a non-power-of-two shared width.

Local artifacts are under the ignored `build/sm_80/aligned-layout-v2/` directory:
`normal-timings-gpu4.csv`, `screen.csv`, `configs.json`, screening sources/scripts,
validation logs, and `ncu-final/{1024,2048}/` reports/text/raw CSV exports.
`ncu-final/layout_comparison.csv` includes the additional source-wavefront metrics.
Intermediate and rejected experiments are kept separately from these final tables.

The next useful experiments are reducing repeated L2 loads without excessive
register growth, and adding another input/operand pipeline stage to cover the
remaining load latency. More shared padding or higher occupancy alone is not
supported as the next fix by these measurements.
