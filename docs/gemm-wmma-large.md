# Larger block tiles, dynamic shared memory, and warp tiling

`src/gemm_wmma_tiled_pipeline_large.cu` is an independent successor to
`gemm_wmma_tiled_pipeline_epilogue.cu`. It supports input stages above 48 KiB
using dynamic shared memory. The investigation tested the requested changes
in order: larger block tiles, deeper pipelines using dynamic storage, then
different warp partitions and register budgets.

The 26 screened configurations did not establish a broadly useful replacement
for the existing automatic dispatcher. Small improvements occurred at selected
shapes: the best repeated 8192-cubed result was 3.3% faster, and a short-K
2048x2048x128 case improved by 1.7%. Other shapes regressed substantially.
The default dispatcher therefore retains the epilogue version's choices;
the experimental configurations remain available through compile-time overrides.

## Implementation and contract

The FP16 row-major contract is unchanged: A[M,K], B[K,N], C[M,N], FP32
accumulation, and `C = alpha * A * B + beta * C`. Exact alpha=1/beta=0 still
uses the specialized epilogue; beta=0 ignores old C, including beta=-0.
M/N=0 does no work and K=0 scales C by beta. Native paths require complete
tiles and 16-byte-aligned A/B. C needs only half alignment; unaligned half2
outputs use scalar stores. Tails and unaligned A/B use the fixed generic
64x64 WMMA pipeline.

Native input storage occupies
`STAGES * (BM + BN) * BK * sizeof(half)` bytes. Configurations at or below
48 KiB retain the original static A/B arrays. Larger ones use one 32-byte-aligned
`extern __shared__` allocation, with all A stages followed by all B stages.
The unused static arrays are eliminated from those specializations.
The XOR layout, asynchronous copies, register operand pipeline, and epilogue
remain the same as in the preceding version.

The launcher queries the active device's
`cudaDevAttrMaxSharedMemoryPerBlockOptin`, calls `cudaFuncSetAttribute` with
`cudaFuncAttributeMaxDynamicSharedMemorySize`, and supplies the allocation size
as the third launch argument. These are kernel/runtime settings, with no
system or driver configuration changes. NVIDIA documents the dynamic-storage
and opt-in requirement for allocations above 48 KiB in its
[Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#unified-shared-memory-l1-texture-cache).

Resource checks and configuration run on each dynamic launch, so there is
no host cache that becomes stale after switching devices or resetting a CUDA
context. If the requested size exceeds the device's capacity, the forced
configuration uses the generic fallback. `GEMM_DYNAMIC_SHARED_LIMIT` optionally
imposes a lower cap in bytes; zero uses the device limit. CUDA API failures
remain visible through the runtime error checks used by the harness.

Explicit `GEMM_BM/BN/BK/WM/WN/STAGES` overrides disable automatic dispatch by
default. `GEMM_MULTISTAGE_MIN_BLOCKS` controls the compiler's register target;
it must be tuned along with the number of warps. The new source supports
two to four stages and allocations up to the sm_80 ceiling of 163 KiB,
subject to the active device's actual limit.

All twelve default kernel SASS instruction sequences match the frozen epilogue
baseline, including the general alpha/beta kernels and the generic fallback.
Normal and Makefile NCU builds match as well. Every screened configuration had
zero stack/local allocation in both epilogue modes. The forced 96 KiB test
kernel also has zero stack/local allocation and zero static shared allocation.

## Measurement setup

Measured on 2026-09-13 with NVIDIA A800 80GB PCIe, 108 SMs, CUDA 12.6.20,
driver 535.247.01, using `-O3 -std=c++14 -arch=sm_80 -Xcompiler -Wall -lineinfo`.
Physical GPU 4 ran ordinary measurements serially; no competing compute
process was observed there. Other GPUs were in use and clocks were not locked.

The baseline is a frozen copy of the preceding epilogue source. Inputs are
random FP16 values with seed 12345 and alpha=1/beta=0. Before timing, all
outputs and guards are checked against cuBLAS with FP32 reductions. Each run
warms five calls and reports the median of five CUDA-event batches of 100 calls.
Allocation, transfers, reference generation, and validation are excluded;
host submission gaps and dynamic-launch configuration costs are included.
Event synchronization completes each batch before reading its elapsed GPU time.

The initial screens use one run per shape/configuration. The confirmation
tables average two runs with reversed variant order. Clocks and workload
history introduce variation, so the initial screens and later confirmations
are reported separately. All screened random-input outputs matched the
cuBLAS reference exactly.

## Step 1: larger block tiles

Five configurations were compiled from the original epilogue source, retaining
static shared storage. All use BK=32:

| Block | Warp | Stages | Minimum-block target | Shared/block | Specialized registers |
|---|---|---:|---:|---:|---:|
| 128x128 | 64x64 | 3 | 2 | 48 KiB | 231 |
| 128x128 | 64x64 | 2 | 2 | 32 KiB | 232 |
| 128x256 | 64x64 | 2 | 1 | 48 KiB | 226 |
| 256x128 | 64x64 | 2 | 1 | 48 KiB | 229 |
| 192x128 | 48x64 | 2 | 1 | 40 KiB | 198 |

At 2048 cubed, the initial baseline took 96.614 us. The 128x128 three-stage
candidate took 115.395 us; the 128x256 and 256x128 two-stage candidates took
166.666 and 165.509 us. At 4096 cubed, the 128x128 three-stage candidate is
already the baseline's selected kernel and remained effectively unchanged.
The larger rectangular tiles were slower.

Logical FP16 A/B bytes for complete tiles are
`2 * M * N * K * (1/BM + 1/BN)`. At 2048 cubed, enlarging the block from
64x128 to 128x256 halves this from 384 to 192 MiB. It also reduces the grid
from 512 to 128 blocks. The latter is only slightly larger than the GPU's
108-SM count, and each larger block permits only one resident block per SM
with the measured register usage. Less traffic does not compensate for the
changed parallelism, scheduling, and pipeline costs on this shape.

## Step 2: dynamic shared storage and deeper pipelines

Six new configurations used BK=32:

- 128x128 / 64x64 warp, four stages, 64 KiB, minimum-block target 2.
- 128x256 / 64x64 warp, three/four stages, 72/96 KiB, target 1.
- 256x128 / 64x64 warp, three/four stages, 72/96 KiB, target 1.
- 192x128 / 48x64 warp, three stages, 60 KiB, target 1.

A separate default-tile control verified that changing the storage abstraction
did not change any of the twelve existing kernel instruction sequences.

In the initial 4096-cubed screen, 128x256 improved from 902.533 us with two
stages to 830.669 us with three. Four stages took 852.163 us. The baseline
in the second screen was 765.614 us. The deeper pipeline recovers part of
the larger tile's regression but does not outperform that baseline.

## Step 3: warp shape and register budget

Fifteen configurations kept BK=32 and changed warp partitions:

- 128x128 blocks with four stages: 32x64 and 64x32 warps with minimum-block
  targets 1/2, 32x32 warps with target 1, and 32x128 or 128x32 warps with target 2.
- 128x256 blocks with three stages: 32x64, 64x32, or 32x128 warps, target 1.
- 256x128 blocks with three stages: 32x64, 64x32, or 128x32 warps, target 1.
- 192x128 blocks with three stages: 32x64 or 96x32 warps, target 1.

The larger warp shapes increase fragment reuse but keep substantial register
pressure. Smaller warp shapes reduce registers per thread while duplicating
more shared-memory operand loads across warps. None of the new warp partitions
beat the baseline on the screened 1024/1536/2048/2304/3072/4096-cubed problems.
The best confirmed short-K case is recorded below.

## Confirmed ordinary timings

The following comparison keeps a 128x256 block for the three successive
experimental rows, making the stage-count and warp-partition effects visible.
These are representative controlled comparisons, not the fastest candidate
at every shape. Values are microseconds, averaged over two reversed-order runs.

| Configuration | 2048 cubed | 4096 cubed | 2048x2048x128 |
|---|---:|---:|---:|
| Epilogue baseline | 101.401 | 777.114 | 15.022 |
| Step 1: 128x256 / 64x64 warp / 2 stages | 166.615 | 914.412 | 19.645 |
| Step 2: 128x256 / 64x64 warp / 3 stages | 153.989 | 838.523 | 19.317 |
| Step 3: 128x256 / 32x64 warp / 3 stages | 154.839 | 865.301 | 19.297 |
| 128x128 / 64x64 warp / 4 stages | 115.619 | 778.562 | 15.682 |
| 128x128 / 32x64 warp / 4 stages / target 2 | 120.494 | 884.695 | 14.766 |
| 128x128 / 128x32 warp / 4 stages / target 2 | 119.716 | 812.012 | 15.508 |
| cuBLAS | 74.972 | 650.967 | 15.345 |

The 14.766-us short-K result is a 1.7% reduction from the baseline, but the
same configuration regresses by 18.8% at 2048 cubed and 13.8% at 4096 cubed.
The screens also covered 2048x4096x1024, 4096x2048x1024, and 384x6144x96,
without a clear improvement over the existing dispatcher.

Larger-matrix checks found a small opportunity at 8192 cubed:

| Configuration | 6144 cubed, initial screen (us) | 8192 cubed, two-run confirmation (us) |
|---|---:|---:|
| Epilogue baseline | 2590.290 | 6859.915 |
| 128x128 / 64x64 warp / 4 stages | 2600.356 | 6789.407 |
| 128x256 / 64x64 warp / 3 stages | 2771.906 | 6665.333 |
| 256x128 / 64x64 warp / 3 stages | 2757.714 | 6635.162 |
| cuBLAS | 2088.827 | 4989.767 |

The best 8192-cubed candidate reduces average time by 3.3%, but remains 33.0%
slower than cuBLAS. The baseline varied from 6759.957 to 6959.872 us; the
candidate varied from 6620.743 to 6649.580 us. Its initial-screen improvement
was only 1.1%. This supports retaining the candidate for explicit experiments,
but does not establish a reliable automatic threshold across other large shapes
or K values. The same configuration regressed at 6144 cubed.

A separate opt-in overhead check compared per-launch configuration with a
temporary per-device cache on the 128x128 four-stage kernel. At 2048 cubed,
two-run means were 115.451 versus 114.744 us; at 4096 cubed, 769.337 versus
773.714 us. The short-K result differed by 0.02 us. Thus the repeated attribute
calls do not explain the much larger kernel regressions. The final source
keeps context-safe per-launch configuration.

## NCU: fewer global loads, more shared-memory and synchronization costs

Nsight Compute 2024.3.0 captured ten reports on physical GPU 5, with
`--set full --replay-mode kernel --cache-control all --clock-control none
--import-source yes --profile-from-start off --launch-count 1`.
The random-input harness validates results and warms five calls before bracketing
one solve with `cudaProfilerStart/Stop`. Temporary administrator execution
enabled counters without changing system settings. These cache-flushed replay
durations are separate from ordinary warmed timings.

The experiment columns below all use a 128x256 block. Step 1 has 64x64 warps
and two stages; step 2 changes to three stages; step 3 changes to 32x64 warps.

| Shape / metric | Baseline | Step 1 | Step 2 | Step 3 | cuBLAS |
|---|---:|---:|---:|---:|---:|
| 2048 cubed: duration (us) | 95.264 | 171.200 | 153.792 | 155.040 | 73.088 |
| L2 reads (MiB) | 384 | 192 | 192 | 192 | 232 |
| Tensor active (% active cycles) | 63.06 | 54.20 | 61.05 | 60.89 | 85.40 |
| Achieved occupancy (%) | 16.02 | 13.00 | 12.53 | 24.88 | 10.66 |
| Executed warp instructions (millions) | 19.339 | 11.906 | 12.106 | 18.260 | 10.044 |
| 4096 cubed: duration (us) | 664.992 | 853.760 | 748.832 | 760.032 | 551.072 |
| L2 reads (MiB) | 2048 | 1536 | 1536 | 1536 | 1536 |
| Tensor active (% active cycles) | 74.42 | 55.06 | 63.83 | 63.49 | 93.17 |
| Achieved occupancy (%) | 12.04 | 12.47 | 12.49 | 24.95 | 12.50 |
| Executed warp instructions (millions) | 104.337 | 92.189 | 93.774 | 142.246 | 61.411 |
| Matrix-load shared wavefronts (millions) | 33.554 | 33.554 | 33.554 | 50.332 | 33.686 |
| Short scoreboard (cycles/issued instruction) | 0.37 | 0.63 | 0.40 | 1.37 | 0.03 |
| MIO throttle (cycles/issued instruction) | 0.05 | 0.61 | 0.63 | 0.55 | 0.03 |
| Barrier (cycles/issued instruction) | 0.87 | 0.91 | 1.09 | 2.35 | 0.05 |

The observed L2 reductions agree with the complete-tile loading formula.
At 4096 cubed, the large tile even matches cuBLAS's L2 read volume, while
Tensor activity and elapsed time remain substantially worse. DRAM reads stay
around 204.5 MB for all custom variants, versus 167.5 MB for cuBLAS; fewer
L2 requests did not remove the same amount of HBM traffic.

Adding the third stage reduces the large tile's long-scoreboard metric from
2.18 to 0.48 cycles per issued instruction at 4096 cubed. The smaller warp
then increases occupancy, but matrix-load shared wavefronts rise by 50%,
executed instructions rise by about 52%, and short-scoreboard/barrier metrics
increase. Higher occupancy alone therefore does not make this variant faster.
Matrix-load wavefronts equal their ideal counts and the three experimental
captures record zero hardware bank conflicts. Async copies still have excess
wavefronts, so this does not imply that every shared-memory cost is eliminated.

These results shift the next investigation toward issuing copies and matrix
loads more evenly through the MMA mainloop, and reducing unnecessary full-block
waiting. Further tile enlargement alone is not supported by these results.

## Validation and reproduction

All 26 screened configurations passed the existing 48-case CPU double-reference
suite. The shared suite now has 52 cases: four additional 256x256 cases cover
one/four/five/seven K chunks, both epilogues, NaN old C, and unaligned C for
the larger block shapes. `make check` passed all thirteen default GEMMs and
the forced dynamic test, with 52 cases each, plus the repository's other tests.

The forced `gemm_wmma_tiled_pipeline_large_dynamic_test` uses a 128x256 block,
32x64 warps, four stages, and 96 KiB of dynamic shared memory. This ensures
the new memory path is tested even though the default dispatcher uses static
tiles. `make sanitize` runs memcheck, racecheck, and synccheck for this forced
configuration as well as the default large version. The complete target
produced 40 zero-error summaries.

A separate forced-dynamic initcheck run left C uninitialized and verified
all outputs/guards for seven shapes; it reported zero errors. A forced build
with `GEMM_DYNAMIC_SHARED_LIMIT=49152` passed all 52 cases through the capacity
fallback. The 96 KiB source compiled for sm_75; a `compute_75` front-end /
`sm_80` back-end build passed all 52 cases on A800, exercising synchronous
copies and K=8 MMA with dynamic storage. This is not a T4 runtime test.

```bash
# Default dispatcher, with all previously selected tiles retained.
CUDA_VISIBLE_DEVICES=4 make run-gemm-wmma-tiled-pipeline-large \
  GEMM_ARGS="4096 4096 4096 100"

# Reproduce the 72 KiB candidate that helped at 8192 cubed.
# Use a separate BIN_DIR when changing compile-time parameters.
CUDA_VISIBLE_DEVICES=4 make run-gemm-wmma-tiled-pipeline-large \
  BIN_DIR=build/sm_80/large-256x128 \
  NVCCFLAGS="-O3 -std=c++14 -arch=sm_80 -Xcompiler -Wall -DGEMM_BM=256 -DGEMM_BN=128 -DGEMM_BK=32 -DGEMM_WM=64 -DGEMM_WN=64 -DGEMM_STAGES=3 -DGEMM_MULTISTAGE_MIN_BLOCKS=1" \
  GEMM_ARGS="8192 8192 8192 100"

CUDA_VISIBLE_DEVICES=6 make check
CUDA_VISIBLE_DEVICES=7 make sanitize
make ncu-gemm-build NCU_GEMMS=gemm_wmma_tiled_pipeline_large
```

The Makefile run target uses the shared CPU-reference benchmark; a large
8192-cubed CPU reference is slow. The recorded large-matrix timings instead
use the local GPU-reference harness described above. Frozen sources, parameter
JSONs, build/screen scripts, correctness logs, SASS hashes, and measurement CSVs
are in the ignored `build/sm_80/reuse123-v1/` directory. NCU reports and text/CSV
exports are under its `ncu/` directory. Generated artifacts are not committed.
