# Multistage input buffering and larger reuse tiles

`src/gemm_wmma_tiled_pipeline_multistage.cu` is an independent successor to
`gemm_wmma_tiled_pipeline_aligned_swizzled.cu` at commit `f3e0ea9`. It reduces
repeated input loads with larger output tiles and gives copies more time to
complete with three input stages. The previous implementations remain separate
benchmarks. Both the previously split swizzled version and this version are now
included in the Makefile's nine-GEMM comparison, correctness, NSYS, and NCU targets.

## Pipeline

The input ring holds STAGES shared-memory tiles. The prologue issues STAGES-1
copy groups. Waiting with `__pipeline_wait_prior(STAGES-2)` completes the first
group while later groups can remain outstanding. Each iteration then:

1. Issues the chunk at `t + STAGES - 1` into the free ring slot, if it exists.
2. Computes chunk t, preloading the next K=16 operand fragments into a second
   register buffer before issuing MMA on the current fragments.
3. Waits for the next input chunk and synchronizes the block before ring reuse.

An empty copy group is still committed when the input is exhausted. This keeps
the wait distance correct for short prologues and the final drain, without any
out-of-bounds source pointer or copy. All warps release the old stage at the block
barrier before it can be overwritten. A final wait drains the remaining groups.
The wait-distance semantics are specified in the
[CUDA 12.6 pipeline primitives interface](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#pipeline-primitives-interface).

With three stages, computation on chunk t overlaps loading t+1 and t+2. The
previous two-buffer version prefetches only t+1. The XOR shared layout, explicit
`ldmatrix`/`mma.sync`, FP32 accumulation, and shuffled `half2` epilogue are retained.
An unaligned C pointer uses scalar stores. sm_75 uses synchronous copies and two
K=8 MMA instructions per K=16 fragment; this does not provide Ampere async overlap.

## Dispatch and tuning

The contract remains contiguous row-major FP16 A[M,K], B[K,N], C[M,N], with FP32
accumulation and `C = alpha * A * B + beta * C`. Dimensions are nonnegative;
M/N=0 is a no-op, K=0 scales C by beta, and beta=0 does not read C.

Default full/aligned inputs require M/N/K divisible by 64/64/32 and A/B aligned
to 16 bytes. C only needs half alignment. The dispatcher selects the first
matching configuration below; all have BK=32, three input stages, and four warps.

| Condition after the full/aligned check | Block BM x BN | Warp WM x WN | Registers/thread | Shared/block |
|---|---|---|---:|---:|
| M/N divisible by 128, K >= 2048, at least 1024 blocks of 128x128 | 128 x 128 | 64 x 64 | 236 | 48 KiB |
| N divisible by 128, K >= 512, at least 256 blocks of 64x128 | 64 x 128 | 32 x 64 | 166 | 36 KiB |
| Remaining complete/aligned inputs | 64 x 64 | 32 x 32 | 96 | 24 KiB |

All three sm_80 kernels have zero stack/local allocation. Compiler register
budgets target two, three, and five blocks respectively; these are not guarantees
of achieved occupancy. The thresholds were measured on A800 and are simple
heuristics, not runtime autotuning or claims of optimality on every GPU or shape.

Increasing the block from 64x64 to 64x128 reduces logical input bytes per output
element by 25%; 128x128 reduces them by 50%. Increasing the warp tile supplies that
reuse with four warps, but raises register use. Small grids and short K cannot
always amortize those costs, which is why the largest tile is selected narrowly.

Tails, unaligned A/B, K=0, and unsupported fast-path widths use a fixed 64x64
generic WMMA pipeline with 26 KiB shared storage. Large fast-path tile overrides
do not inflate the generic path's shared-memory requirement.

Compile-time controls:

- `GEMM_BM/BN/BK/WM/WN/STAGES`: explicitly selecting any disables automatic tile
  selection by default. `GEMM_AUTO_TILE=0` also forces one configuration.
- Tile dimensions must contain whole 16x16 warp fragments; BM/BN must contain
  whole warp tiles. Fast-path BN/BK must be powers of two, at least 16.
- STAGES can be 2–4, with `STAGES * (BM+BN) * BK * sizeof(half) <= 48 KiB`.
  Larger static shared allocations are deliberately rejected at compile time.
- `GEMM_REGISTER_PIPELINE=0` disables register operand prefetch for comparison.
- `GEMM_MULTISTAGE_MIN_BLOCKS=0` chooses the budgets above; a positive value
  overrides the compiler budget.
- `GEMM_GROUP_M` optionally groups output rows in the block traversal. Default 1
  retains row-major traversal; groups 4/8 did not provide a consistent gain.

## Measurement setup

Measured on 2026-09-13, NVIDIA A800 80GB PCIe (108 SMs), CUDA 12.6.20,
driver 535.247.01, with `-O3 -std=c++14 -arch=sm_80 -Xcompiler -Wall -lineinfo`.
No competing compute process was observed on the selected GPUs. Other GPUs on
the machine were in use. Clocks were not locked; system/driver settings were not
changed. Results are specific to these shapes and this setup.

Tile screening ran serially on physical GPU 5, using all-ones inputs and checking
every output and guard. Every candidate separately passed the CPU-reference suite
with random inputs and nontrivial alpha/beta. Final timing used physical GPU 4
and random FP16 A/B with alpha=1, beta=0, checking every output against cuBLAS with
FP32 reductions and checking guards. The maintained CPU-reference suite is
separate from this large-matrix GPU-reference timing harness.

Each timing warms up five calls and takes the median of five CUDA-event batches
of 100 calls. Allocation, transfers, reference generation, and validation are
excluded; host submission gaps are included. The table reports the first of two
rounds; the second reverses variant order. Both rounds are saved in the raw CSV.

| M x N x K | Previous swizzled (us) | Multistage (us) | Speedup | cuBLAS (us) |
|---|---:|---:|---:|---:|
| 128 x 128 x 128 | 5.294 | 4.977 | 1.06x | 7.240 |
| 512 x 512 x 512 | 10.711 | 8.294 | 1.29x | 7.240 |
| 1024 x 1024 x 1024 | 22.262 | 22.016 | 1.01x | 15.421 |
| 1536 x 1536 x 1536 | 75.919 | 46.264 | 1.64x | 34.488 |
| 2048 x 2048 x 2048 | 123.464 | 96.973 | 1.27x | 74.793 |
| 3072 x 3072 x 3072 | 500.582 | 355.113 | 1.41x | 292.424 |
| 4096 x 4096 x 4096 | 1103.944 | 765.686 | 1.44x | 636.682 |
| 1024 x 4096 x 1024 | 62.331 | 56.064 | 1.11x | 47.206 |
| 2048 x 4096 x 1024 | 124.703 | 105.851 | 1.18x | 79.186 |
| 2048 x 2048 x 128 | 15.452 | 16.404 | 0.94x | 15.309 |
| 4096 x 4096 x 512 | 148.838 | 117.842 | 1.26x | 84.122 |
| 1009 x 513 x 257 | 27.822 | 27.822 | 1.00x | 19.374 |

The second round measured 97.106 us at 2048 cubed and 768.717 us at 4096 cubed.
Those cases now take about 1.30x and 1.20x cuBLAS time, respectively. At 1024 cubed
the improvement is negligible; the short-K 2048x2048x128 case regresses around
6%. A deeper pipeline and more reuse do not benefit every shape. Full/aligned
outputs matched cuBLAS exactly in these runs; the tail case had maximum absolute
error 0.0078125, within `atol=0.01, rtol=0.01`.

## Tile experiments

Selected all-ones screening results on GPU 5, separate from the random-data
timings above. BK=32 throughout; `R` means register operand prefetch enabled.

| Block / warp | Input stages | Registers | 1024 cubed (us) | 2048 cubed (us) | 4096 cubed (us) |
|---|---:|---:|---:|---:|---:|
| Previous 64x64 / 32x32 | 2 | 94 | 22.098 | 117.852 | 1009.930 |
| 64x64 / 32x32, R | 3 | 96 | 21.852 | 113.050 | 1088.123 |
| 64x64 / 32x32, R | 4 | 96 | 22.395 | 113.797 | 1028.710 |
| 128x64 / 32x32, R | 3 | 79 | 26.552 | 106.762 | 1024.573 |
| 64x128 / 32x32, R | 3 | 80 | 27.474 | 108.452 | 963.369 |
| 128x128 / 32x64, R | 2 | 128 | 26.757 | 124.979 | 786.258 |
| 128x128 / 32x64, R | 3 | 128 | 27.136 | 119.552 | 808.407 |
| 64x128 / 32x64, R | 3 | 166 | 23.532 | 92.292 | 756.890 |
| 128x128 / 64x64, R | 3 | 236 | 28.068 | 134.062 | 644.495 |

The wider four-warp configurations were preferable to the eight-warp variants
on the larger measured problems. Additional fourth-stage storage did not improve
the final choices. Grouped output traversal was also tested and left disabled.
The final three kernel SASS encodings match the screened specializations,
including register budgets; integrating dispatch did not change those kernels.

## NCU: repeated reads and compute waits

Nine reports were captured serially on GPU 5 with Nsight Compute 2024.3.0,
`--set full --replay-mode kernel --cache-control all --clock-control none
--import-source yes --profile-from-start off --launch-count 1`. The random-data
harness checks against cuBLAS, warms five calls, then brackets one selected call
with `cudaProfilerStart/Stop`. This excludes the reference kernel from collection.
Reports required 48–49 replay passes. Temporary administrator execution enabled
counters without changing system settings. cuBLAS implementation source was
unavailable. Aggregate counters below do not rely on the sampling timelines.

These are cache-flushed replay kernel durations, not the ordinary warmed event
times above. Comparisons are within each measurement scope.

| Shape / metric | Previous swizzled | Multistage | cuBLAS |
|---|---:|---:|---:|
| 1024 cubed: duration (us) | 29.472 | 22.816 | 16.896 |
| L2 reads (MiB) | 64 | 64 | 38 |
| Tensor active (% active cycles) | 25.89 | 39.48 | 60.70 |
| Long scoreboard (cycles/issued instruction) | 5.91 | 0.71 | 0.40 |
| 2048 cubed: duration (us) | 118.432 | 95.232 | 73.376 |
| L2 reads (MiB) | 512 | 384 | 232 |
| Tensor active (% active cycles) | 49.16 | 62.86 | 84.93 |
| Long scoreboard (cycles/issued instruction) | 5.18 | 0.72 | 0.09 |
| Executed warp instructions (millions) | 27.38 | 18.79 | 10.04 |
| Achieved occupancy (%) | 29.01 | 15.75 | 10.62 |
| 4096 cubed: duration (us) | 984.992 | 674.016 | 551.488 |
| L2 reads (MiB) | 4096 | 2048 | 1536 |
| Tensor active (% active cycles) | 49.56 | 73.88 | 93.12 |
| Long scoreboard (cycles/issued instruction) | 6.40 | 0.34 | 0.06 |
| Executed warp instructions (millions) | 210.17 | 113.57 | 61.41 |
| Achieved occupancy (%) | 30.50 | 12.29 | 12.49 |

L2 reads fall by exactly 25% at 2048 and 50% at 4096, matching the larger tile's
logical reuse. Long-scoreboard stalls fall substantially at all three sizes,
consistent with the input pipeline providing more lead time. At 1024, the
unchanged tile isolates the deeper input/operand schedule: it improves the
cache-flushed NCU duration by 1.29x, while ordinary warmed timing is nearly
unchanged. Those observations must not be combined into one claimed speedup.

At 4096, actual DRAM read bytes also fall from 305.68 MB to 204.52 MB (cuBLAS:
167.61 MB). L2 read traffic counts hits as well as misses and is much larger than
these DRAM reads. The new kernel's DRAM throughput is only 18.0% of peak in this
capture, so the remaining gap is not evidence of saturated device DRAM bandwidth.

Shared matrix-load wavefronts equal their ideal counts for all three custom
configurations. Hardware shared-load conflicts are 0, 15,913, and 927 for the
new kernels; shared-store conflicts are zero. Larger tiles do not make every
stall smaller: at 2048, barrier stalls increase from 0.52 to 0.69 cycles per
issued instruction. Higher register use also lowers occupancy. Nevertheless,
more reuse and less input waiting increase Tensor utilization and reduce time.
Metric definitions are in the [NCU profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference).

The remaining 4096 NCU gap is about 1.22x: the custom kernel reads 33% more from
L2, executes about 1.85x as many warp instructions, and sustains 73.9% Tensor
activity versus 93.1% for cuBLAS. Shared-operand dependencies, epilogue/control
instructions, and reuse beyond the current tile remain relevant costs.

## Validation

- `make -j2 check` passed with all nine GEMM variants. The suite now has 36 cases,
  adding two/four/five/seven/eight K chunks to test short prologues, ring wrap,
  and drain, including nontrivial alpha/beta and unaligned C.
- `make sanitize` passed with zero errors or hazards. It runs memcheck,
  racecheck, and synccheck over the full suite for all four pipelined versions.
- Every screened configuration passed the 36-case CPU double-reference suite.
  The selected 64x128 and 128x128 kernels additionally passed all three
  sanitizer tools in forced-tile builds. Their SASS matches the final kernels.
- All three automatic dispatch paths were exercised by the random-data timing
  and NCU runs, with every output and output guard checked against cuBLAS.
- sm_75 compilation and independent sm_80 kernel-object compilation passed.
  A compute_75-to-sm_80 build also passed the 36-case suite on A800, exercising
  the synchronous-copy/two-K=8-MMA source branch. This is not a T4 runtime test.

## Run

```bash
CUDA_VISIBLE_DEVICES=4 make run-gemm-wmma-tiled-pipeline-multistage GEMM_ARGS="2048 2048 2048 100"
CUDA_VISIBLE_DEVICES=4 make check-gemm
CUDA_VISIBLE_DEVICES=4 make sanitize

# Force the 64x128 block / 32x64 warp configuration in a separate directory.
make BIN_DIR=build/sm_80/multistage-wide \
  NVCCFLAGS='-O3 -std=c++14 -arch=sm_80 -Xcompiler -Wall -lineinfo -DGEMM_BM=64 -DGEMM_BN=128 -DGEMM_WM=32 -DGEMM_WN=64 -DGEMM_STAGES=3' \
  build/sm_80/multistage-wide/gemm_wmma_tiled_pipeline_multistage_bench
CUDA_VISIBLE_DEVICES=4 build/sm_80/multistage-wide/gemm_wmma_tiled_pipeline_multistage_bench --check-only

# Existing NCU targets profile only the requested variants.
CUDA_VISIBLE_DEVICES=4 make ncu-gemm \
  NCU_GEMMS='gemm_wmma_tiled_pipeline_aligned_swizzled gemm_wmma_tiled_pipeline_multistage gemm_cublas' \
  GEMM_ARGS='2048 2048 2048 2' NCU_DIR=build/sm_80/ncu-multistage
```

NCU requires performance-counter access; use the existing `NCU_RUN` prefix for
temporary administrator execution where required. The targets do not change
driver permissions. Changing compile-time options requires a fresh `BIN_DIR`
or a forced rebuild.

Local experiment sources, configuration lists, logs, and raw measurements are
under the ignored `build/sm_80/multistage-v1/` directory. `normal-random.csv`
contains both final timing rounds. The three `screen-*.csv` files record distinct
sweeps; `sweep-source.cu` preserves the pre-dispatch source. Raw profiler reports
are under `ncu/{1024,2048,4096}/`.
