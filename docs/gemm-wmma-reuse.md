# Increasing A/B reuse with a 96x128 tile

`src/gemm_wmma_tiled_pipeline_reuse.cu` is an independent successor to
`gemm_wmma_tiled_pipeline_mainloop.cu`. It adds a 96x128 block / 48x64 warp
configuration for sufficiently populated complete/aligned grids. The selected
configuration reduces warmed execution time by 15.0% at 1536 cubed, 15.9% at
2304 cubed, and 8.1% at 3072 cubed in this experiment.

## Implementation and dispatch

The new tile has four warps, BK=32, and three input stages. It retains the XOR
shared layout, native `ldmatrix`/`mma.sync`, register operand buffering, and
packed output stores. No additional shared-layout transformation or pipeline
schedule is introduced.

Compared with the previous 64x128 block / 32x64 warp tile, each block and warp
computes 50% more output elements. For complete tiles, logical global input
bytes over the GEMM are `2 * M * N * K * (1/BM + 1/BN)`. Changing BM from 64
to 96 with BN=128 multiplies this by 7/9, a 22.2% reduction. The larger warp
tile similarly reuses its B operands across three M fragments instead of two.
This is the threadblock/warp reuse tradeoff described in NVIDIA's
[Efficient GEMM documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#threadblock-level-gemm).

The new kernel uses 237 registers/thread and 42 KiB shared/block. Its
`__launch_bounds__(128, 2)` target permits two resident blocks per SM on this
A800. The previous 64x128 kernel uses 166 registers and 36 KiB, permitting
three. Extra reuse must compensate for fewer resident warps and more work in
each block; small grids did not benefit.

The FP16 row-major contract is unchanged: A[M,K], B[K,N], C[M,N], FP32
accumulation, and `C = alpha * A * B + beta * C`. M/N=0 does no work; K=0
scales C by beta; beta=0 does not read C. Native paths require 16-byte-aligned
A/B, while C only needs half alignment. Tails and unaligned A/B use the fixed
generic WMMA pipeline.

Default native dispatch first requires complete 64x64 output tiles and K
divisible by 32. It selects the first matching row below; every configuration
has four warps and three input stages.

| Condition after the base alignment check | Block / warp | BK |
|---|---|---:|
| M/N divisible by 128, K >= 2048, at least 1024 blocks of 128x128 | 128x128 / 64x64 | 32 |
| M divisible by 192, N divisible by 128, at least 192 blocks of 96x128 | 96x128 / 48x64 | 32 |
| N divisible by 128, K >= 512, at least 256 blocks of 64x128 | 64x128 / 32x64 | 32 |
| K divisible by 64, K >= 192, at most 324 blocks of 64x64 | 64x64 / 32x32 | 64 |
| Remaining complete/aligned inputs | 64x64 / 32x32 | 32 |

M multiples of 192 satisfy both the base 64-row alignment and whole 96-row
tiles. The threshold of 192 new output blocks is conservative and measured
on this 108-SM A800. The existing 128x128 path takes precedence because it
offers greater reuse on large enough matrices. These rules are not runtime
autotuning or a claim of optimality for every shape or GPU.

All six compiled kernels, including the generic fallback, have zero stack/local
allocation. The new specialization's SASS matches the screened candidate;
the five existing specializations match the previous source. Normal,
line-info, and Makefile NCU builds also match. Explicit tile overrides still
disable automatic dispatch by default. A forced BM=96 build can accept M
divisible by 96 even when M is not divisible by 64.

## Ordinary measurements

Measured on 2026-09-13 with NVIDIA A800 80GB PCIe (108 SMs), CUDA 12.6.20,
driver 535.247.01, using `-O3 -std=c++14 -arch=sm_80 -Xcompiler -Wall`;
profile objects add `-lineinfo`. No system/driver configuration was changed
and clocks were not locked. Physical GPU 4 ran the final comparisons serially,
with no competing compute process observed there. Other GPUs were in use.

The baseline is the mainloop source frozen at the start of this experiment,
including its BK=64 path. Inputs are random FP16 values with seed 12345,
alpha=1, beta=0. Every output is checked against cuBLAS with FP32 reductions,
and output guards are checked. Each run warms five calls and takes the median
of five CUDA-event batches of 100 calls. Allocation, transfers, reference
generation, and validation are excluded; host submission gaps are included.
The table averages two runs, with reversed variant order in the second run.
CPU double-reference checks are separate from this large-matrix timing harness.

| M x N x K | Mainloop baseline (us) | Reuse version (us) | Time reduction | cuBLAS (us) |
|---|---:|---:|---:|---:|
| 1024 x 1024 x 1024 | 19.876 | 19.830 | 0.2% | 15.365 |
| 1152 x 1152 x 1152 | 24.919 | 24.843 | 0.3% | 18.124 |
| 1536 x 1536 x 1536 | 46.316 | 39.383 | 15.0% | 34.555 |
| 1920 x 1920 x 1920 | 90.537 | 90.311 | 0.2% | 65.909 |
| 2048 x 2048 x 2048 | 100.055 | 98.841 | 1.2% | 74.178 |
| 2304 x 2304 x 2304 | 139.100 | 117.007 | 15.9% | 106.644 |
| 3072 x 3072 x 3072 | 360.745 | 331.397 | 8.1% | 297.165 |
| 4096 x 4096 x 4096 | 765.686 | 765.732 | -0.01% | 639.856 |
| 4608 x 4608 x 4608 | 1056.912 | 1056.932 | -0.00% | 881.224 |
| 384 x 6144 x 32 | 8.617 | 7.915 | 8.1% | 8.714 |
| 384 x 6144 x 96 | 11.857 | 10.020 | 15.5% | 10.768 |
| 384 x 4096 x 512 | 16.773 | 16.799 | -0.2% | 13.296 |
| 1536 x 1536 x 128 | 12.109 | 10.398 | 14.1% | 10.552 |
| 1536 x 3072 x 512 | 36.203 | 32.901 | 9.1% | 28.820 |
| 3072 x 1536 x 512 | 36.409 | 33.040 | 9.3% | 29.291 |
| 1009 x 513 x 257 | 27.898 | 27.848 | 0.2% | 19.369 |

The 1536/2304/3072-cubed results now take 1.140x / 1.097x / 1.115x cuBLAS
time. At 1024 and 4096 cubed the selected kernels are unchanged; this work
does not close their remaining cuBLAS gaps. Even within the new path, 1920
cubed shows negligible improvement. The small differences on unchanged paths
are measurement variation, not evidence of a kernel improvement.

The timing suite also checked 512/768 cubed and the generic 288x512x512 shape.
The listed aligned cases matched cuBLAS exactly; the generic 1009x513x257 case
had maximum absolute error 0.0078125, within `atol=0.01, rtol=0.01`.

## Candidates and selection limits

Twelve configurations were screened on GPU 4 with the same random-input
timing harness, independently of the final two-round measurements. Candidate
parameters and all results are retained in the local artifacts. Selected rows:

| Block / warp | BK / stages | Registers | 1536 cubed (us) | 3072 cubed (us) | 4096 cubed (us) |
|---|---|---:|---:|---:|---:|
| Previous automatic dispatch | varies | varies | 46.316 | 358.154 | 765.768 |
| 96x64 / 48x32 | 32 / 3 | 162 | 58.225 | 394.250 | — |
| 96x128 / 48x64 | 32 / 3 | 237 | 39.444 | 329.226 | — |
| 96x128 / 32x64 | 32 / 3 | 160 | 46.029 | 377.508 | — |
| 96x128 / 48x64 | 16 / 4 | 176 | 48.538 | 394.957 | — |
| 256x128 / 64x64 | 32 / 2 | 226 | 65.096 | 417.802 | 914.442 |
| 64x128 / 32x64 | 64 / 2 | 242 | 61.901 | 403.927 | 940.227 |

A dash means the shape does not contain complete 96-row tiles, so that forced
candidate was not timed there. The 128x256 candidates and BK=16 three/four-stage
large tiles also failed to beat the corresponding baseline shapes. Tested
allocations stayed within the existing 48 KiB static-shared-memory limit;
these results do not rule out larger dynamic-shared-memory designs.

All screened native kernels had zero stack allocation. Thus avoiding spills
alone did not make a larger tile faster. A 256x128 block with 256 threads and
226 registers/thread allows only one resident block per SM. The selected
96x128 tile balances reuse, work per warp, and residency better on these shapes.

Small-grid checks showed why the selected tile needs a guard: 768 cubed took
18.196 us versus 12.442 us for the previous dispatcher. At 768x1536x2048 it
took 38.410 versus 35.471 us. Additional cutoff screening on GPU 3 measured
384x4096x512 at 17.756 versus 16.783 us. The final dispatcher retains the
previous path for these shapes. Some shapes below the conservative threshold
did improve, but the current rule does not try to capture every such case.

## NCU: less traffic and fewer instructions

Nsight Compute 2024.3.0 captured nine final reports serially on GPU 5 using
`--set full --replay-mode kernel --cache-control all --clock-control none
--import-source yes --profile-from-start off --launch-count 1`. After output
validation and five warmups, the harness brackets one selected call with
`cudaProfilerStart/Stop`, excluding its cuBLAS reference kernel. Temporary
administrator execution enabled counters without changing system settings.
These cache-flushed replay durations are distinct from ordinary warmed timings.

| Shape / metric | Previous mainloop | Reuse version | cuBLAS |
|---|---:|---:|---:|
| 1536 cubed: duration (us) | 48.032 | 40.448 | 36.160 |
| Executed warp instructions | 8,193,024 | 5,754,624 | 4,905,216 |
| L2 reads (MiB) | 162 | 126 | 90 |
| Tensor active (% active cycles) | 60.57 | 66.23 | 79.74 |
| Achieved occupancy (%) | 16.64 | 11.27 | 12.46 |
| Short scoreboard (cycles/issued instruction) | 0.59 | 0.28 | 0.11 |
| Long scoreboard (cycles/issued instruction) | 0.90 | 0.92 | 0.45 |
| 2304 cubed: duration (us) | 128.256 | 109.440 | 97.312 |
| Executed warp instructions | 26,459,136 | 18,297,792 | 13,917,744 |
| L2 reads (MiB) | 546.75 | 425.25 | 364.5 |
| Tensor active (% active cycles) | 66.58 | 77.03 | 88.70 |
| 3072 cubed: duration (us) | 309.568 | 287.904 | 253.856 |
| Executed warp instructions | 61,304,832 | 42,040,320 | 28,495,872 |
| L2 reads (MiB) | 1296 | 1008 | 864 |
| Tensor active (% active cycles) | 67.44 | 76.44 | 90.39 |
| Long scoreboard (cycles/issued instruction) | 0.95 | 0.75 | 0.03 |

L2 reads decrease by exactly 22.2% in all three shapes, matching the tile-reuse
calculation. Executed warp instructions decrease by 29.8–31.4%. Lower achieved
occupancy coexists with higher Tensor activity and shorter execution time.
This is not evidence of reduced DRAM traffic: actual DRAM reads are similar
or slightly higher in these captures. L2 reads include cache hits.

Matrix-load shared wavefronts equal their ideal counts for both custom
configurations. The selected kernel recorded zero hardware shared-load/store
conflicts in all three captures. Async copies still have excess shared
wavefronts: at 1536 cubed, 1,474,560 actual versus 1,032,192 ideal. Do not
interpret the matrix-load result as eliminating every shared-memory overhead.

The remaining gap includes extra input reads and instructions. At 1536 cubed,
the new version reads 40% more from L2 and executes 17.3% more warp instructions
than cuBLAS. At 3072 cubed those differences are 16.7% and 47.5%. Further
improvements need additional reuse or lower operand/control/epilogue overhead;
these measurements do not support increasing tile size unconditionally.

## Validation and reproduction

- `make -j2 check` passed, including all eleven GEMM versions and 42 cases each.
- `make -j2 sanitize` passed with 31 zero-error/zero-hazard summaries, including
  memcheck, racecheck, and synccheck for all six pipelined versions.
- Two new 384x6144 cases exercise the 192-block automatic-dispatch boundary
  with K=32/96, nontrivial alpha/beta, repeated calls, guards, and unaligned C.
  A 288x384x160 case covers non-64-row matrices in a forced BM=96 build.
- The forced 96x128 build passed all 42 CPU double-reference cases and all
  three sanitizer tools. Every screened candidate passed the previous 39-case
  suite plus a separate complete-tile CPU-reference case for its own geometry.
- sm_75 compilation passed. A compute_75-to-sm_80 build passed all 42 cases
  on A800, exercising synchronous copies and the two-K=8-MMA source branch.
  This is not a T4 runtime test.
- Final automatic dispatch passed the random-input comparisons and the NCU
  harness's full-output checks. SASS/resource comparison passed, as did the
  Makefile NCU build and export of all nine final reports.

```bash
CUDA_VISIBLE_DEVICES=4 make run-gemm-wmma-tiled-pipeline-reuse GEMM_ARGS='1536 1536 1536 100'
CUDA_VISIBLE_DEVICES=4 make run-gemm-wmma-tiled-pipeline-mainloop GEMM_ARGS='1536 1536 1536 100'
CUDA_VISIBLE_DEVICES=4 make check-gemm

# Force the new tile in an independent build directory.
make BIN_DIR=build/sm_80/reuse-forced \
  NVCCFLAGS='-O3 -std=c++14 -arch=sm_80 -Xcompiler -Wall -DGEMM_BM=96 -DGEMM_BN=128 -DGEMM_WM=48 -DGEMM_WN=64' \
  build/sm_80/reuse-forced/gemm_wmma_tiled_pipeline_reuse_bench
CUDA_VISIBLE_DEVICES=4 build/sm_80/reuse-forced/gemm_wmma_tiled_pipeline_reuse_bench --check-only

# Run profiling in a shell with access to performance counters.
CUDA_VISIBLE_DEVICES=4 make ncu-gemm \
  NCU_GEMMS='gemm_wmma_tiled_pipeline_mainloop gemm_wmma_tiled_pipeline_reuse gemm_cublas' \
  GEMM_ARGS='1536 1536 1536 2' NCU_DIR=build/sm_80/ncu-reuse
```

Ignored local artifacts are under `build/sm_80/step2-v1/`: the frozen baseline,
screening configurations and results, `dispatch-screen.csv`, `cutoff-screen.csv`,
`final-random.csv`, timing/capture harnesses, build/check/sanitizer logs,
resource reports, and SASS hashes. `ncu-candidate/1536/` contains the initial
three reports; `ncu-final/{1536,2304,3072}/` contains the nine final reports and
their exported text/CSV summaries. Detailed counters are in
`ncu-final/comparison_detail.csv`.
