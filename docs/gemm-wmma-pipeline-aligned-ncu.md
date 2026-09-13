# Aligned WMMA pipeline versus cuBLAS: NCU follow-up

These captures precede the shared-layout changes. See the
[shared-layout experiment](gemm-wmma-shared-layout.md) for the resulting
implementation, new NCU captures, and larger block/warp tile trials.

The full/aligned specialization removes much of the generic pipeline's control
work, but cuBLAS still uses input data more efficiently and sustains more Tensor
Core work. Fresh captures show a 1.76x duration gap at 1024 cubed and a 2.22x gap
at 2048 cubed. The larger case exposes shared-memory pressure that was less
prominent in the smaller grid.

## Collection

Collected on 2026-09-13 from commit `5eed6d5`, on physical GPU 5, NVIDIA A800
80GB PCIe (108 SMs), driver 535.247.01, CUDA 12.6.20, Nsight Compute 2024.3.0.
The existing kernel implementations were profiled without modifications.

- Both variants were built with `-O3 -std=c++14 -arch=sm_80 -Xcompiler -Wall
  -lineinfo` and run serially on the same GPU.
- FP16 contiguous aligned A/B/C, FP32 accumulation, alpha=1, beta=0. The cuBLAS
  wrapper disables reduced-precision reductions. All four runs passed the CPU
  reference, with maximum absolute error 0.03125 within the test's tolerance.
- `--set full --replay-mode kernel --cache-control all --clock-control none
  --launch-skip 7 --launch-count 1 --import-source yes` captured one launch per
  variant and shape, each requiring 47 replay passes. Skip 7 omits two correctness
  calls and five warmups. The benchmark repeat argument was 2.
- Durations below are NCU kernel durations, excluding allocation, transfers,
  validation, and host submission. The benchmark's own event times while running
  under NCU are affected by profiling and must not be used for this comparison.
- GPU clocks and system settings were left unchanged. Measured SM frequencies
  were approximately 1.35–1.39 GHz. Every report had three dropped PM samples;
  the short kernels do not support reliable phase-level analysis from that
  timeline. PC sampling is statistical. cuBLAS source was unavailable, but its
  kernel name, SASS, and counters were captured.

## Performance and resource use

| Metric | 1024 aligned | 1024 cuBLAS | 2048 aligned | 2048 cuBLAS |
|---|---:|---:|---:|---:|
| Kernel duration (us) | 30.368 | 17.216 | 163.520 | 73.504 |
| Useful throughput (TFLOP/s) | 70.72 | 124.74 | 105.06 | 233.73 |
| Tensor active (% of active cycles) | 25.27 | 59.31 | 36.66 | 85.25 |
| Tensor active (% of elapsed cycles) | 23.18 | 42.70 | 34.10 | 79.37 |
| Achieved occupancy (%) | 14.56 | 6.24 | 27.47 | 10.70 |
| Registers/thread | 92 | 196 | 92 | 254 |
| Threads/block | 128 | 128 | 128 | 128 |
| Blocks | 256 | 88 | 1024 | 208 |
| User shared memory/block (KiB) | 26 | 56 | 26 | 72 |

Throughput is `2*M*N*K / duration`; the Tensor percentages have different cycle
denominators and are not the percentage of theoretical GEMM FLOP/s achieved.
The earlier ordinary warm benchmark's 1.90x gap at 1024 cubed used a different
timing/cache scope and is not contradicted by this NCU result.

At 2048 cubed our achieved occupancy approaches its theoretical 31.25%, yet
the gap widens. cuBLAS uses more registers and has lower occupancy, but spends
far more of its active cycles issuing Tensor work. Occupancy alone is therefore
an unsuitable optimization target for this comparison.

The selected cuBLAS kernels have `tilesize96x128x32_stage4` at 1024 and
`tilesize160x128x32_stage4` at 2048, both with `warpsize2x2x1` and
`tensor16x8x16`. The custom kernel uses 64x64x32, four warps with a 32x32 tile
each, and two shared input stages. cuBLAS dimensions refer to the library's
column-major operation; the wrapper swaps operands to compute the row-major
result, so its output axes are exchanged. This does not affect these square
problem comparisons.

## More reuse, less data movement

| Metric | 1024 aligned | 1024 cuBLAS | 2048 aligned | 2048 cuBLAS |
|---|---:|---:|---:|---:|
| L2 read traffic from TEX (MiB) | 64 | 38 | 512 | 232 |
| DRAM read traffic (bytes) | 4,216,192 | 4,242,688 | 16,799,616 | 16,848,000 |
| DRAM throughput (% of peak) | 7.18 | 12.76 | 6.27 | 13.39 |
| Executed warp instructions | 3,326,976 | 1,701,216 | 23,400,448 | 10,043,904 |
| HMMA instructions | 524,288 | 540,672 | 4,194,304 | 4,259,840 |
| HMMA share of instructions | 15.76% | 31.78% | 17.92% | 42.41% |
| Integer/address/logic instruction group | 1,812,480 | 541,024 | 12,296,192 | 2,542,592 |

L2 traffic is `lts__t_sectors_srcunit_tex_op_read.sum * 32`. Our smaller output
tiles reload A/B more often across blocks, even when the data hits L2. At 2048,
this costs 2.21x the L2 read traffic. Similar DRAM read byte counts and low DRAM
throughput do not support an HBM bandwidth saturation explanation.

For a complete output tile, ignoring the epilogue, reuse yields an input
arithmetic intensity of `BM*BN/(BM+BN)` FLOP/byte with FP16 storage. This is 32
for 64x64, approximately 54.9 for 96x128, and 71.1 for 160x128. This calculation
explains the direction of the measured traffic change; it is not a direct
prediction of runtime speedup.

cuBLAS executes slightly more HMMA work because its tiles do not exactly cover
these dimensions: about 3.1% extra at 1024 and 1.6% at 2048. It nevertheless
finishes sooner, so the improvement does not come from doing fewer multiply-adds.
Larger warp tiles amortize operand loads and control instructions over more MMA
operations. The integer/address/logic group above includes IMAD, IADD, ISETP,
LEA, SHF, LOP and their listed uniform counterparts; it is an opcode grouping,
not a measurement assigning all those instructions to address calculation.

## Shared-memory layout is a concrete remaining cost

| Matrix-load source metric, summed over LDSM | 1024 aligned | 1024 cuBLAS | 2048 aligned | 2048 cuBLAS |
|---|---:|---:|---:|---:|
| Actual wavefronts | 2,097,152 | 640,640 | 16,777,216 | 3,863,808 |
| Ideal wavefronts | 1,048,576 | 640,640 | 8,388,608 | 3,863,808 |
| Actual / ideal | 2.00x | 1.00x | 2.00x | 1.00x |
| Whole-kernel shared-load bank conflicts | 1,048,576 | 0 | 8,433,869 | 18,666 |

Both our A `LDSM.16.M88.4` and B `LDSM.16.MT88.4` loads require twice their
ideal wavefront count. SKEW=16 preserves alignment but does not eliminate these
conflicts. At 2048, our ideal matrix-load work is already 2.17x cuBLAS because
of lower operand reuse; the additional factor of two produces 4.34x actual LDSM
wavefronts. Improving the layout alone leaves the reuse difference to address.

Async global-to-shared copies also have excess source wavefronts: 589,824 versus
131,072 at 1024, and 4,718,592 versus 1,047,012 at 2048. These source metrics and
whole-kernel hardware bank-conflict counters measure different things and should
not be added together. In particular, cuBLAS's LDSM source accesses meet their
ideal count even though its 2048 whole-kernel hardware counter is nonzero.

NVIDIA's CUTLASS examples use coordinated permuted shared layouts and
`ldmatrix` access mappings to avoid bank conflicts. Such a change must update
both stores and loads; applying an XOR only to the current stores would change
the matrix seen by `wmma::load_matrix_sync`. See the
[CUTLASS shared-memory layout example](https://docs.nvidia.com/cutlass/4.3.3/media/docs/cpp/implicit_gemm_convolution.html#shared-memory-layouts)
and [warp-level GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html#warp-level-matrix-multiply-api).

## Waiting changes with problem size

The following hardware ratios are warp stall cycles per issued instruction,
not percentages of kernel wall time:

| Stall | 1024 aligned | 1024 cuBLAS | 2048 aligned | 2048 cuBLAS |
|---|---:|---:|---:|---:|
| Long scoreboard | 5.41 | 0.39 | 2.77 | 0.08 |
| Short scoreboard | 1.68 | 0.09 | 3.41 | 0.35 |
| MIO throttle | 0.47 | 0.02 | 3.78 | 0.10 |
| Barrier | 0.49 | 0.10 | 2.45 | 0.33 |
| Fixed-latency wait | 1.81 | 1.88 | 1.98 | 2.60 |

At 1024, 1,598 of 1,798 long-scoreboard samples (88.9%) fall on two HMMA PCs
immediately following `DEPBAR.LE SB0, 0x0`. At 2048, the same pair accounts for
3,748 of 4,362 samples (85.9%). The generated loop has the following pattern:

```text
LDSM ...                  // load the current chunk's remaining fragments
DEPBAR.LE SB0, 0x0         // async-copy dependency wait
HMMA ...                  // dominant long-scoreboard sample location
HMMA ...
```

Although the source calls `__pipeline_wait_prior(0)` after the WMMA loop, the
compiler schedules its wait before the last eight HMMA instructions of each
K chunk. This supports investigating lost overlap at the copy dependency and
the timing of shared-to-register loads. A sample on HMMA does not by itself
identify Tensor arithmetic as the cause: stall samples can be reported on a
consumer waiting for an earlier producer. Long scoreboard denotes an L1TEX
dependency, whereas short scoreboard and MIO throttle can indicate shared-memory
dependencies and queue pressure. See the
[NCU metric definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference).

At 2048, additional active warps hide some long-latency waits, but shared-memory
dependencies, MIO pressure, and block synchronization become more pronounced.
The MIO sample hotspots include LDSM, LDGSTS, and the loop barrier. This makes
layout and reuse improvements especially relevant to the larger case. The
kernel's memory-barrier stall ratio is zero; searching for a large `membar`
counter would miss the waits observed here.

cuBLAS uses four stages according to the selected kernel names and shows far
smaller long-scoreboard ratios. This is consistent with better latency hiding,
but the captures also differ in tiles, register use, shared layout, and instruction
scheduling. They cannot isolate how much of the advantage comes from stage count.

## Next experiments, in order

1. **Improve shared layout and operand reuse.** First measure legal WMMA padding
   alternatives against the current LDSM actual/ideal counts. If that interface
   prevents a useful layout, add a separate `ldmatrix`/`mma.sync` implementation
   with coordinated permuted stores and loads. Then compare 128x64 or 64x128
   output tiles with 64x32 or 32x64 warp tiles, retaining four warps. Track
   L2 read traffic, LDSM wavefronts, register use, and total kernel time at both
   sizes. These are candidates, not measured improvements.
2. **Improve overlap at both levels.** Compare register-fragment buffering and
   interleaving LDSM with independent HMMA work, then test three input stages
   with correct prologue, steady-state waits, and drain. Inspect generated SASS
   as well as long/short-scoreboard samples. CUTLASS describes both shared-tile
   and register-fragment buffering in its
   [pipelining discussion](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#pipelining).
   At the current tile sizes, three stages would raise user shared storage from
   26 to 37 KiB, and four stages to 48 KiB, before changes in register allocation.
   More stages consume occupancy resources and need measurement.
3. **Amortize loop/control work and optimize the epilogue.** BK=64 is a separate
   candidate to reduce chunk count and barriers, with increased storage cost.
   The current epilogue uses 32,768 scalar STG instructions at 1024 versus
   cuBLAS's 4,224 vector STG instructions. Vector stores are a smaller target
   than the mainloop, and would need their own output-alignment handling.

The custom kernel's 131,072 LDS instructions at 1024 include 98,304
`@!PT LDS RZ, [RZ]` instructions that perform no actual memory access. Do not
interpret their count as input reloads or register spilling. Source-wavefront
metrics separately identify the real accesses.

Change one variable at a time and compare duration, correctness, and sanitizer
results. Reductions in wavefronts or normalized stall ratios are diagnostic
evidence; they do not translate directly into additive speedup estimates.

## Reproduction and local artifacts

```bash
make -j2 ncu-gemm-build NCU_GEMMS="gemm_wmma_tiled_pipeline_aligned gemm_cublas"
# Requires performance-counter access; temporary sudo usage is in the README.
CUDA_VISIBLE_DEVICES=5 make ncu-gemm \
  NCU_GEMMS="gemm_wmma_tiled_pipeline_aligned gemm_cublas" \
  GEMM_ARGS="1024 1024 1024 2" NCU_DIR=build/ncu-gap-1024
CUDA_VISIBLE_DEVICES=5 make ncu-gemm \
  NCU_GEMMS="gemm_wmma_tiled_pipeline_aligned gemm_cublas" \
  GEMM_ARGS="2048 2048 2048 2" NCU_DIR=build/ncu-gap-2048
```

This run is retained locally under `build/sm_80/ncu/cublas-gap-meiSAU/`:
`1024/` and `2048/` contain the NCU reports, text exports, raw CSVs, and per-PC
instruction counts. `gap_comparison.csv`, `gap_details.json`, `capture.json`,
and `analyze.py` record the extracted values, source/SASS hotspots, settings,
and extraction logic. Generated artifacts are ignored by Git.
