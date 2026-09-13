# NCU analysis: pipelined WMMA versus cuBLAS

The next experiments should first simplify the aligned input path and improve
the shared-memory layout. Deeper prefetching remains useful to test, but the new
capture gives stronger evidence for excess address/control instructions and
shared-memory wavefronts. DRAM bandwidth and occupancy alone do not explain the gap.

## Capture

Collected on 2026-09-13, NVIDIA A800 80GB PCIe (physical GPU 5, 108 SMs),
driver 535.247.01, CUDA 12.6.20, Nsight Compute 2024.3.0. Source revision:
`fa1b7ac` (`add gemm_wmma_tiled_pipeline`). Binaries were rebuilt with
`-O3 -std=c++14 -arch=sm_80 -Xcompiler -Wall -lineinfo`.

All three implementations used M=N=K=1024, alpha=1, beta=0, FP16 inputs/output,
and FP32 accumulation, and passed CPU-reference validation. The Makefile command
used `NCU_GEMMS="gemm_wmma_tiled gemm_wmma_tiled_pipeline gemm_cublas"`,
`GEMM_ARGS="1024 1024 1024 2"`, and the default full metric set, kernel replay,
cache flushing, unchanged clocks, launch skip 7, and launch count 1. This captures
the first benchmark launch after two correctness calls and five warmups. The
reports used 47, 49, and 47 replay passes respectively.

Counter access used temporary sudo for NCU only. The sudo ticket was invalidated
after collection; `RmProfilingAdminOnly` remains 1. No system settings changed.

Reports and exports are in `build/sm_80/ncu/pipeline-mFHaFQ/`, ignored by Git:

- Each variant has `.ncu-rep`, `.txt`, `.raw.csv`, and `.instructions.json` files.
- `comparison.csv` contains the Makefile's summary metrics.
- `source_analysis.json` and `source_aggregates.json` contain PC hotspots,
  dynamic opcode counts, and shared-memory wavefront breakdowns.
- `capture.json` records the collection configuration; `inspect_sources.py`
  reproduces the source/PC inspection using the installed NCU Python interface.

## What changed

| Metric | Original tiled | Pipeline | cuBLAS |
|---|---:|---:|---:|
| Captured kernel duration (us) | 166.080 | 46.112 | 17.376 |
| Dynamic warp instructions | 29,062,144 | 9,336,832 | 1,701,216 |
| HMMA instructions | 524,288 | 524,288 | 540,672 |
| HMMA share of warp instructions | 1.80% | 5.62% | 31.78% |
| Tensor pipe active (% of peak sustained active) | 4.22 | 16.39 | 58.84 |
| Registers/thread | 96 | 62 | 196 |
| Achieved occupancy | 14.86% | 14.72% | 6.17% |
| Theoretical occupancy | 31.25% | 37.50% | 12.50% |
| DRAM throughput (% of peak sustained elapsed) | 1.32 | 4.74 | 12.64 |
| Shared-load bank conflicts | 1,048,576 | 1,048,576 | 0 |

Pipeline is 3.60x faster than the original tiled implementation in this capture.
cuBLAS is still 2.65x faster than pipeline. The pipeline reduced total instructions
by 67.9% while retaining the same HMMA count; it still executes 5.49x as many
warp instructions as cuBLAS. Tensor-active percentages use active-cycle
denominators and are not percentages of end-to-end peak GEMM FLOP/s.

These NCU durations are from one captured launch per variant with cache flushing
and replay. The separate ordinary benchmark measured roughly 43.9 versus 15.3 us
(2.87x) for pipeline/cuBLAS. Use that benchmark for steady-state timing; do not use
the benchmark's CUDA-event printout produced while running inside NCU.

## 1. Simplify the aligned main loop

Pipeline's warp-state counters, in stall cycles per issued instruction:

| Reason | Original tiled | Pipeline | cuBLAS |
|---|---:|---:|---:|
| Fixed-latency dependency (`wait`) | 1.28 | 2.28 | 1.88 |
| Long scoreboard | 4.48 | 0.94 | 0.39 |
| Short scoreboard | 0.33 | 0.78 | 0.10 |
| Branch resolving | 0.08 | 0.44 | 0.03 |
| Block barrier | 0.14 | 0.16 | 0.09 |
| Memory barrier (`membar`) | 0 | 0 | 0 |

Pipeline averages 6.35 warp cycles per issued instruction; fixed-latency `wait`
accounts for about 36% of that interval. This metric does not mean time inside
the source-level `__pipeline_wait_prior()` call. It includes dependencies between
instructions. Samples are attached to the waiting consumer instruction; source
and SASS must be inspected to identify its dependencies.
[NCU stall definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference)

The hottest individual `wait` locations include branches at the pack loop,
bounds check, and alignment check in `stage_input_tile` (source lines 16, 21,
23), plus the address calculation at line 22. Pipeline has 537,600 BRA,
364,544 BSSY, and 233,472 BSYNC instructions. Roughly 6.18 million instructions
have integer/address/logic opcodes (IMAD/IADD/ISETP/LEA/SHF/LOP and their uniform
counterparts). By comparison, cuBLAS's `wait` samples mostly sit on HMMA:
634 of 799 samples, versus 278 of 2,194 for pipeline.

This aligned shape executes zero scalar LDG input instructions: fallback is not
being taken. Branch efficiency is 100%, so warp divergence is not the observed
problem. Even converged loops/checks and address dependencies consume instructions.

**Experiment:** dispatch a complete/aligned tile specialization from `solve`.
Use compile-time pack counts, unroll the fixed per-thread copy loop, and advance
precomputed A/B addresses by K-chunk strides. Keep `size_t` global addressing
and the existing generic tail/unaligned path. Check whether instruction count,
branch resolving, and fixed-dependency waits fall, alongside ordinary timings.

## 2. Improve both sides of the shared-memory layout

Pipeline has 1,048,576 hardware shared-load bank conflicts, unchanged from tiled.
The report shows these account for 48.98% of the 2,140,927 hardware shared-load
wavefronts. Source-correlated counters isolate the fragment loads more precisely:

| Path | Actual source wavefronts | Ideal | Excessive |
|---|---:|---:|---:|
| Pipeline LDSM loads (A+B) | 2,097,152 | 1,048,576 | 1,048,576 |
| Pipeline LDGSTS copies | 1,114,112 | 524,288 | 589,824 |
| Pipeline output STS.64 | 65,536 | 32,768 | 32,768 |
| cuBLAS LDSM loads (A+B) | 640,640 | 640,640 | 0 |
| cuBLAS LDGSTS copies | 476,032 | 344,960 | 131,072 |

The extra LDSM wavefronts occur at the A/B `load_matrix_sync` instructions.
The async copies also have excessive source wavefronts. The shared layout and
per-lane copy mapping should therefore be evaluated together. These source
wavefront counters are distinct from hardware bank-conflict counts, and neither
their percentages nor NCU's estimated speedups are fractions of removable wall time.
[Shared-memory metric definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#shared-memory)

**Experiment:** first compare SKEW=0/16/32 and lane-to-copy mappings while keeping
the math tile fixed. Then consider a permuted layout with matching `ldmatrix`
loads and `mma.sync`, or suitable CUTLASS components. Arbitrary swizzles cannot
be passed unchanged to WMMA's conventional row/column-major load interface.
Validate both copy-side and LDSM-side wavefront ratios, plus short-scoreboard
hotspots on the dependent HMMA instructions.

One accounting detail: 393,216 of the opcode-counted LDS instructions are
`@!PT LDS RZ, [RZ]`. They issue with all lanes predicated off and perform no
shared-memory reads; they also appear in the uninstrumented binary. Do not
interpret the 425,984 total LDS opcode count as that many actual shared loads.

## 3. Test deeper prefetching and instruction scheduling

Long-scoreboard waits remain at 0.94 cycles/instruction. Of the 934 PC samples
for this reason, 824 occur at the first HMMA immediately after the loop's
`DEPBAR.LE SB0, 0x0`; another 105 occur at the barrier after the prologue wait.
This is evidence to inspect the async-completion region, not evidence that HMMA
itself loads global memory. The generated loop includes this ordering:

```text
LDSM ... current tile's second fragment group
DEPBAR.LE SB0, 0x0
HMMA ... second fragment group
...
BAR.SYNC.DEFER_BLOCKING
```

The compiler placed the wait before some of the current tile's final HMMA
instructions, despite its later position in the CUDA source. The current
two-buffer pipeline also drains all submitted copies each round.

**Experiment:** try three stages with prefetch/commit/wait/drain logic that only
waits for the stage being consumed. Inspect SASS to verify the resulting overlap.
Also try interleaving next fragment loads with independent current MMA work.
Keep the block-wide lifetime protection for buffers; the zero `membar` counter
does not justify removing synchronization. At current tile sizes, 3/4 input
stages need 37/48 KiB static shared memory, versus 26 KiB now.

## Lower-priority work and measurement limits

- Retune BM/BN/BK and warp tile sizes after the above changes. Pipeline has 256
  blocks for 108 SMs, with 9.42 achieved active warps/SM. Its theoretical six-block
  residency is not filled by this grid. However, cuBLAS has only 88 blocks and
  lower occupancy while executing much less supporting work per HMMA. Increasing
  occupancy or reducing tile size is not an objective by itself.
- Consider vectorized FP16 output and alpha=1/beta=0 specialization after main-loop
  work. The output scratch stores contribute only 32,768 of the 1,671,168 total
  excessive source wavefronts. They are a smaller opportunity than A/B movement.
- DRAM reads are about 4.2 MB in all three reports; pipeline reaches only 4.74%
  of peak DRAM throughput, with about 90% L2 hit rate. The evidence favors reducing
  instruction and dependency costs over treating this as DRAM bandwidth saturation.
- NCU reports no eligible warp in 62.78% of pipeline scheduler cycles, but cuBLAS
  has lower issue activity and is faster. Read issue statistics together with
  useful tensor work per instruction and total duration.
- Clocks were unchanged. Each PM timeline dropped three samples and used a
  maximum 20,000-cycle interval, too coarse for detailed phase-duration claims.
  PC stall samples are statistical, and metrics may come from separate replay
  passes. The exact expected speedup of each proposed optimization remains unmeasured.

The recommended order is: **aligned-path simplification -> shared layout/copy
mapping -> deeper pipeline/scheduling -> tile and epilogue tuning**. Measure one
change at a time with correctness/sanitizer checks and ordinary benchmark timing.
