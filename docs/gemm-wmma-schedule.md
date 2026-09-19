# Distributed copies, operand scheduling, and compact addresses

`src/gemm_wmma_tiled_pipeline_schedule.cu` is an independent successor to
`gemm_wmma_tiled_pipeline_large.cu`. The investigation changed the mainloop in
three steps: distribute asynchronous copies through MMA, overlap fragment loads
and stage transitions with MMA, then reduce shared/global address bookkeeping.
The XOR shared layout and alpha=1/beta=0 epilogue are retained.

## Contract and configuration

The contract remains row-major FP16 A[M,K], B[K,N], and C[M,N], with FP32
accumulation and `C = alpha * A * B + beta * C`. M/N=0 is a no-op; K=0 scales
C by beta. Beta=0, including negative zero, ignores old C. Native tiles require
complete dimensions and 16-byte-aligned A/B; C may be only half-aligned. Tails
and unaligned A/B use the fixed 64x64 generic WMMA pipeline.

`GEMM_SCHEDULE` selects cumulative experiments:

| Value | Mainloop |
|---|---|
| 0 | Previous version's loop and addresses |
| 1 | Distribute next-stage A/B copies among MMA row groups |
| 2 | Also carry operand registers across stages and interleave fragment loads |
| 3, default | Also use 32-bit shared addresses and a compact global-copy iterator |

`GEMM_INTERLEAVE_LOADS=0` isolates the stage-boundary change in step 2.
`GEMM_ADDRESS_MODE=1` isolates 32-bit operand addressing from compact global
copy pointers; mode 2 enables both. The default is mode 2 for schedule 3 and
mode 0 for earlier schedules. `GEMM_REGISTER_PIPELINE=0` disables register
prefetch and the early stage transition. BK=16 retains the original stage
boundary because it has only one MMA K group.

Explicit `GEMM_BM/BN/BK/WM/WN/STAGES` overrides disable automatic tile selection
unless `GEMM_AUTO_TILE` is explicitly set. The previous dynamic shared-memory
opt-in, capacity check, and generic fallback remain. No system configuration
changes are required by the implementation.

Automatic selection applies the new 128x128 block / 64x64 warp / BK=32 path
only for exact alpha=1/beta=0, complete 128-row/column tiles, M/N in [1024,4096],
and K in [128,8192], divisible by 32. Its grid must have either 64..108 or
256..1024 blocks. Three stages occupy 48 KiB. Grids of 256..324 blocks with
K>=1024 first try four stages (64 KiB), falling back to the static three-stage
path if opt-in capacity is unavailable. These are measured A800 bands, not a
portable autotuner or a claim that every intermediate shape is optimal.

Other shapes and general alpha/beta keep the preceding dispatcher. In
particular, 1536 cubed loses with the new tile's 144-block grid. Three-stage
6144/8192-cubed results also regress; four stages recover most of that loss
without a consistent advantage. Those shapes therefore retain the previous
kernels.

## Step 1: distribute copies

The preceding loop issued the next stage's entire A/B copy batch at its head.
The new iterator assigns each 16-byte copy to one compile-time group and issues
that group after an MMA row. Pointer advances occur exactly once per copied
chunk. All copies finish issuing before the last MMA K group. There is still
one commit per complete stage, including empty drain stages, so the fixed
`wait_prior(STAGES-2)` distance remains valid for short K.

This experiment was deliberately evaluated before changing operand scheduling.
At 4096 cubed it was slower: NCU duration increased from 662.912 to 690.432 us,
while executed warp instructions increased from 104.337 to 111.485 million.
Reducing the long-scoreboard metric alone did not improve elapsed time.

The grouping and stage-boundary approach can be compared with NVIDIA's
[CUTLASS 3.5.1 Ampere multistage loop](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/gemm/threadblock/mma_multistage.h#L456).
This is an implementation reference, not evidence of cuBLAS's internal schedule.

## Step 2: overlap operands and the stage boundary

Two operand register slots persist across mainloop iterations. Once the final
K group's operands have been loaded, the block can wait for the next shared
stage and release its current shared stage before doing that final group's MMA.
The final group then prefetches the first operands of the next iteration into
the alternate slot. The last iteration skips that prefetch to avoid reading an
unfilled drain stage.

The interleaved variant loads each next B fragment during the first M-fragment
row, and each next A fragment after its current row's MMA. Each fragment is
loaded once per K group and retains its reuse across the other output dimension.
The block barrier remains: it establishes readiness across producer warps and
prevents a faster warp from reusing a stage still read by another warp.

NCU duration at 4096 cubed falls to 636.704 us with the boundary change, then
613.504 us with interleaved fragment loads. The latter has 255 registers/thread
and an 8-byte stack allocation in the specialized kernel. It improves scheduling
but increases register lifetimes and address work, motivating step 3.

## Step 3: shared addresses and compact copy pointers

The SASS contains repeated address formation and predicated updates of several
64-bit input pointers. The operand loader now keeps 32-bit shared-memory bases
and lane offsets. K/fragment XOR adjustments stay separate from whole-stage
and whole-16-row offsets. This preserves the existing shared layout.

When a thread's copy iterations differ only by whole input rows, the global
iterator stores one advancing source pointer and a row stride instead of an
independently advancing pointer for every copy. Layouts that do not satisfy
that condition retain the original per-copy representation. Inactive copy
threads remain predicated, and no out-of-bounds input pointer is formed.

The fixed 128x128 / 64x64 / BK32 / three-stage specialized kernel shows:

| Variant | Static instructions in K-loop backedge | Registers/thread | Stack bytes |
|---|---:|---:|---:|
| Previous loop | 193 | 231 | 0 |
| Step 1 | 206 | 230 | 0 |
| Step 2, boundary and interleaving | 218 | 255 | 8 |
| Step 3, shared addresses only | 177 | 255 | 0 |
| Step 3, shared addresses and compact pointers | 152 | 254 | 0 |

These static counts include predicated-off instructions; they are not dynamic
instruction counts or a timing model. Every variant still has 64 HMMA,
16 LDSM, eight LDGSTS, and one block barrier in that loop. The optimization
changes their scheduling and surrounding work rather than their arithmetic.

## NCU comparison

NCU 2024.3.0 collected full reports on one warmed, validated launch, using kernel
replay, `--cache-control all`, and `--clock-control none`. Profile capture was
bracketed by cudaProfilerStart/Stop. Temporary administrator execution enabled
counters without changing driver or system settings. These cache-flushed replay
durations must not be mixed with the ordinary warmed timings below.

The following 4096-cubed ablation holds the custom tile, warp shape, BK, and
three-stage allocation fixed:

| Metric | Previous | Step 1 | Step 2, interleaved | Step 3 | cuBLAS |
|---|---:|---:|---:|---:|---:|
| NCU duration (us) | 662.912 | 690.432 | 613.504 | 597.440 | 550.816 |
| Tensor active (% active cycles) | 74.39 | 73.70 | 86.15 | 88.34 | 93.16 |
| Executed warp instructions (million) | 104.337 | 111.485 | 117.989 | 83.022 | 61.411 |
| Short scoreboard (cycles/issued instruction) | 0.37 | 0.44 | 0.18 | 0.02 | 0.03 |
| Long scoreboard | 0.55 | 0.17 | 0.13 | 0.32 | 0.06 |
| Barrier | 0.87 | 0.23 | 0.19 | 0.04 | 0.05 |
| MIO throttle | 0.05 | 0.04 | 0.02 | 0.07 | 0.03 |

The combined change reduces the fixed-tile replay duration by 9.9%, leaving
8.5% above cuBLAS. Shared-load dependency and barrier metrics improve markedly,
while the long-scoreboard metric rises again after address simplification.
These are diagnostic clues, not independent contributions to elapsed time.
The final three-stage kernel still makes 2048 MiB of L2 reads at 4096 cubed,
versus cuBLAS's 1536 MiB, and executes about 35% more warp instructions.
Further work should consider prefetch lead distance and reuse together with
this new mainloop, rather than assuming higher occupancy alone will help.

## Ordinary measurements

The independent timing runs below were collected on 2026-09-13 on NVIDIA A800 80GB PCIe (108 SMs), CUDA 12.6.20, driver 535.247.01.
Inputs are random FP16 with seed 12345; alpha=1/beta=0. Each process validates
all outputs and guards against cuBLAS using FP32 reductions before timing.
Five warmups precede the median of five CUDA-event batches of 100 launches.
Allocation, transfers, reference generation, and validation are excluded.
Host submission gaps and dynamic-launch configuration costs are included.
GPU measurements run serially on one device; clocks are not locked.

The final dispatcher selects the configurations below. Values are means of two
runs with reversed implementation order from the earlier independent screen,
not a fresh measurement of the final wrapper. On 2026-09-19 the devices were
busy with other workloads, and the user requested validation only, retaining
these earlier performance results. SASS comparisons check that the selected
kernels match those measured configurations.

| Shape (M,N,K) | Previous automatic (us) | Selected configuration (us) | cuBLAS (us) | Selected vs cuBLAS |
|---|---:|---:|---:|---:|
| 1024 cubed | 19.272 | 17.828 | 15.375 | +16.0% |
| 1536 cubed, previous path retained | 38.728 | 38.728 | 34.852 | +11.1% |
| 2048 cubed, four stages | 98.171 | 89.308 | 74.936 | +19.2% |
| 2304 cubed, four stages | 119.496 | 107.095 | 105.902 | +1.1% |
| 3072 cubed | 337.004 | 312.653 | 304.445 | +2.7% |
| 4096 cubed | 773.161 | 716.841 | 652.892 | +9.8% |
| 6144 cubed, previous path retained | 2611.175 | 2611.175 | 2118.805 | +23.2% |
| 8192 cubed, previous path retained | 6796.262 | 6796.262 | 4975.416 | +36.6% |
| 2048,4096,1024 | 105.263 | 83.353 | 80.768 | +3.2% |
| 4096,2048,1024 | 108.011 | 83.983 | 80.256 | +4.6% |
| 2048,2048,128 | 15.006 | 13.563 | 15.314 | -11.4% |

Positive percentages mean slower. At 4096 cubed the selected configuration
reduces our time by 7.3%; matching cuBLAS would need another 8.9% reduction
from that selected time. At 2048 cubed the corresponding reductions are 9.0%
and a further 16.1%. Results within a few percent should be interpreted in
light of unlocked clocks and run-to-run variation.

Additional two-order screens covered 1280/2560/3584 squares, rectangles such as
1024x4096 and 1536x3072, and K values from 32 to 8192. These support the bounded
dispatch bands, while retaining the previous short-K<128 and large-grid paths.
The default new path is limited to exact alpha=1/beta=0 because performance
selection was measured with those scalars; forced builds test general scalars
for correctness as well.

## Validation

On 2026-09-19, `make check` passed all fourteen default GEMMs plus four forced
configurations (the previous large-tile test and three scheduled tests), with
57 CPU-reference cases per executable, as well as the repository's other
operator checks. `make sanitize` completed with 52 zero-error summaries,
including memcheck, racecheck, and synccheck on the default scheduled binary and
all three forced scheduled configurations. No fresh performance measurements
were taken during this validation pass on the shared device.

The five new cases cover BK64 one/three/nine-chunk rings,
NaN old C, unaligned output, and both scheduled automatic grid bands.

Final SASS comparison confirms that all twelve retained legacy kernels match
the preceding implementation. The selected three/four-stage kernels match the
frozen configurations measured above. Normal, optional perf, and Makefile NCU
builds contain the same sixteen kernel instruction sequences. Default kernels
and the forced three/four-stage tests have zero stack/local allocation; the
forced BK64/two-stage/16-warp stress configuration has up to 40 bytes of stack
and is a correctness target, not an automatically selected performance path.

The earlier compatibility pass checked the 55-case suite on forced dynamic
storage, an artificial 48 KiB capacity cap, BK=16, inactive copy threads, disabled
register prefetch, and a compute_75-front-end/sm_80-back-end build. A real sm_75
binary also compiles, including a fresh final-source compile; no Turing GPU was
available for runtime validation. Initcheck with uninitialized C passed for
seven shapes in the forced dynamic build. The optional perf harness also passes
argument-validation checks and validated single-launch runs without timing.

## Reproduction

Small CPU-reference correctness checks and sanitizer runs:

```sh
make -j3
CUDA_VISIBLE_DEVICES=0 make check
CUDA_VISIBLE_DEVICES=0 make sanitize
```

`GEMM_SCHEDULE_TESTS` adds fixed 128x128/BK32 three- and four-stage checks plus
a 128x256/BK64 two-stage dynamic-storage check. These force the new mainloop
on small inputs, independently of the automatic dispatch bands.

For large-shape timings, the optional `*_perf` targets use
`benchmarks/gemm.cu` instead of the cubic CPU reference. They always validate
against cuBLAS before reporting timing; `make check` remains the independent
CPU-reference check.

```sh
make -j3 build/sm_80/gemm_wmma_tiled_pipeline_large_perf \
  build/sm_80/gemm_wmma_tiled_pipeline_schedule_perf \
  build/sm_80/gemm_cublas_perf
CUDA_VISIBLE_DEVICES=0 build/sm_80/gemm_wmma_tiled_pipeline_large_perf 4096 4096 4096
CUDA_VISIBLE_DEVICES=0 build/sm_80/gemm_wmma_tiled_pipeline_schedule_perf 4096 4096 4096
CUDA_VISIBLE_DEVICES=0 build/sm_80/gemm_cublas_perf 4096 4096 4096
```

Reproduce the controlled fixed-tile steps with separate build directories:

```sh
for step in 0 1 2 3; do
  make BIN_DIR="build/sm_80/schedule-$step" \
    NVCCFLAGS="-O3 -std=c++14 -arch=sm_80 -lineinfo -DGEMM_SCHEDULE=$step \
      -DGEMM_BM=128 -DGEMM_BN=128 -DGEMM_BK=32 -DGEMM_WM=64 -DGEMM_WN=64 \
      -DGEMM_STAGES=3 -DGEMM_MULTISTAGE_MIN_BLOCKS=2" \
    "build/sm_80/schedule-$step/gemm_wmma_tiled_pipeline_schedule_perf"
done
```

The existing Makefile NSYS/NCU targets include the new implementation. For a
large random-input NCU capture, compile a perf target with `-lineinfo`, then
use `--profile-from-start off --launch-count 1` and pass `--profile` to that
executable. The benchmark brackets only one warmed launch. Counter permissions
must already be available to the profiler process.

Ignored artifacts in `build/sm_80/schedule123-v1/` contain frozen sources,
configuration JSON, compiler/resource logs, ordinary CSV results, SASS counts,
NCU reports, and correctness/sanitizer logs. The source files and
commands above do not depend on those artifacts.
