# GEMM hardware-counter analysis

The tiled WMMA implementation improves reuse and reduces instruction count, but
still spends much of its time feeding Tensor Cores. The strongest next experiments
are wider/asynchronous global-to-shared copies and a shared-memory layout that
matches fragment loads. Raising occupancy alone is not the objective: cuBLAS has
the lowest achieved occupancy here and is by far the fastest implementation.

## Measurement scope

Collected on 2026-09-13: A800 80GB PCIe, physical GPU 3, 108 SMs, driver
535.247.01, CUDA 12.6, Nsight Compute 2024.3.0. All five implementations passed the
CPU-reference correctness check for M=N=K=1024, alpha=1, beta=0, FP16 inputs/output
and FP32 accumulation. Binaries used `-O3 -lineinfo -std=c++14 -arch=sm_80`.

The capture used `--set full --launch-skip 7 --launch-count 1`, kernel replay,
`--cache-control all`, and `--clock-control none`. The eighth launch is the first
benchmark call after two correctness calls and five warmups. Each report has one
captured launch, with 47–48 replay passes for metric collection. Counters were
accessed through temporary sudo privileges; driver permissions were not changed.

The original reports are in `build/sm_80/ncu/capture-zXeoo7/`; copies and the current
Makefile exports are in `build/sm_80/ncu/`. These generated artifacts are ignored by
Git. `source_hotspots.json` and `instruction_counts.json` in the latter directory
record additional inspection through the installed NCU Python Report Interface.

## Execution and resource use

| Implementation | NCU duration (us) | Registers/thread | Achieved occupancy (%) | Theoretical occupancy (%) | Tensor active (%) |
|---|---:|---:|---:|---:|---:|
| gemm | 815.552 | 32 | 91.85 | 100.00 | 0.00 |
| gemm_tile | 436.096 | 29 | 92.27 | 100.00 | 0.06 |
| gemm_wmma | 390.528 | 48 | 34.66 | 50.00 | 1.83 |
| gemm_wmma_tiled | 166.752 | 96 | 14.85 | 31.25 | 4.21 |
| gemm_cublas | 17.248 | 196 | 6.18 | 12.50 | 58.74 |

Within this capture, shared-memory tiling improves scalar GEMM by 1.87x, tiled
WMMA improves original WMMA by 2.34x, and cuBLAS improves tiled WMMA by 9.67x.
These are ratios of single captured launches, not statistically established
speedups across shapes or runs.

Tensor active is
`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`; achieved
occupancy is `sm__warps_active.avg.pct_of_peak_sustained_active`. Neither is a
percentage of theoretical GEMM FLOP/s. The scalar variants contain no HMMA in
their captured instruction streams; the tiny nonzero Tensor counter for
`gemm_tile` should not be interpreted as evidence that it uses Tensor Cores.

## What changes between implementations

### Scalar GEMM: too many narrow memory instructions

`gemm` executes 67,108,864 warp-level `LDG.E.U16.CONSTANT` instructions and the
same number of `HADD2.F32` conversion instructions. The total dynamic warp-level
instruction count is 249,724,928. Its leading stall is LG throttle: 16.18 cycles
per issued warp instruction out of 28.86 total, approximately 56%.

PC sampling places the largest LG-throttle hotspots on the 16-bit loads in
`src/gemm.cu:14`. This supports a diagnosis of pressure from frequent memory
instructions. L1 and L2 hit rates are 92.01% and 98.11%, while DRAM throughput is
only 0.27% of peak: high cache hit rates do not remove the cost of issuing and
servicing all those loads. Calling this a saturated-DRAM-bandwidth kernel would
be inconsistent with the counters.

### Shared-memory scalar GEMM: reuse removes most global loads

`gemm_tile` reduces warp-level global-load instructions to 4,194,304, a 16x
reduction. That agrees with the 16x16 shared tile's reuse structure. Duration
drops from 815.55 to 436.10 us while occupancy remains around 92%.

It still executes 204,570,624 instructions, including 71,303,168 `HADD2.F32`
conversions. Barrier stalls average 3.95 cycles per issued instruction, long
scoreboard 3.01, and MIO throttle 2.92. Shared loads, conversion work and
synchronization have replaced much of the previous global-load issue pressure.
The scalar arithmetic path still does not use Tensor Core MMA instructions.

### Original WMMA: matrix instructions surrounded by scalar work

Original WMMA executes 524,288 `HMMA.16816.F32` instructions, but 77,316,096
instructions in total. Its global-to-shared staging still uses 4,194,304 scalar
16-bit global loads followed by scalar stores and conversion/address work.

Long-scoreboard stalls average 10.39 of 16.45 cycles per issued instruction,
approximately 63%. The strongest sampled consumers are half-to-float conversion
instructions waiting on global loads. Separately, shared-load bank-conflict
counters report 2,097,274 conflicts. Source-correlated excessive wavefronts are
concentrated on two fragment-load instructions:

```text
LDSM.16.M88.4     1,048,576 excessive wavefronts
LDSM.16.MT88.4    1,048,576 excessive wavefronts
```

These correspond to the `load_matrix_sync` paths for A and B; the transposed
load maps directly to `src/gemm_wmma.cu:39`. Shared-memory conflicts and waiting
on global-load results are distinct observations, not interchangeable stall
definitions. NVIDIA defines long scoreboard as waiting on a L1TEX operation;
the producing instruction must be inspected to identify the dependency.
[Metric definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference)

With one warp per block, the 32-block/SM architectural limit caps theoretical
occupancy at 32 active warps out of 64 (50%), despite modest register use.

### Tiled WMMA: better reuse, but a synchronous staging path remains

The 64x64 block and 32x32 warp output reduce global-load instructions to
1,048,576 and total instructions to 29,062,144. The HMMA count remains 524,288:
the implementation performs the same core matrix work with less surrounding
work. Long-scoreboard cycles fall to 4.50 per issued instruction, but still
account for about 55% of the 8.15-cycle issue interval.

The highest sampled long-scoreboard consumers are `STS.U16` at the A/B staging
assignments in `src/gemm_wmma_tiled.cu:44` and `:49`. These stores depend on
the preceding global loads. The source also confirms that staging and MMA run
in successive phases, with no asynchronous-copy pipeline or double buffering.

The 16-half row padding has not eliminated fragment-load conflicts. Hardware
counters report 1,048,576 shared-load bank conflicts. Eight LDSM instructions
each have 131,072 source-correlated excessive wavefronts. The aggregate
shared-load conflict count is lower than original WMMA, but the corresponding
conflict/total-wavefront ratios are similar: approximately 49% in both. Different
work counts prevent interpreting the halved total as a halved conflict rate.
Also, excessive wavefronts and hardware bank-conflict counters are different
metrics and should not be substituted for each other.
[Shared-memory table definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#shared-memory)

Theoretical residency is five blocks/SM, limited by registers, but this shape
launches only 256 blocks across 108 SMs, about 2.37 blocks/SM. Two to three
128-thread blocks correspond to 12.5–18.75% warp occupancy, consistent with the
measured 14.85%. The short grid helps explain the gap from the 31.25% theoretical
ceiling; it is not solely a register-allocation problem.

### cuBLAS: much more matrix work per issued instruction

The chosen kernel is named
`sm80_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize96x128x32_stage4_warpsize2x2x1_tensor16x8x16_kernel`.
Its instruction stream contains `LDGSTS.E.BYPASS.LTC128B.128`, `LDGDEPBAR`, LDSM
and HMMA. This is direct evidence of a 128-bit global-to-shared transfer path,
unlike the custom kernels' scalar load/store staging. The name also identifies
a four-stage variant, although the complete scheduling algorithm was not
reconstructed from its closed-source implementation.

| Implementation | Total dynamic warp instructions | HMMA instructions |
|---|---:|---:|
| gemm | 249,724,928 | 0 |
| gemm_tile | 204,570,624 | 0 |
| gemm_wmma | 77,316,096 | 524,288 |
| gemm_wmma_tiled | 29,062,144 | 524,288 |
| gemm_cublas | 1,701,216 | 540,672 |

cuBLAS executes about 17x fewer total instructions than tiled WMMA while its
HMMA count is slightly larger, consistent with extra work on its padded output
tiles. Long-scoreboard cycles are only 0.40 per issued instruction, Tensor
active is 58.74%, and measured shared-load/store bank-conflict counts are zero.
Together these support better data delivery and less scalar overhead, without
assigning an exact fraction of the speedup to any single technique.

Its grid has 88 blocks, fewer than 108 SMs, with four warps per block. A single
resident block on a participating SM is 4/64 = 6.25% occupancy, close to the
measured 6.18%. Occupancy over active cycles does not fully describe unused SMs,
and does not measure useful matrix work per instruction. More residency is not
automatically worth sacrificing this kernel's reuse or instruction efficiency.

## Next experiments, in order

1. For tiled WMMA's aligned path, replace scalar 16-bit staging with wider
   transfers and test an asynchronous-copy/double-buffered pipeline. Verify
   generated SASS and long-scoreboard changes. Ampere supports asynchronous
   global-to-shared copies that can overlap compute and avoid an intermediate
   register transfer. Preserve alignment checks and tail handling.
   [Ampere asynchronous copies](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#asynchronous-data-copy-from-global-memory-to-shared-memory)
2. Redesign/test the shared layout against both A and B fragment accesses,
   including a swizzle or a different tile orientation. Track per-PC ideal and
   actual wavefronts rather than assuming padding alone solved the issue.
3. Revisit tile sizes and small/irregular-shape dispatch after reducing data-path
   overhead. Record instruction counts, duration and occupancy together; do not
   optimize occupancy as an isolated score.

These are proposed experiments, not implemented or measured improvements. Each
change still needs numerical/tail checks and an unprofiled timing comparison.

## Limits and validation

- Cache flushing and NCU's timing mechanism differ from the earlier warm-cache
  NSYS capture. In particular, 390.53 vs roughly 264 us for WMMA is not evidence
  of a code regression. Replay can affect CUDA-event timings printed by the
  application; this analysis uses `gpu__time_duration.sum`.
  [NCU workload duration and cache control](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#workload-durations)
- Clocks were deliberately not locked. The results represent one launch per
  implementation and are not a stability study or a universal shape ranking.
- PM Sampling reports dropped samples: 28, 11, 9, 2 and 2, respectively. Its
  time series is incomplete. The PC stall samples used above are a separate
  sampling facility, but their counts are also samples, not exact stall times.
- cuBLAS source import is unavailable; its metrics and SASS are available.
  Automatic rule `Est. Speedup` values are estimates, not achieved improvements.
- The new Makefile workflow was exercised through sudo on all five kernels at
  128 cubed with `NCU_SET=basic`, including missing-metric handling. Full exports
  were checked against the original 1024-cubed reports. `make check` passed on
  A800 with CUDA 12.6; no kernel source was changed for this integration.
