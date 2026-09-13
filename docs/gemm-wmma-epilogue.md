# Compile-time alpha=1, beta=0 epilogue

`src/gemm_wmma_tiled_pipeline_epilogue.cu` is an independent successor to
`gemm_wmma_tiled_pipeline_reuse.cu`. It specializes the output calculation
for exactly alpha=1, beta=0. The previous versions remain available for
comparison. On this A800, measured time decreases by 3.0% at 1024 cubed,
2.2% at 1536 cubed, and 8.4–13.7% on the tested short-K shapes. This is not
a universal speedup: 2304 and 3072 cubed regress by about 3% in the final runs.

## Implementation

The host `solve` checks `alpha == 1.0f && beta == 0.0f` once, then dispatches
`solve_epilogue<true>` or `<false>`. The template argument reaches both the
native MMA kernel and the generic WMMA fallback. The true specialization
converts the FP32 accumulator directly to FP16; its device code does not
multiply by alpha, test beta, or read the previous C. Both +0 and -0 beta
select this path. Other scalar values retain the general epilogue.

The row-major FP16 A[M,K], B[K,N], C[M,N] contract and FP32 accumulation
are unchanged. M/N=0 does no work. K=0 with alpha=1/beta=0 writes zeros,
including when the previous C contains NaNs. Native paths retain the
16-byte-aligned A/B requirement and accept C with only half alignment;
misaligned half2 outputs use scalar stores. Tails and unaligned A/B use
the generic pipeline, with the same alpha/beta specialization.

The [reuse version's tile dispatcher](gemm-wmma-reuse.md#implementation-and-dispatch),
XOR shared layout, three input stages, and register operand buffering are
retained. In the specialized native epilogue:

- Warp tiles with WM < 48 retain the original output traversal and use
  `__floats2half2_rn` to pack adjacent values.
- Warp tiles with WM >= 48 visit all N fragments within each eight-row
  stripe before moving to the next stripe. Each pair uses two scalar
  conversions and bit packing, followed by the existing shuffle/store scheme.

The two loop bodies intentionally remain separate: changing the general
path's traversal also changed its generated mainloop. The final six general
kernels have exactly the same SASS instruction words as the reuse baseline.
Normal, line-info, and Makefile NCU builds match for all twelve kernels.
The final source cleanup was also checked against the measured kernel code.

With CUDA 12.6.20 targeting sm_80, the five specialized native kernels have
no ordinary global-load (`LDG`) instructions and no alpha/beta parameter
references. A/B loading uses `LDGSTS`. The generic fallback still needs
ordinary A/B loads, so the same no-LDG test is not applicable there.
All twelve kernels have zero stack and local allocation.

| Block / warp / BK | General registers/thread | Specialized registers/thread | Shared/block |
|---|---:|---:|---:|
| 64x64 / 32x32 / 32 | 96 | 96 | 24 KiB |
| 64x64 / 32x32 / 64 | 128 | 121 | 48 KiB |
| 64x128 / 32x64 / 32 | 166 | 160 | 36 KiB |
| 96x128 / 48x64 / 32 | 237 | 232 | 42 KiB |
| 128x128 / 64x64 / 32 | 236 | 231 | 48 KiB |
| Generic 64x64 WMMA fallback | 62 | 62 | 26 KiB |

The reuse baseline already skips C loads at runtime when beta=0. This
change removes general-case code at compile time; it does not save an
additional C read relative to that baseline's beta=0 execution. Register
count reductions alone do not establish a speedup or more resident blocks.

## Ordinary measurements

Measured on 2026-09-13 with NVIDIA A800 80GB PCIe, 108 SMs, CUDA 12.6.20,
driver 535.247.01, and `-O3 -std=c++14 -arch=sm_80 -Xcompiler -Wall`.
No system/driver settings were changed and clocks were not locked. Physical
GPU 4 ran the comparisons serially, with no competing compute process
observed there; other GPUs were in use.

The baseline is the reuse source frozen at the start of this experiment.
Inputs are random FP16 values, seed 12345, with alpha=1/beta=0. Outputs
and guards are checked against cuBLAS with FP32 reductions before timing.
Each run warms five calls and reports the median of five CUDA-event batches
of 100 calls. Allocation, transfers, reference generation, and validation
are excluded; host submission gaps are included. Stop-event synchronization
completes each batch before reading its elapsed GPU time. The table averages
two runs, with reversed variant order in the second run.

| M x N x K | Reuse (us) | Epilogue (us) | Time reduction | cuBLAS (us) |
|---|---:|---:|---:|---:|
| 128 x 128 x 64 | 4.551 | 3.927 | 13.7% | 4.874 |
| 512 x 512 x 512 | 7.562 | 7.188 | 5.0% | 7.562 |
| 768 x 768 x 768 | 12.518 | 11.868 | 5.2% | 10.701 |
| 1024 x 1024 x 1024 | 19.861 | 19.267 | 3.0% | 15.370 |
| 1152 x 1152 x 1152 | 24.898 | 24.294 | 2.4% | 18.145 |
| 1536 x 1536 x 1536 | 39.373 | 38.523 | 2.2% | 34.652 |
| 2048 x 2048 x 2048 | 99.164 | 98.519 | 0.7% | 74.071 |
| 2304 x 2304 x 2304 | 115.947 | 119.327 | -2.9% | 103.055 |
| 3072 x 3072 x 3072 | 320.358 | 329.457 | -2.8% | 297.124 |
| 4096 x 4096 x 4096 | 765.753 | 755.681 | 1.3% | 639.340 |
| 384 x 6144 x 32 | 7.860 | 6.794 | 13.6% | 8.674 |
| 384 x 6144 x 96 | 10.009 | 8.765 | 12.4% | 10.757 |
| 2048 x 2048 x 128 | 16.369 | 14.991 | 8.4% | 15.340 |
| 1009 x 513 x 257 | 27.873 | 27.761 | 0.4% | 19.276 |
| 288 x 512 x 512 | 19.814 | 19.737 | 0.4% | 7.219 |

Time reduction is `(reuse - epilogue) / reuse`; a negative value is slower.
The specialized version still takes 25.3%, 11.2%, 33.0%, and 18.2% longer
than cuBLAS at 1024/1536/2048/4096 cubed. Short-K outputs benefit more because
the output calculation is a larger fraction of their work.

Small changes should not be treated as precise speedups. At 3072 cubed,
the two reuse runs were 309.668 and 331.049 us, while the new version took
326.943 and 331.971 us. At 2304 cubed the new version was slower in both
runs. These results do not justify claiming an improvement for long-K
96x128 tiles. The tail cases show negligible benefit. All listed native
cases matched the cuBLAS reference exactly; the 1009x513x257 fallback had
maximum absolute error 0.0078125, within `atol=0.01, rtol=0.01`.

## Why the output order also changed

The first attempt only made the scalar values compile-time constants.
Although it reduced registers, it increased the executed instruction count
at 1536 cubed from 5,754,624 to 6,506,496 and slowed ordinary timing from
39.434 to 41.313 us. Changing the epilogue can alter register allocation
and instruction scheduling elsewhere in the same kernel.

Five packing/traversal candidates were screened with the same random-input
harness and each passed the 48-case correctness suite. Selected single-run
screen results, separate from the final two-round measurements:

| Variant | 1536 cubed (us) | 2048 cubed (us) | 4096 cubed (us) |
|---|---:|---:|---:|
| Reuse baseline | 39.424 | 96.686 | 765.829 |
| Constant-only epilogue | 41.359 | 98.560 | 770.212 |
| Original traversal, half2 conversion | 41.421 | 96.604 | 780.247 |
| Eight-row stripe traversal, scalar conversion | 38.472 | 106.906 | 755.640 |

The final version selects the original traversal/half2 form for smaller
warp tiles and the stripe traversal/scalar form for larger warp tiles.
This avoids the substantial regressions from applying either form to all
tiles. It does not eliminate the smaller regressions in the final table.

## Nsight Compute

Nsight Compute 2024.3.0 captured twelve reports serially on GPU 5 using
`--set full --replay-mode kernel --cache-control all --clock-control none
--import-source yes --profile-from-start off --launch-count 1`.
After validation and five warmups, the harness brackets one call with
`cudaProfilerStart/Stop`, excluding its cuBLAS reference kernel. Temporary
administrator execution enabled counters without changing system settings.
These cache-flushed replay durations are separate from warmed ordinary timing.

| Shape / metric | Reuse | Epilogue | cuBLAS |
|---|---:|---:|---:|
| 1024 cubed: duration (us) | 20.256 | 19.872 | 16.928 |
| Executed warp instructions | 2,686,976 | 2,444,288 | 1,701,216 |
| Tensor active (% active cycles) | 42.56 | 44.42 | 60.07 |
| L2 reads (MiB) | 64 | 64 | 38 |
| 1536 cubed: duration (us) | 40.128 | 39.168 | 36.192 |
| Executed warp instructions | 5,754,624 | 5,345,280 | 4,905,216 |
| Tensor active (% active cycles) | 66.32 | 68.41 | 79.69 |
| L2 reads (MiB) | 126 | 126 | 90 |
| 2048 cubed: duration (us) | 94.912 | 95.520 | 73.120 |
| Executed warp instructions | 18,792,448 | 19,339,264 | 10,043,904 |
| Tensor active (% active cycles) | 62.85 | 63.06 | 85.42 |
| L2 reads (MiB) | 384 | 384 | 232 |
| 4096 cubed: duration (us) | 673.408 | 666.208 | 551.200 |
| Executed warp instructions | 113,569,792 | 104,337,408 | 61,411,328 |
| Tensor active (% active cycles) | 74.27 | 74.35 | 93.14 |
| L2 reads (MiB) | 2048 | 2048 | 1536 |

Instruction count decreases by 9.0%, 7.1%, and 8.1% at 1024/1536/4096
cubed, but increases by 2.9% at 2048 cubed. The latter has essentially
unchanged ordinary performance and is slightly slower under NCU. L2 read
traffic is identical between custom versions at every captured shape.
This specialization does not improve A/B reuse.

The remaining cuBLAS gap includes both extra input traffic and compute
waiting. For example, at 1536 cubed the specialized kernel's long-scoreboard
stall metric is 0.86 versus cuBLAS's 0.43 cycles per issued instruction;
at 4096 cubed it is 0.55 versus 0.06. Tensor activity remains substantially
lower. These observations support further work on input reuse and mainloop
scheduling; a simpler epilogue alone does not address those costs.

## Validation and reproduction

`make check` passed, including all twelve GEMMs with the expanded 48-case
suite. The six added cases vary alpha and beta independently, poison C with
NaNs for native/tail/K=0/unaligned-output paths, and cover beta=-0. The CPU
reference now ignores old C when beta=0. Existing checks exercise repeated
calls, partial dimensions, input/output alignment, and short pipeline rings.

`make sanitize` passed with 34 zero-error summaries, including memcheck,
racecheck, and synccheck for all seven pipelined versions. An additional
initcheck harness left C uninitialized and verified output/guards for seven
shapes covering all five native tiles, the generic tail path, and K=0;
it reported zero errors. This supplements the NaN tests, which alone only
check output semantics and cannot prove the absence of a discarded load.

The source compiled for sm_75. A `compute_75` front-end / `sm_80` back-end
build passed all 48 cases on A800, exercising synchronous copies and K=8 MMA;
this is not a T4 runtime test. A build with
`-DGEMM_SPECIALIZE_ALPHA_BETA=0` also passed all 48 cases and retained exactly
the baseline's six general kernels.

```bash
CUDA_VISIBLE_DEVICES=4 make run-gemm-wmma-tiled-pipeline-epilogue \
  GEMM_ARGS="1024 1024 1024 100"
CUDA_VISIBLE_DEVICES=6 make check
CUDA_VISIBLE_DEVICES=7 make sanitize

# Disable specialization in a separate build directory for comparison.
CUDA_VISIBLE_DEVICES=4 make run-gemm-wmma-tiled-pipeline-epilogue \
  BIN_DIR=build/sm_80/epilogue-general \
  NVCCFLAGS="-O3 -std=c++14 -arch=sm_80 -Xcompiler -Wall -DGEMM_SPECIALIZE_ALPHA_BETA=0" \
  GEMM_ARGS="1024 1024 1024 100"

# On an account with access to performance counters:
CUDA_VISIBLE_DEVICES=5 make ncu-gemm \
  NCU_GEMMS="gemm_wmma_tiled_pipeline_reuse gemm_wmma_tiled_pipeline_epilogue gemm_cublas" \
  GEMM_ARGS="1536 1536 1536 100" NCU_DIR=build/sm_80/ncu-epilogue-1536
```

The Makefile target uses the shared CPU-reference benchmark. Local random-input
timing/profile harnesses, frozen baseline, compiled candidates, logs, SASS
hashes, and CSV results are under the ignored `build/sm_80/epilogue-v1/`.
Final NCU reports and text/CSV exports are in `ncu-final/`; `ncu-naive/`
preserves the initial constant-only experiment. These artifacts are not
committed to the repository.
