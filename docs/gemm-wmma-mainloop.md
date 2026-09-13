# Reducing GEMM mainloop overhead

`src/gemm_wmma_tiled_pipeline_mainloop.cu` is an independent successor to
`gemm_wmma_tiled_pipeline_multistage.cu` at commit `b940f4b`. It selects BK=64
for small, complete output grids. At 1024 cubed this halves the K-loop iterations
and cuts executed warp instructions by 35.3%, reducing warmed event time by 9.9%.
The previous source remains a separate benchmark.

## Selected implementation

The existing XOR shared layout, three input stages, register operand buffering,
`ldmatrix`/`mma.sync`, FP32 accumulation, and packed output stores are retained.
BK=64 executes four K=16 operand groups per input chunk instead of two. This
amortizes loop control, pointer updates, commits, and block barriers over twice
as much computation. It does not reduce the logical A/B bytes needed by a block.

The contract remains contiguous row-major FP16 A[M,K], B[K,N], C[M,N], with
`C = alpha * A * B + beta * C`. M/N=0 is a no-op, K=0 scales C, and beta=0
does not read C. A/B require 16-byte alignment for the native path; C only
requires half alignment. Tails and unaligned A/B use the fixed generic WMMA
pipeline. The base aligned check still accepts K divisible by 32, so K=32/96
and other nonmultiples of 64 retain the previous native path.

After the base M/N/K divisibility check of 64/64/32, dispatch selects the first
matching row. Every native tile uses 128 threads and three input stages.

| Condition | Block / warp | BK | Registers/thread | Shared/block |
|---|---|---:|---:|---:|
| M/N divisible by 128, K >= 2048, at least 1024 blocks of 128x128 | 128x128 / 64x64 | 32 | 236 | 48 KiB |
| N divisible by 128, K >= 512, at least 256 blocks of 64x128 | 64x128 / 32x64 | 32 | 166 | 36 KiB |
| K divisible by 64, K >= 192, at most 324 blocks of 64x64 | 64x64 / 32x32 | 64 | 128 | 48 KiB |
| Remaining complete/aligned inputs | 64x64 / 32x32 | 32 | 96 | 24 KiB |

The new tile doubles shared storage. On this 108-SM A800 it permits three
resident blocks per SM; 324 blocks fit without an additional wave. Larger
grids can lose despite fewer instructions. K >= 192 provides at least three
BK=64 chunks, including work beyond the two-chunk input prologue. These are
measured A800 thresholds, not portable autotuning.

The BK=64 kernel uses `__launch_bounds__(128, 4)` to allow more registers than
the BK=32 small tile's five-block compiler target. Shared memory still limits
BK=64 residency to three blocks per SM: launch bounds do not promise actual
occupancy. All five final kernels, including the generic fallback, have zero
stack/local allocation. Their SASS encodings match the selected BK=64 candidate
or the corresponding previous kernel; ordinary and line-info builds also match.

The previous compile-time controls are retained: explicitly setting any of
`GEMM_BM/BN/BK/WM/WN/STAGES` disables automatic dispatch by default.
`GEMM_MULTISTAGE_MIN_BLOCKS` overrides the compiler target, and
`GEMM_REGISTER_PIPELINE=0` disables register operand prefetch. Static shared
storage remains limited to 48 KiB. Cross-chunk prefetch and stage-unrolling
experiment code is not included in the selected source.

## Measurements

Measured on 2026-09-13 with NVIDIA A800 80GB PCIe (108 SMs), CUDA 12.6.20,
driver 535.247.01, and `-O3 -std=c++14 -arch=sm_80 -Xcompiler -Wall`.
NCU builds add `-lineinfo`. Clocks were not locked; no system or driver settings
were changed. Physical GPU 4 ran ordinary measurements serially; no competing
compute process was observed there. Other GPUs on the machine were in use.

Random FP16 inputs use seed 12345, alpha=1, beta=0. Every output is checked
against cuBLAS with FP32 reductions, along with output guards. Each measurement
warms five calls and takes the median of five CUDA-event batches of 100 calls.
Allocation, transfers, reference generation, and validation are excluded;
host submission gaps are included. The table averages two such measurements,
reversing variant order in the second round. CPU double-reference validation
is separate from this large-matrix timing harness.

| M x N x K | Previous multistage (us) | Mainloop (us) | Time reduction | cuBLAS (us) |
|---|---:|---:|---:|---:|
| 512 x 512 x 512 | 8.172 | 7.557 | 7.5% | 7.240 |
| 768 x 768 x 768 | 13.799 | 12.473 | 9.6% | 10.619 |
| 1024 x 1024 x 1024 | 22.052 | 19.876 | 9.9% | 15.365 |
| 1152 x 1152 x 1152 | 25.718 | 24.822 | 3.5% | 18.114 |
| 1280 x 1280 x 1280 | 33.684 | 33.654 | 0.1% | 22.697 |
| 1408 x 1408 x 1408 | 45.639 | 45.660 | -0.04% | 32.784 |
| 1536 x 1536 x 1536 | 46.326 | 46.341 | -0.03% | 34.611 |
| 2048 x 2048 x 2048 | 96.978 | 100.751 | -3.9% | 74.235 |
| 4096 x 4096 x 4096 | 765.808 | 765.773 | 0.00% | 638.183 |
| 2048 x 2048 x 128 | 16.348 | 16.358 | -0.06% | 15.376 |
| 4096 x 4096 x 128 | 54.543 | 54.502 | 0.08% | 46.382 |
| 2048 x 512 x 512 | 14.438 | 13.128 | 9.1% | 11.187 |
| 4096 x 256 x 1024 | 24.228 | 22.743 | 6.1% | 17.531 |
| 64 x 16384 x 1024 | 27.751 | 26.588 | 4.2% | 27.382 |
| 1009 x 513 x 257 | 27.837 | 27.802 | 0.1% | 19.277 |

The initial 2048-cubed results showed a 3.9% regression despite identical
kernel instructions and resources. Six additional alternating rounds gave
97.044–100.291 us for the previous binary and 97.004–100.321 us for the new
binary; their medians were 98.284 and 97.234 us. A separately linked final
profile binary measured 98.970–101.499 us. NCU differed by 0.6% on this path.
The overlapping ranges do not establish a stable change at 2048 cubed;
the original measurements are retained above rather than discarded.

At 1024 cubed, the selected version takes 1.294x cuBLAS time: 4.511 us, or
29.4%, longer. The previous gap was 43.5%. This optimization reduces that gap
but does not reach parity.

Additional checks exercised both sides of the 324-block cutoff with
1152x1152x192 and 320x4160x192, short K, and aligned K=992 (not divisible by
64). Their outputs passed. The listed BK=64 shapes matched cuBLAS exactly;
the generic tail case had maximum absolute error 0.0078125. K=992 had maximum
absolute error 0.015625, within `atol=0.01, rtol=0.01`.

## Candidate selection

Screening tested BK=32/64, two/three input stages, compiler block targets
3/4/5, and two additional scheduling changes:

- Cross-chunk prefetch moved the wait/barrier before the last MMA of a chunk,
  then loaded the following chunk's first operands into alternate registers.
- Unrolling the three input-ring phases made shared-stage addresses constant.

Selected candidate results below are separate from the final dispatch table.
All use fixed 64x64 blocks, 32x32 warp tiles, and three input stages. Warmed
times average two random-input rounds on GPU 4; NCU ran on GPU 5.

| Candidate at 1024 cubed | Registers | Warmed (us) | NCU (us) | Warp instructions |
|---|---:|---:|---:|---:|
| Previous BK=32 | 96 | 22.068 | 22.688 | 4,153,344 |
| BK=32, cross-chunk prefetch, target 4 blocks | 128 | 20.009 | 20.992 | 3,196,928 |
| BK=64, existing operand loop, target 4 blocks | 128 | 19.825 | 20.416 | 2,686,976 |
| BK=64, cross-chunk prefetch + phase unrolling, target 4 blocks | 109 | 20.004 | 20.832 | 2,970,624 |

Cross-chunk prefetch helped BK=32, but the simpler BK=64 loop won both timing
scopes. The unrolled variant lowered register use and short-scoreboard stalls,
yet did not improve total time. Some more constrained cross-chunk builds
spilled: BK=64 with a five-block target used a 104-byte stack frame and took
about 34.6 us in the all-ones screen. Those variants were not selected.

Forcing BK=64 globally also hurt other shapes. Random-input screening measured
1280 cubed at 43.735 us versus 33.720 us for the previous dispatcher, and
1408 cubed at 52.562 versus 45.578 us. At 2048x2048x128 it took about 20.04 us
versus 16.29 us. These observations motivated the grid and K guards.

## NCU analysis

Nsight Compute 2024.3.0 captured one warmed, validated call per variant/shape
on GPU 5 using `--set full --replay-mode kernel --cache-control all
--clock-control none --import-source yes --profile-from-start off
--launch-count 1`. The harness brackets the selected call with
`cudaProfilerStart/Stop`, excluding its cuBLAS reference call. Temporary
administrator execution enabled counters; system configuration was unchanged.
These cache-flushed replay durations are separate from ordinary warmed timings.

| 1024 cubed metric | Previous | Mainloop | cuBLAS |
|---|---:|---:|---:|
| Kernel duration (us) | 22.656 | 20.480 | 16.928 |
| Executed warp instructions | 4,153,344 | 2,686,976 | 1,701,216 |
| Tensor active (% active cycles) | 39.70 | 42.55 | 60.52 |
| L2 reads (MiB) | 64 | 64 | 38 |
| Achieved occupancy (%) | 14.54 | 14.51 | 6.27 |
| Short scoreboard (cycles/issued instruction) | 1.13 | 0.90 | 0.09 |
| Long scoreboard (cycles/issued instruction) | 0.74 | 2.62 | 0.36 |
| Barrier (cycles/issued instruction) | 0.20 | 0.38 | 0.09 |

The main gain is lower instruction overhead, not additional A/B reuse.
The mainloop still reads 68.4% more from L2 than cuBLAS and executes 57.9%
more warp instructions. Tensor activity remains 42.6% versus 60.5%.
Fewer barriers do not imply a lower barrier-stall ratio: the denominator is
issued instructions, and waiting/scheduling also changes. Long-scoreboard
waiting increases in this capture, so the result does not show that BK=64
improves every dependency stall.

Shared matrix-load wavefronts remain equal to their ideal count (1,048,576).
Async-copy shared wavefronts change from 761,856 actual / 507,904 ideal to
491,520 / 491,520. The layout formula is unchanged, but changing BK changes
the per-thread copy mapping. This is another observed benefit of the selected
configuration, without a reduction in total L2 reads.

For the retained larger paths, executed instructions and L2 read totals are
unchanged. At 2048 cubed, NCU measured 95.200 / 95.744 / 73.120 us for previous /
mainloop / cuBLAS, with 18,792,448 instructions for both custom versions.
At 4096 cubed it measured 673.792 / 672.896 / 552.896 us, with 113,569,792
instructions for both custom versions. The next gains still require better
reuse or fewer operand/control/epilogue instructions; the experiments here
do not justify adding more pipeline stages unconditionally.

## Validation and reproduction

- `make -j2 check` passed, including all ten GEMM versions and all 39 GEMM cases.
- `make -j2 sanitize` passed with 28 zero-error/zero-hazard summaries, including
  memcheck, racecheck, and synccheck on all five pipelined versions.
- Three new CPU double-reference cases cover three/five/seven BK=64 chunks,
  nontrivial alpha/beta, and unaligned C. A forced BK=64 build passed all
  39 cases and all three sanitizer tools, including its one/two-chunk prologues.
- Every screened candidate passed the 39-case CPU suite. Final automatic
  dispatch and its cutoff cases also passed full-output cuBLAS comparisons.
- sm_75 compilation passed. A compute_75-to-sm_80 build passed all 39 cases
  on A800, exercising the synchronous-copy/two-K=8-MMA branch; this is not a
  T4 runtime test.
- Ordinary/profile SASS comparison passed. The new version is included in
  build, run, check, sanitizer, NSYS, and NCU Makefile targets.

```bash
CUDA_VISIBLE_DEVICES=4 make run-gemm-wmma-tiled-pipeline-mainloop GEMM_ARGS='1024 1024 1024 100'
CUDA_VISIBLE_DEVICES=4 make run-gemm-wmma-tiled-pipeline-multistage GEMM_ARGS='1024 1024 1024 100'
CUDA_VISIBLE_DEVICES=4 build/sm_80/gemm_cublas_bench 1024 1024 1024 100

# Force BK=64 independently of the default dispatcher.
make BIN_DIR=build/sm_80/mainloop-bk64 \
  NVCCFLAGS='-O3 -std=c++14 -arch=sm_80 -Xcompiler -Wall -DGEMM_BK=64' \
  build/sm_80/mainloop-bk64/gemm_wmma_tiled_pipeline_mainloop_bench
CUDA_VISIBLE_DEVICES=4 build/sm_80/mainloop-bk64/gemm_wmma_tiled_pipeline_mainloop_bench --check-only

make -j2 ncu-gemm-build NCU_GEMMS='gemm_wmma_tiled_pipeline_multistage gemm_wmma_tiled_pipeline_mainloop gemm_cublas'
# Run profiling in a shell with access to performance counters.
CUDA_VISIBLE_DEVICES=4 make ncu-gemm \
  NCU_GEMMS='gemm_wmma_tiled_pipeline_multistage gemm_wmma_tiled_pipeline_mainloop gemm_cublas' \
  GEMM_ARGS='1024 1024 1024 2' NCU_DIR=build/sm_80/ncu-mainloop
```

Local ignored artifacts are under `build/sm_80/mainloop-v1/`: frozen baseline
and candidate sources, screening configurations/results, `candidate-random.csv`,
`dispatch-screen.csv`, `final-random.csv`, `recheck-2048.csv`, timing/capture
harnesses, build/check/sanitizer logs, resource reports, and SASS hashes.
`ncu-candidates/1024/` contains five candidate reports;
`ncu-final/{1024,2048,4096}/` contains the nine final reports and exported stats.
`ncu-final/comparison_detail.csv` records the counters used above.
