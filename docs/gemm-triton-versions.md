# Triton GEMM v1 / v2 / v3 versus CUDA and cuBLAS

Measured on 2026-09-19 on physical GPU 2, NVIDIA A800 80GB PCIe (108 SMs),
driver 535.247.01. CUDA is `gemm_wmma_tiled_pipeline_schedule.cu`; cuBLAS uses
`gemm_cublas.cu`. The three Triton sources and both CUDA operators were left
unchanged. The shared test and benchmark runners now accept all three versions.

V2 and V3 materially improve large matrices over V1. V3's warmed API latency is
9.9% lower than CUDA at 6144 cubed and 17.3% lower at 8192 cubed; it still takes
10.8% and 10.2% longer than cuBLAS, respectively. V2 and V3 are effectively tied
on these large shapes. The separate kernel-duration experiment supports the
same large-matrix conclusion. Small differences and 4096-cubed results remain
sensitive to measurement order and unlocked clocks.

## Environment and measurement scope

All three Triton versions were rerun with **Triton 3.3.1**, using PyTorch
2.5.1+cu121. V2's `do_bench` autotune argument and benchmark `quantiles` argument
are unavailable in the installed Triton 3.1.0. Triton 3.3.1 was installed only
in ignored `build/triton-3.3.1/` and selected with `PYTHONPATH`; no system,
driver, or existing Python environment was changed. V3 uses the older graph
autotune interface, also available in 3.1. See the upstream
[3.3.1 autotuner](https://github.com/triton-lang/triton/blob/v3.3.1/python/triton/runtime/autotuner.py)
and [benchmark implementation](https://github.com/triton-lang/triton/blob/v3.3.1/python/triton/testing.py).

Native adapters were compiled with CUDA Toolkit 12.6.20. The Python process
resolved CUDA runtime 12.1 and cuBLAS 12.1.3. The earlier
[v1 measurements](gemm-triton-performance.md) used Triton 3.1.0; use the v1
column below for this comparison rather than mixing compiler versions.

GPU 2 had 0% sampled compute utilization and 6342 MiB free before and after the
API run. A resident VLLM service occupied memory. Process sampling showed no
reported SM activity for that service during either measurement pass; this was
a relatively idle shared GPU, not an exclusive reservation. The API process
peaked at 896 MiB of PyTorch allocations and 1010 MiB reserved; device-wide
free memory stayed at least 4856 MiB. The process-local allocator cap was 5%
of device memory. No other processes or clocks were changed. SM clock samples
ranged from 645 to 1410 MHz across preparation, tuning and measurement; these
coarse samples cannot establish the clock for each individual GEMM.

All implementations use identical contiguous row-major FP16 A/B/C buffers,
FP32 accumulation, alpha=1 and beta=0. All twelve shapes passed full output
and guard checks against cuBLAS before timing, with zero observed output
difference on these inputs. Independent CPU-reference checks are recorded below.

## Warmed API timings

Times are microseconds. Each result is the mean of four round medians; each
round takes the median of five CUDA-event batches of 100 calls. Consecutive
rounds reverse implementation order, with five warmup calls before the rounds.
JIT, autotuning, allocations, transfers and correctness checks are excluded.
CUDA-event batches include Python/ctypes submission gaps. Positive percentages
mean v3 takes longer than the named baseline.

| M,N,K | Triton v1 | Triton v2 | Triton v3 | CUDA schedule | cuBLAS | v3 vs CUDA | v3 vs cuBLAS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1024 cubed | 33.974 | 24.266 | 24.509 | 18.371 | 15.296 | +33.4% | +60.2% |
| 1536 cubed | 50.739 | 48.812 | 50.212 | 43.589 | 38.075 | +15.2% | +31.9% |
| 2048 cubed | 113.004 | 105.841 | 109.617 | 104.530 | 84.572 | +4.9% | +29.6% |
| 2304 cubed | 153.267 | 136.852 | 138.360 | 123.392 | 119.788 | +12.1% | +15.5% |
| 3072 cubed | 376.266 | 319.549 | 321.510 | 312.387 | 303.388 | +2.9% | +6.0% |
| 4096 cubed | 890.278 | 713.879 | 713.434 | 721.085 | 656.248 | -1.1% | +8.7% |
| 6144 cubed | 3492.052 | 2371.589 | 2374.871 | 2634.837 | 2143.652 | -9.9% | +10.8% |
| 8192 cubed | 8356.421 | 5472.133 | 5484.969 | 6632.474 | 4978.570 | -17.3% | +10.2% |
| 2048,4096,1024 | 117.553 | 103.552 | 107.802 | 99.845 | 94.318 | +8.0% | +14.3% |
| 4096,2048,1024 | 116.593 | 102.528 | 107.295 | 99.635 | 91.942 | +7.7% | +16.7% |
| 2048,2048,128 | 34.668 | 24.305 | 24.210 | 15.828 | 15.649 | +53.0% | +54.7% |
| 384,6144,96 | 33.651 | 23.997 | 24.179 | 10.291 | 10.785 | +135.0% | +124.2% |

V2 round-median ranges reached 10-15% of the mean at 1536/2048/2304 cubed and
the two rectangular shapes. The 1-5% v2/v3 differences there do not establish
a stable ranking. At 4096 cubed all three custom API paths are close, but the
separate kernel pass does not establish a stable tie with CUDA.

## GPU kernel duration cross-check

Nsight Systems captured CUDA/NVTX activity without CPU sampling or context
switch tracing. This separate pass froze each Triton version's tile choice
from the API run and invoked its raw JIT kernel. Outputs and guards passed
again. For each shape and implementation, two reversed-order rounds each
launched 100 calls. Summed kernel duration inside each NVTX range is divided
by 100, then averaged across the two rounds. Unlabeled setup, validation and
warmup work is excluded. Times are microseconds.

| M,N,K | Triton v1 | Triton v2 | Triton v3 | CUDA schedule | cuBLAS | v3 vs CUDA | v3 vs cuBLAS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1024 cubed | 19.863 | 19.335 | 19.298 | 17.668 | 14.590 | +9.2% | +32.3% |
| 1536 cubed | 49.208 | 46.058 | 44.461 | 38.684 | 33.887 | +14.9% | +31.2% |
| 2048 cubed | 104.132 | 104.032 | 102.771 | 97.796 | 78.402 | +5.1% | +31.1% |
| 2304 cubed | 145.722 | 129.191 | 132.147 | 117.124 | 111.566 | +12.8% | +18.4% |
| 3072 cubed | 349.072 | 316.659 | 340.309 | 302.528 | 288.010 | +12.5% | +18.2% |
| 4096 cubed | 850.174 | 692.952 | 732.645 | 674.201 | 641.367 | +8.7% | +14.2% |
| 6144 cubed | 3291.293 | 2318.705 | 2332.224 | 2584.586 | 2074.401 | -9.8% | +12.4% |
| 8192 cubed | 8086.780 | 5254.063 | 5292.919 | 6323.810 | 4818.613 | -16.3% | +9.8% |
| 2048,4096,1024 | 111.774 | 101.011 | 98.987 | 93.173 | 86.819 | +6.2% | +14.0% |
| 4096,2048,1024 | 110.396 | 100.010 | 98.755 | 93.985 | 85.319 | +5.1% | +15.7% |
| 2048,2048,128 | 14.059 | 14.087 | 14.085 | 16.965 | 16.518 | -17.0% | -14.7% |
| 384,6144,96 | 8.464 | 8.299 | 8.341 | 9.808 | 10.357 | -15.0% | -19.5% |

At 6144/8192 cubed, the improvement over CUDA also appears in GPU execution
time. For short K, v3's GPU kernel takes about 14.09/8.34 us, less than CUDA's
16.97/9.81 us, while the API path stays near 24 us. This supports prioritizing
submission overhead for short operations, for example by evaluating batched
CUDA Graph replay in a separate benchmark. Such graph timings are not measured
here and must not be mixed into the API table.

The 3072/4096-cubed profiled results are order-sensitive. For example, v2's
4096-cubed round means were 665.09 and 720.81 us, while v3 was 726.91 and
738.38 us despite choosing the same configuration and sharing the same kernel
body. These passes use unlocked clocks and different repetition/statistical
rules. They do not prove v3 regressed there, nor can their difference from API
time be subtracted as an exact CPU-overhead estimate.

## What changed in v2 and v3

Compared with v1, both new versions use grouped one-dimensional CTA ordering,
32-bit element offsets when bounds permit, compile-time alpha=1/beta=0 paths,
and unmasked loads/stores for full tiles. They tune into scratch output, retain
original C, and cache the selected JIT launcher for subsequent calls. Their
candidate sets include 128x256 and 256x128 tiles, giving autotuning larger
reuse choices than v1. The measurements compare these changes together and do
not attribute the speedup to any one change in isolation.

The v2 and v3 `_gemm` function bodies are identical. V3 changes the tuning API,
synchronizes input production before first-use side-stream tuning, and adds
128x128x32 / 4 warps / 3 stages with GROUP_M=1 and 8 (34 candidates versus 32).
That added choice was selected for 2304 cubed and both rectangular shapes;
it did not produce a consistent measured advantage over v2. First-use tuning
and synchronization costs are excluded from the warmed timings.

| M,N,K | v2: BM/BN/BK, warps, stages, GROUP_M | v3: BM/BN/BK, warps, stages, GROUP_M |
|---|---|---|
| 1024 cubed | 64/64/64, 4, 3, 1 | 64/64/64, 4, 3, 1 |
| 1536 cubed | 64/128/32, 4, 3, 1 | 64/128/32, 4, 3, 8 |
| 2048 cubed | 128/64/64, 4, 4, 8 | 128/64/64, 4, 4, 1 |
| 2304 cubed | 128/128/32, 4, 4, 1 | 128/128/32, 4, 3, 1 |
| 3072 cubed | 256/128/64, 8, 3, 1 | 128/256/64, 8, 3, 8 |
| 4096 cubed | 256/128/64, 8, 3, 8 | 256/128/64, 8, 3, 8 |
| 6144 cubed | 128/256/64, 8, 3, 8 | 128/256/64, 8, 3, 8 |
| 8192 cubed | 256/128/64, 8, 3, 8 | 256/128/64, 8, 3, 8 |
| 2048,4096,1024 | 128/128/32, 4, 4, 1 | 128/128/32, 4, 3, 1 |
| 4096,2048,1024 | 128/128/32, 4, 4, 1 | 128/128/32, 4, 3, 1 |
| 2048,2048,128 | 128/64/32, 4, 3, 8 | 128/64/32, 4, 3, 8 |
| 384,6144,96 | 64/128/32, 4, 3, 1 | 64/128/32, 4, 3, 1 |

All selected v2/v3 configurations used ALPHA_ONE=true, BETA_ZERO=true and
USE_I64=false. Full metadata, including v1 choices, is in the raw CSV.

## Validation and reproduction

- `make check`: the existing native CUDA suite passed.
- Triton 3.3.1: v2 passed 281 checks (57 public cases + 32 candidates x 7
  representative cases); v3 passed 295 (57 + 34 x 7). CPU double reference,
  two calls per case, input preservation and output guards were checked.
- All v2/v3 candidates passed memcheck, racecheck and synccheck on tails,
  unaligned A/B and beta=0 / NaN C / K=0: 96 and 102 checks per tool,
  respectively, with zero errors or race warnings.
- The original Triton 3.1.0 environment also passed a v3 compatibility subset:
  four public cases plus all 34 candidates on two tail/alignment cases (72
  checks). A v1 regression subset passed nine checks after the runner refactor.
  These are compatibility checks, not a second performance comparison.
- All five implementations passed full-output comparison on the twelve
  performance shapes both before API timing and in the frozen-config profile.

The sanitizer uses `--all-configs --fixed-only` to instrument every candidate
without repeating the autotuner's internal benchmarks. Ordinary correctness
checks retain the real public-solve tuning path. The tested dimensions use the
32-bit offset path; extremely large 64-bit-offset problems are not validated
by this run.

Activate a Python environment with CUDA PyTorch and Triton 3.3.1, and select
a relatively idle GPU. If using the project-local package from this run:

```sh
export PYTHONPATH="$PWD/build/triton-3.3.1${PYTHONPATH:+:$PYTHONPATH}"
make check-gemm-triton TRITON_SOURCE=src/gemm.triton.v3.py
make sanitize-gemm-triton TRITON_SOURCE=src/gemm.triton.v3.py
make bench-gemm-triton-compare \
  TRITON_SOURCE=src/gemm.triton.v2.py \
  TRITON_BENCH_ARGS="--reference-source src/gemm.triton.py --reference-source src/gemm.triton.v3.py --rounds 4"
```

Set `PYTHON` to the intended interpreter and `CUDA_VISIBLE_DEVICES` to the
selected physical device. The exact measured run also applied the process-local
allocator cap. CSV implementation names map as follows: `triton` = v2,
`triton_reference` = v1, `triton_reference_2` = v3. The source-path mapping is
recorded in the sibling JSON. The default Makefile selection remains v1.

Ignored artifacts are under `build/sm_80/triton-v2-perf-20260919/` (the directory
name was kept when v3 was added):

- `api.csv`, `api.json`, `api.log`, `run.json`: API timings, versions and command.
- `kernel-comparison.csv`, `kernels_nvtx_kern_sum.csv`, `kernels.nsys-rep`:
  kernel timings and profiler report.
- `profile_compare.py`, `profile-command.json`, `profile-validation.json`:
  frozen-config harness, invocation and validation.
- `check-v2.log`, `check-v3.log`, `make-check.log`, `sanitize-v2.log`,
  `sanitize-v3.log`, `compat-v3-triton31.log`, `compat-v1-triton31.log`,
  `validation.json`: validation logs and summary.
- GPU/process monitors, before/after snapshots and source hashes preserve the
  measurement conditions. Generated data is not committed.
