# Triton versus CUDA GEMM on A800

Measured on 2026-09-19 using physical GPU 2, NVIDIA A800 80GB PCIe (108 SMs),
driver 535.247.01. The CUDA implementation is
`src/gemm_wmma_tiled_pipeline_schedule.cu`; Triton is `src/gemm.triton.py`.
Neither operator was modified for this measurement.

GPU 2 had 0% sampled utilization and 6342 MiB free before the run. A resident
VLLM service still occupied memory. During the main and profiling runs, process
sampling reported SM activity for the benchmark process and no SM activity
value for the resident service. This was a relatively idle shared device,
not an exclusively reserved GPU. No clocks, driver settings or other processes
were changed. Clocks varied under load; small percentage differences are not
evidence of a stable win.

Environment: PyTorch 2.5.1+cu121, Triton 3.1.0, native compilation with Toolkit
12.6.20. The Python process actually resolved CUDA runtime 12.1 and cuBLAS
12.1.3. These numbers should not be combined with the earlier standalone C++
measurements using a different cuBLAS environment.

## Warmed API timings

All three implementations use identical row-major FP16 A/B/C buffers, FP32
accumulation, and alpha=1/beta=0. JIT, autotuning, allocation, transfers,
reference generation and validation are outside timing. CUDA events include
Python/ctypes submission gaps between launches. Results are the mean of four
rounds, each taking the median of five batches of 100 calls; consecutive rounds
reverse implementation order. All twelve shapes passed full output and guard
checks, with zero observed difference from the cuBLAS FP32-reduction reference.

Times are microseconds. Positive differences mean Triton takes longer than CUDA.

| M,N,K | Triton | CUDA schedule | cuBLAS | Triton vs CUDA |
|---|---:|---:|---:|---:|
| 1024 cubed | 24.681 | 18.304 | 15.291 | +34.8% |
| 1536 cubed | 47.276 | 43.310 | 38.172 | +9.2% |
| 2048 cubed | 101.473 | 102.868 | 82.516 | -1.4% |
| 2304 cubed | 148.321 | 123.233 | 117.903 | +20.4% |
| 3072 cubed | 356.726 | 307.704 | 293.187 | +15.9% |
| 4096 cubed | 854.497 | 713.659 | 637.673 | +19.7% |
| 6144 cubed | 2970.150 | 2590.971 | 2103.260 | +14.6% |
| 8192 cubed | 7052.142 | 6470.897 | 4913.912 | +9.0% |
| 2048,4096,1024 | 117.676 | 99.215 | 94.653 | +18.6% |
| 4096,2048,1024 | 116.019 | 99.876 | 92.288 | +16.2% |
| 2048,2048,128 | 24.435 | 15.808 | 15.603 | +54.6% |
| 384,6144,96 | 24.253 | 10.299 | 10.834 | +135.5% |

The 2048-cubed Triton round medians varied from 93.19 to 105.26 us in the main
run, so its apparent 1.4% advantage is not a stable win. A separate four-round
repeat measured Triton 102.986 us, CUDA 103.160 us, and cuBLAS 82.693 us,
again effectively tied between the custom API paths. The main run's per-round
values and selected Triton tile/warp/stage configurations are retained in CSV.

The main process peaked at 884 MiB of PyTorch allocations and 1310 MiB reserved.
Device-wide free memory remained at least 4518 MiB. The before/after samples
both showed 0% utilization and the original 6342 MiB free. SM-clock samples
across preparation, tuning and timing ranged from 870 to 1410 MHz; that range
cannot be assigned to individual GEMM measurements from the coarse monitor.

## GPU kernel cross-check

Nsight Systems 2024.4.2 captured CUDA/NVTX activity, without CPU sampling or
context-switch tracing. This separate pass froze each shape's Triton
configuration to the API run's choice and launched the underlying JIT kernel.
It validated all outputs and guards again, then collected two reversed-order
rounds of 100 warmed calls per implementation. NVTX range names identify the
shape, implementation and round. Total kernel duration inside each labeled
range is divided by 100; both rounds are averaged. Unlabeled setup, validation
and warmup kernels are excluded.

This removes Python submission gaps from the reported duration. It is a
separate profiled experiment, with different repetition counts and unlocked
clocks; subtracting these values from API timings is not a precise estimate of
CPU overhead.

| M,N,K | Triton kernel (us) | CUDA kernel (us) | cuBLAS kernel (us) | Triton vs CUDA |
|---|---:|---:|---:|---:|
| 1024 cubed | 19.823 | 17.676 | 14.748 | +12.1% |
| 1536 cubed | 42.999 | 38.764 | 34.046 | +10.9% |
| 2048 cubed | 99.655 | 94.099 | 77.146 | +5.9% |
| 2304 cubed | 143.220 | 115.665 | 109.721 | +23.8% |
| 3072 cubed | 341.086 | 310.176 | 295.227 | +10.0% |
| 4096 cubed | 849.664 | 702.498 | 647.441 | +20.9% |
| 6144 cubed | 2949.659 | 2582.120 | 2139.339 | +14.2% |
| 8192 cubed | 6979.890 | 6427.355 | 4877.279 | +8.6% |
| 2048,4096,1024 | 118.866 | 95.938 | 88.983 | +23.9% |
| 4096,2048,1024 | 113.430 | 92.323 | 84.674 | +22.9% |
| 2048,2048,128 | 16.415 | 18.802 | 18.090 | -12.7% |
| 384,6144,96 | 9.586 | 10.588 | 11.137 | -9.5% |

The large square results support a real kernel gap, rather than only a Python
launch-cost difference. Short-K results need the two measurement scopes kept
separate: Triton's kernel was faster in this cross-check, while its API timing
was slower. The API path's roughly 24-us floor makes short operations especially
sensitive to submission overhead. These results concern this particular Triton
implementation and its seven candidates, not the best achievable Triton GEMM.

## Reproduction and artifacts

Activate the CUDA-enabled Python environment, select a relatively idle GPU, then:

```sh
make bench-gemm-triton-compare TRITON_BENCH_ARGS="--rounds 4"
```

The main run additionally capped this process's PyTorch allocator to 5% of total
device memory (about 4 GiB); its actual peak was well below that limit. This is
a process-local allocator setting, not a device configuration change.

Ignored artifacts in `build/sm_80/triton-perf-20260919/` include:

- `api.csv`, `api.json`, `api.log`: four-round API measurements and versions.
- `recheck-2048.csv` / `.json` / `.log`: independent 2048-cubed repeat.
- `kernel-comparison.csv`, `kernels_nvtx_kern_sum.csv`, `kernels.nsys-rep`:
  per-shape kernel comparison and original profiler data.
- `profile_compare.py`, `profile-command.json`, `profile-validation.json`:
  exact frozen-config profiling harness, command and correctness results.
- `gpu2-monitor.csv`, `gpu2-processes.txt`, their `profile-` counterparts,
  `before.csv`, `after.csv`, `run.json`, `sources.json`: load, resource usage,
  invocation and source hashes.
