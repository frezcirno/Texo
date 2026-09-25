# Texo

A collection of CUDA operator implementations, correctness tests, and benchmarks.
This repository follows the process of learning GPU programming: start with a
readable implementation, validate it against a CPU reference, then measure changes.

The kernels are standalone examples, not a single linkable library. Most expose an
`extern "C" void solve(...)` entry point and accept device pointers. Build each
operator separately; the Makefile renames entry points when comparing variants.

## Requirements

- Linux, GNU Make, and a C++ compiler supported by the CUDA Toolkit.
- CUDA Toolkit 12.6 is the development baseline. cuBLAS, CUB, and Cooperative Groups come
  with the toolkit; no external C++ dependencies are vendored.
- An NVIDIA GPU and compatible driver to run tests. Compilation alone needs no GPU.
- `compute-sanitizer` for the optional memory and synchronization checks.
- Optional Triton targets require a Python environment with CUDA-enabled PyTorch
  and Triton; select it with `PYTHON=/path/to/python`.

The default target is `sm_80` (A100/A800). Use `sm_75` for T4. Build artifacts are
kept in separate directories for each architecture.

## Quick start

```bash
make -j2                         # Build tests and benchmarks
make check                       # Run the GPU correctness suite
make check-full                  # Include large MSE regression cases
make sanitize                    # Selected Compute Sanitizer checks
```

Override the compiler or architecture when needed:

```bash
make -j2 NVCC=/usr/local/cuda/bin/nvcc NVCC_ARCH=sm_75
make check NVCC_ARCH=sm_75

# Choose an available GPU; CUDA device 0 then refers to physical GPU 3.
CUDA_VISIBLE_DEVICES=3 make check
```

Use `make help` to list targets. The default CUDA path is `/usr/local/cuda`;
`CUDA_HOME`, `NVCC`, `NVCCFLAGS`, and `BIN_DIR` can be overridden. Use `make -B`
after changing compiler flags or toolkit while retaining the same build directory.

## Operators

| Operator | Source | Test / benchmark |
| --- | --- | --- |
| Sum reduction: manual, Cooperative Groups, CUB | `src/sum*.cu` | `make run-reduce` |
| Maximum reduction | `src/max.cu` | `make run-max` |
| Top-k selection in descending order | `src/top_k.cu` | `make run-top-k` |
| ReLU / Leaky ReLU / SiLU / SwiGLU | `src/relu.cu`, `src/leaky_relu.cu`, `src/silu.cu`, `src/swiglu.cu` | `make run-relu`, `make run-leaky-relu`, `make run-silu`, `make run-swiglu` |
| Sigmoid / GEGLU | `src/sigmoid.cu`, `src/geglu.cu` | `make run-sigmoid`, `make run-geglu` |
| RGB to grayscale | `src/rgb2grayscale.cu` | `make run-rgb2grayscale` |
| Clip to an interval | `src/clip.cu` | `make run-clip` |
| Square matrix addition / copy | `src/mat_add.cu`, `src/mat_copy.cu` | `make run-mat-add`, `make run-mat-copy` |
| In-place reversal / array interleave | `src/reverse.cu`, `src/interleave.cu` | `make run-reverse`, `make run-interleave` |
| Repeated FNV-1a hashing | `src/rainbow.cu` | `make run-rainbow` |
| Softmax: three- and four-kernel variants | `src/softmax_*kernel.cu` | `make run-softmax` |
| Scaled dot-product attention | `src/mha.cu` | `make run-attention` |
| Valid 2D / 3D cross-correlation | `src/conv2d.cu`, `src/conv3d.cu` | `make run-conv2d`, `make run-conv3d` |
| Valid 1D cross-correlation | `src/conv1d.cu` | `make run-conv1d` |
| Matrix-vector multiplication | `src/mat_vec_mul.cu` | `make run-mat-vec` |
| FP16 GEMM: scalar, tiled, WMMA, multi-warp WMMA, pipelined WMMA, aligned pipeline, swizzled pipeline, multistage pipeline, mainloop scheduling, larger reuse tile, specialized epilogue, large-tile experiments, distributed mainloop, cuBLAS | `src/gemm.cu`, `src/gemm_tile.cu`, `src/gemm_wmma.cu`, `src/gemm_wmma_tiled.cu`, `src/gemm_wmma_tiled_pipeline.cu`, `src/gemm_wmma_tiled_pipeline_aligned.cu`, `src/gemm_wmma_tiled_pipeline_aligned_swizzled.cu`, `src/gemm_wmma_tiled_pipeline_multistage.cu`, `src/gemm_wmma_tiled_pipeline_mainloop.cu`, `src/gemm_wmma_tiled_pipeline_reuse.cu`, `src/gemm_wmma_tiled_pipeline_epilogue.cu`, `src/gemm_wmma_tiled_pipeline_large.cu`, `src/gemm_wmma_tiled_pipeline_schedule.cu`, `src/gemm_cublas.cu` | `make run-gemm-compare` |
| Batched FP32 matrix multiplication | `src/batched_mm.cu` | `make run-batched-mm` |
| Quantized INT8 matrix multiplication | `src/mm_int8.cu` | `make run-mm-int8` |
| FP16 GEMM in Triton | `src/gemm.triton.py` | `make check-gemm-triton`, `make bench-gemm-triton-compare` |
| Mean categorical cross entropy | `src/cat_ce.cu` | `make run-cat-ce` |
| Mean squared error with FP64 reduction | `src/mse.cu` | `make run-mse` |
| Zero-padded blur / cross-correlation | `src/gauss_blur.cu` | `make run-gauss-blur` |
| Dot product, inclusive prefix sum, histogram | `src/dot.cu`, `src/scan.cu`, `src/histogram.cu` | Compile checks only |

`src/mha.cu` currently implements a single attention operation without a batch or
head dimension; the filename does not imply a complete multi-head attention layer.
The older top-k placeholder in `experimental/top_k.cu` is excluded from the default
build; the working implementation is in `src/top_k.cu`. A standalone reduction comparison is in
`benchmarks/reduce_max.cu` (`make run-max-compare`).

See [operator contracts and limitations](docs/operators.md) before reusing a kernel.

The optional [Triton GEMM integration](docs/gemm-triton.md) reuses the CUDA
correctness cases and compares Triton, CUDA schedule, and cuBLAS on identical
buffers. `make check WITH_TRITON=1` includes its checks; the default CUDA build
does not require Python packages.
Use `TRITON_SOURCE=src/gemm.triton.v2.py` or `src/gemm.triton.v3.py` to select
another version. Repeated `--reference-source` arguments in `TRITON_BENCH_ARGS`
compare multiple Triton versions with CUDA and cuBLAS in one process.

## Running individual tests

```bash
build/sm_80/mat_vec_mul_bench 4096 1024 100   # M N repeats
build/sm_80/gemm_bench 17 33 19 20           # M N K repeats
build/sm_80/gemm_cublas_bench 1024 1024 1024 100
build/sm_80/cat_ce_test 1025 65              # samples classes
build/sm_80/mse_test 50000000                # number of elements
build/sm_80/top_k_test 50000000 100          # N k; includes wrapper timing
build/sm_80/gauss_blur_test 17 35 3 5        # image rows/cols, kernel rows/cols
```

Compare all fourteen GEMM implementations on the same selected GPU:

```bash
CUDA_VISIBLE_DEVICES=3 make check-gemm
CUDA_VISIBLE_DEVICES=3 make run-gemm-compare  # default: M=N=K=1024, 100 repeats
CUDA_VISIBLE_DEVICES=3 make run-gemm-compare GEMM_ARGS="256 2048 512 100"
```

All fourteen use the same CPU reference, FP16 inputs/output, FP32 accumulation, and
row-major `C = alpha * A * B + beta * C` contract. The cuBLAS baseline links with
`-lcublas` and uses `cublasGemmEx` with FP32 reductions; Tensor Core algorithm
selection is left to cuBLAS. Its handle is reused on one selected device and the
default stream; initialization occurs before timing.

GEMM timing uses aligned buffers, five warmup calls, and the median of five
CUDA-event batches. It excludes allocation, copies, and CPU validation, but includes
any host submission gaps between GPU operations. Output guards preserve 256-byte
alignment; a separate correctness case also tests an unaligned output pointer.

Profile all ten WMMA variants and cuBLAS with Nsight Systems:

```bash
CUDA_VISIBLE_DEVICES=3 make nsys-gemm  # M=N=K=1024, 100 repeats per batch
CUDA_VISIBLE_DEVICES=3 make nsys-gemm GEMM_ARGS="512 512 512 100" NSYS_DIR=build/nsys-512
make nsys-gemm-stats                  # Reprint the default directory's reports
make nsys-gemm-stats NSYS_DIR=build/nsys-512
```

`nsys-gemm` builds the eleven benchmarks, profiles them serially even with `make -j`,
then prints kernel and launch/queue/execution summaries in microseconds. Reports
are saved as `gemm_wmma.nsys-rep`, `gemm_wmma_tiled.nsys-rep`,
`gemm_wmma_tiled_pipeline.nsys-rep`, `gemm_wmma_tiled_pipeline_aligned.nsys-rep`,
`gemm_wmma_tiled_pipeline_aligned_swizzled.nsys-rep`, `gemm_wmma_tiled_pipeline_multistage.nsys-rep`,
`gemm_wmma_tiled_pipeline_mainloop.nsys-rep`, `gemm_wmma_tiled_pipeline_reuse.nsys-rep`,
`gemm_wmma_tiled_pipeline_epilogue.nsys-rep`, `gemm_wmma_tiled_pipeline_large.nsys-rep`,
`gemm_wmma_tiled_pipeline_schedule.nsys-rep`, and
`gemm_cublas.nsys-rep` under `NSYS_DIR` (default: `build/sm_80/nsys`). Open these in
the Nsight Systems GUI to compare timelines. Reruns overwrite the eleven reports;
use a different `NSYS_DIR` to retain another shape or run. Override `NSYS` for the
tool path, `NSYS_FLAGS` for collection options, and `NSYS_REPORTS` for stats reports.
The defaults trace CUDA, NVTX, and OS runtime calls, with CPU sampling and context
switch tracing disabled for environments with restricted profiling permissions.

These reports include initialization, two correctness calls, five warmup calls,
and five benchmark batches. At 100 repeats, a one-kernel-per-call implementation
has 507 kernel instances. Stats include warmup/check calls; select the benchmark
region in the GUI for steady-state comparisons. Tracing can affect timings;
use the ordinary benchmarks as well when reporting performance.

Profile all fourteen GEMMs with Nsight Compute:

```bash
make -j2 ncu-gemm-build               # Separate binaries with -lineinfo
CUDA_VISIBLE_DEVICES=3 make ncu-gemm  # Requires GPU performance-counter access
make ncu-gemm-stats                  # Re-export existing reports, no GPU access needed
```

When counters require administrator access, build as the repository owner first.
Then, from an account with sudo privileges in the same repository directory:

```bash
sudo -v
CUDA_VISIBLE_DEVICES=3 make ncu-gemm \
  NCU_RUN='sudo -n --preserve-env=CUDA_VISIBLE_DEVICES' NCU_DIR=/tmp/texo-ncu
make ncu-gemm-stats NCU_DIR=/tmp/texo-ncu
```

Only the profiler is prefixed with `NCU_RUN`; the targets do not change driver
permissions or system configuration. Use a writable `NCU_DIR` when switching
accounts. No account names or passwords are stored in the build configuration.

Defaults: `NCU_SET=full`, `NCU_LAUNCH_SKIP=7`, `NCU_LAUNCH_COUNT=1`, and the same
`GEMM_ARGS` as the other benchmarks. The skip omits two correctness calls and five
warmups for a custom shape with one kernel per call; use positive M/N/K and repeats.
Collection is serial, uses kernel replay with cache flushing, and leaves clocks
unmodified. These timings need not match warm-cache Nsight Systems measurements.
Override `NCU_FLAGS` to change the replay/cache policy. cuBLAS implementation source
is unavailable even though custom kernels are built with line information.

`NCU_BIN_DIR` defaults to `build/sm_80/ncu-bin`; `NCU_DIR` defaults to
`build/sm_80/ncu`. Each variant produces `.ncu-rep`, `.txt`, and `.raw.csv` files;
`comparison.csv` contains key metrics, one row per captured launch. Python 3 is
used for the summary. Missing metrics from smaller sets display as N/A. Reruns
overwrite these files; change `NCU_DIR` to preserve a capture. For a quick run:

```bash
CUDA_VISIBLE_DEVICES=3 make ncu-gemm GEMM_ARGS="128 128 128 2" \
  NCU_SET=basic NCU_GEMMS="gemm_wmma gemm_cublas" NCU_DIR=build/ncu-smoke
```

See [GEMM counter analysis](docs/gemm-ncu-analysis.md) for the measured differences
between the five implementations and the limits of the collected metrics.

The multi-warp WMMA experiment uses a 64x64 output block, four warps, and a 32x32
output per warp. See [WMMA tiling experiments](docs/gemm-wmma-tiling.md) for the
thread mapping, tested configurations, measurements, and tuning commands.

`gemm_wmma_tiled_pipeline.cu` adds two shared-memory input buffers and overlaps
16-byte global-to-shared async copies with WMMA on sm_80+. Partial or unaligned
groups use scalar loads and zero-padding; sm_75 builds use synchronous staging.
The default tile sizes match `gemm_wmma_tiled`, with 26 KiB shared memory/block.
See [WMMA pipeline experiment](docs/gemm-wmma-pipeline.md) for synchronization,
validation, and timing results. It is included in `check-gemm`, `check`,
`sanitize`, `run-gemm-compare`, `nsys-gemm`, and `ncu-gemm`.

```bash
CUDA_VISIBLE_DEVICES=3 make run-gemm-wmma-tiled-pipeline GEMM_ARGS="1024 1024 1024 100"
```

`gemm_wmma_tiled_pipeline_aligned.cu` specializes complete tiles with 16-byte-aligned
A/B pointers. It precomputes copy addresses and removes per-copy matrix checks.
See the [aligned pipeline experiment](docs/gemm-wmma-pipeline-aligned.md) and
[NCU follow-up against cuBLAS](docs/gemm-wmma-pipeline-aligned-ncu.md).

`gemm_wmma_tiled_pipeline_aligned_swizzled.cu` is the separate XOR shared-layout
successor. It adds explicit `ldmatrix`/`mma.sync` and packed output stores, using
16 KiB shared memory for the default 64x64 block and 32x32 warp tile. See the
[shared-layout experiment](docs/gemm-wmma-shared-layout.md).

`gemm_wmma_tiled_pipeline_multistage.cu` extends that design with deeper input
buffering, register operand prefetch, and larger data-reuse tiles. See the
[multistage experiment](docs/gemm-wmma-multistage.md) for dispatch, validation,
and before/after measurements.

`gemm_wmma_tiled_pipeline_mainloop.cu` adds a BK=64 path for small, complete
output grids to reduce mainloop instructions. Cross-chunk register prefetch and
input-stage unrolling were also measured but did not beat this simpler loop. See
the [mainloop experiment](docs/gemm-wmma-mainloop.md) for the selected parameters,
validation, and measurements. Each version has an independent benchmark and
is included in `check-gemm`, `check`, `sanitize`, `run-gemm-compare`, `nsys-gemm`,
and `ncu-gemm`.

`gemm_wmma_tiled_pipeline_reuse.cu` adds a 96x128 block / 48x64 warp tile for
sufficiently populated aligned grids, reducing repeated A/B reads. See the
[reuse-tile experiment](docs/gemm-wmma-reuse.md) for dispatch, NCU results,
and the tested configurations that did not improve performance.

`gemm_wmma_tiled_pipeline_epilogue.cu` adds a compile-time specialization for
alpha=1, beta=0 on native and generic paths. Other scalar values use the general
epilogue. See the [epilogue experiment](docs/gemm-wmma-epilogue.md) for generated
code, validation, and performance measurements.

`gemm_wmma_tiled_pipeline_large.cu` adds dynamic shared input storage above
48 KiB for larger block/warp experiments. See the [large-tile experiment](docs/gemm-wmma-large.md)
for the three-stage investigation and measured configuration tradeoffs.
`check-gemm`, `check`, and `sanitize` also exercise a forced 96 KiB /
16-warp configuration independently of the automatic dispatcher.

`gemm_wmma_tiled_pipeline_schedule.cu` distributes copies and fragment loads
through MMA and reduces address bookkeeping. It selects the new path in measured
A800 grid bands and retains previous kernels elsewhere. See the
[scheduling experiment](docs/gemm-wmma-schedule.md) for cumulative step controls,
NCU evidence, and optional large-shape `*_perf` targets using a cuBLAS reference.

```bash
CUDA_VISIBLE_DEVICES=3 make run-gemm-wmma-tiled-pipeline-aligned-swizzled GEMM_ARGS="1024 1024 1024 100"
CUDA_VISIBLE_DEVICES=3 make run-gemm-wmma-tiled-pipeline-multistage GEMM_ARGS="2048 2048 2048 100"
CUDA_VISIBLE_DEVICES=3 make run-gemm-wmma-tiled-pipeline-mainloop GEMM_ARGS="1024 1024 1024 100"
CUDA_VISIBLE_DEVICES=3 make run-gemm-wmma-tiled-pipeline-reuse GEMM_ARGS="1536 1536 1536 100"
CUDA_VISIBLE_DEVICES=3 make run-gemm-wmma-tiled-pipeline-epilogue GEMM_ARGS="1024 1024 1024 100"
CUDA_VISIBLE_DEVICES=3 make run-gemm-wmma-tiled-pipeline-large GEMM_ARGS="4096 4096 4096 100"
CUDA_VISIBLE_DEVICES=3 make run-gemm-wmma-tiled-pipeline-schedule GEMM_ARGS="1024 1024 1024 100"
```

Tests return nonzero on numerical mismatches or CUDA errors. The newer suites
check CPU references, partial blocks, repeated calls, and output guards. Floating
point tolerances are printed by the test or defined next to its validation code.
A passing suite covers its tested cases, not all possible inputs.

`make check-full` allocates about 400 MB of GPU input storage for the largest MSE
case, plus host buffers and runtime overhead. On a shared GPU, select a device with
sufficient free memory.

## Repository layout

```text
src/           Standalone CUDA operators
tests/        CPU references, correctness tests, and operator benchmarks
benchmarks/    Standalone algorithm comparisons
experimental/  Unfinished exercises, excluded from normal builds
docs/         Contracts, validation, and benchmarking notes
```

The tests directory was previously named `test/`; it is now `tests/`.

## Validation and benchmarks

See [testing and benchmarking](docs/testing.md) for commands and measurement scope,
and the [validation record](docs/validation.md) for the preparation checks.
The GitHub Actions workflow compiles both `sm_75` and `sm_80` targets without a GPU;
it does **not** run GPU correctness tests. Performance depends on shapes, GPU,
clock state, and cache reuse. Do not treat A800 timings as T4 results.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). Small kernels with a reproducible test and
an explanation of the tradeoff are welcome.

## Dependencies

CUDA, CUB, and Cooperative Groups are external NVIDIA dependencies
and remain subject to their own applicable licenses.
