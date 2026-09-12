# Texo

A collection of CUDA operator implementations, correctness tests, and benchmarks.
This repository follows the process of learning GPU programming: start with a
readable implementation, validate it against a CPU reference, then measure changes.

The kernels are standalone examples, not a single linkable library. Most expose an
`extern "C" void solve(...)` entry point and accept device pointers. Build each
operator separately; the Makefile renames entry points when comparing variants.

## Requirements

- Linux, GNU Make, and a C++ compiler supported by the CUDA Toolkit.
- CUDA Toolkit 12.6 is the development baseline. CUB and Cooperative Groups come
  with the toolkit; no external C++ dependencies are vendored.
- An NVIDIA GPU and compatible driver to run tests. Compilation alone needs no GPU.
- `compute-sanitizer` for the optional memory and synchronization checks.

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
| Clip to an interval | `src/clip.cu` | `make run-clip` |
| Square matrix addition / copy | `src/mat_add.cu`, `src/mat_copy.cu` | `make run-mat-add`, `make run-mat-copy` |
| In-place reversal / array interleave | `src/reverse.cu`, `src/interleave.cu` | `make run-reverse`, `make run-interleave` |
| Repeated FNV-1a hashing | `src/rainbow.cu` | `make run-rainbow` |
| Softmax: three- and four-kernel variants | `src/softmax_*kernel.cu` | `make run-softmax` |
| Scaled dot-product attention | `src/mha.cu` | `make run-attention` |
| Valid 2D / 3D cross-correlation | `src/conv2d.cu`, `src/conv3d.cu` | `make run-conv2d`, `make run-conv3d` |
| Valid 1D cross-correlation | `src/conv1d.cu` | `make run-conv1d` |
| Matrix-vector multiplication | `src/mat_vec_mul.cu` | `make run-mat-vec` |
| FP16 GEMM with FP32 accumulation | `src/gemm.cu` | `make run-gemm` |
| Mean categorical cross entropy | `src/cat_ce.cu` | `make run-cat-ce` |
| Mean squared error with FP64 reduction | `src/mse.cu` | `make run-mse` |
| Zero-padded blur / cross-correlation | `src/gauss_blur.cu` | `make run-gauss-blur` |
| Dot product, inclusive prefix sum, histogram | `src/dot.cu`, `src/prefix_sum.cu`, `src/histogram.cu` | Compile checks only |

`src/mha.cu` currently implements a single attention operation without a batch or
head dimension; the filename does not imply a complete multi-head attention layer.
The older top-k placeholder in `experimental/top_k.cu` is excluded from the default
build; the working implementation is in `src/top_k.cu`. A standalone reduction comparison is in
`benchmarks/reduce_max.cu` (`make run-max-compare`).

See [operator contracts and limitations](docs/operators.md) before reusing a kernel.

## Running individual tests

```bash
build/sm_80/mat_vec_mul_bench 4096 1024 100   # M N repeats
build/sm_80/gemm_bench 17 33 19 20           # M N K repeats
build/sm_80/cat_ce_test 1025 65              # samples classes
build/sm_80/mse_test 50000000                # number of elements
build/sm_80/top_k_test 50000000 100          # N k; includes wrapper timing
build/sm_80/gauss_blur_test 17 35 3 5        # image rows/cols, kernel rows/cols
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
