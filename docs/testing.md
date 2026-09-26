# Testing and benchmarking

## Build and check

```bash
make -j2 all compile-kernels
make check
make check-full
make sanitize
```

`all` links the operator tests/benchmarks. `compile-kernels` additionally compiles
every file in `src/` independently, including examples without runtime tests.
No kernels from different standalone operators are linked into one library.

`check` runs all maintained operator tests with modest benchmark sizes. It covers
partial blocks, multiple blocks, finite outputs, and numerical agreement according
to each executable's tolerances. It does not run the unimplemented top-k exercise.
`check-full` adds the default MSE suite, including 50 million inputs to catch the
rounding error seen with repeated float atomic additions, plus top-k selection with
N=50000000 and k=100. The top-k suite compares against a CPU partial sort and reports
the best of two warm wrapper wall times, including allocation and freeing.
`check-full` also runs an INT8 8192x4096x2048 cancellation regression with an
analytic reference for every output element.

The newer elementwise, matrix addition/copy, reversal, interleave, 1D convolution,
hash and RGB-to-grayscale tests also run in `make check`. Each operator has its own
executable; six elementwise executables share `tests/elementwise.cpp`, while
sigmoid uses `tests/sigmoid.cpp`. Their small helper
`tests/test_utils.h` checks CUDA errors, CPU references, output guards, and repeated
calls, initializing output before each invocation. Input storage is checked for changes;
reversal instead verifies the in-place result and restoration after a second call.
The shared output helper preserves 16-byte alignment. Tests use positive sizes around
warp/block boundaries and non-multiples of block sizes.

Copy, addition, reversal, interleave, ReLU, clip and hashes use exact comparisons.
Leaky ReLU, sigmoid, SiLU and SwiGLU use CPU double references with atol=1e-6 and
rtol=3e-6; GEGLU uses atol=5e-6 and rtol=3e-6 to account for float erf cancellation.
1D cross-correlation uses atol=1e-4 and rtol=1e-5; RGB-to-grayscale uses atol=1e-7
and rtol=1e-6. Clip tests include interval
boundaries and equal bounds; hashes include signed bit patterns and multiple rounds.

`sigmoid_test` retains the basic mixed/zero/negative/positive checks and adds
warp/block boundaries, a million-element tail, independent unaligned input/output
pointers, NaN/infinities, signed zeros, tiny inputs and saturation near float exp
overflow. Every case runs twice on the same buffers with negated inputs on the
second call, nonzero output poison, output guards and bitwise input-preservation
checks. It uses a stable CPU double reference with the tolerances above; a dense
[-80,80] sweep uses relative tolerance only (3e-6) to check tiny normal results.
NaNs must propagate, infinities map exactly to 0/1, and signed zeros map exactly
to 0.5. Positive N is required by the existing operator contract. Run the 28
cases with `make run-sigmoid`; they also run in `make check` and under memcheck
in `make sanitize`.
Validated on 2026-09-25 with NVIDIA A800 80GB PCIe, CUDA Toolkit 12.6.20,
`-O3 -std=c++14 -arch=sm_80`: all 28 cases, the full `make check`, and the
sigmoid memcheck run passed (zero reported errors).

`batched_mm_test` has 44 cases covering rectangular matrices, independently
partial BATCH/M/N blocks, K=1 and long reductions, batch-specific data, an active
last batch with all others zero, identity/zero matrices, and a nonzero final
reduction element. It checks both multiplication with cleared C and three
consecutive `C += A*B` calls starting from nonzero C. Independent CPU double
products use atol=1e-5 and rtol=1e-5, with float rounding of C between accumulated
calls; zero products and inactive batches must preserve finite C exactly.
Unaligned A/B/C are tested separately and together. All calls check input storage
and output guards. K=0 passes null A/B and must preserve finite C; zero BATCH,
M or N must leave storage bitwise unchanged and permit null pointers.
Run it with `make run-batched-mm`; it also runs in `make check` and under
memcheck in `make sanitize`.
Validated on 2026-09-25 with NVIDIA A800 80GB PCIe, CUDA Toolkit 12.6.20,
`-O3 -std=c++14 -arch=sm_80`: all 44 cases, the full `make check`, and the
batched-MM memcheck run passed (zero reported errors).

`mm_int8_test` has 56 default cases covering rectangular matrices, independent
16x16 block boundaries, K=1/257/1025/2048, full signed INT8 inputs, independent input/output
zero points (including endpoints), quantized zeros and identity matrices, and
a nonzero last reduction element. It checks round-to-nearest, ties-to-even on
both signs, the adjacent FP32 values around 0.5, all three scales, nonbinary
scales, both saturation limits, and saturation after adding zero_point_C.
A/B/C are tested at independent and combined byte offsets. K=0 uses null A/B
and must fill C with zero_point_C. M/N are positive; empty outputs and invalid
scales are outside this operator's test contract.

The reference uses an INT64 dot product and double scaling, with an explicit
floor/parity implementation of ties-to-even rounding. Results
must match exactly. The main cases use binary scales and bounded sums to avoid
ambiguity from FP32 accumulation; the decimal-scale case stays away from half
boundaries. These are selected quantization regressions, not a claim of exact
FP32-versus-ideal agreement for all scales and reduction lengths. The reported
LeetGPU 3x5x2 failure pins all 15 expected bytes directly, including C[1,3]=58;
its sign-reversed case pins -58. A double formula applied to the binary FP32
scales can land on a different side of a half boundary, so those external
expectations are not regenerated from that formula. The old `roundf` kernel
reproduced 59 versus 58; `nearbyintf` passes the case and both-sign tie checks.
See NVIDIA's [rounding functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html)
for the difference between halfway-away-from-zero and halfway-to-even.
Two K=2048 cases pair integer products that cancel, leaving a small residual,
using scales 0.1/0.1/0.01. The former per-term FP32 scaling/reduction failed the
random case with 8 instead of 9 even after fixing the rounding rule. INT64
accumulation followed by one final scaling passes it. Two K=33026 cases also
check positive/negative dot products beyond INT32 range with unsaturated outputs.
Every case runs twice with C initialized to opposite INT8 endpoints, checking overwrites,
unchanged input bytes and output guards. Run `make run-mm-int8`; the test is
also included in `make check` and memcheck in `make sanitize`.

`make run-mm-int8 MM_INT8_ARGS=--large` runs the separate 8192x4096x2048 case,
also included in `make check-full`. Repeated A halves multiply opposite B halves;
the remaining term determines a row/column-specific integer output. This gives
an exact analytic reference for all 33,554,432 outputs without a cubic CPU
matrix multiply. It uses the reported dimensions/scales, with constructed data;
the platform's truncated input arrays cannot be replayed verbatim. The kernel
retains FP32 operations in the order `float(dot) * scale_A * scale_B / scale_C`;
precombining scales or using double can change half-boundary results.

The original lower-saturation branch assigned +128 before converting to INT8.
It happened to yield -128 and pass the baseline tests on this A800/compiler;
the branch now assigns -128 directly to avoid the out-of-range conversion.

Validated on 2026-09-25 with NVIDIA A800 80GB PCIe, CUDA Toolkit 12.6.20,
`-O3 -std=c++14 -arch=sm_80`: all 56 default cases, the full-size analytic
regression, and the complete `make check-full` passed. Memcheck passed all 56
default cases with zero errors. The `sm_75` (T4) build also passed; no physical
T4 runtime or online resubmission is claimed by these local checks.

`sanitize` runs memory checks on GEMM, batched and INT8 matrix multiplication, blur,
categorical cross entropy, MSE, top-k, interleave and sigmoid,
and synchronization checks on the two Cooperative Groups loss reductions and top-k. It is a
selected set, not a sanitizer audit of every operator.

## Optional Triton GEMM

The optional Triton GEMM runner consumes `gemm_cublas_bench --list-cases`,
which exports the CUDA suite's 57 cases as JSON lines without initializing a GPU.
Run `make check-gemm-triton PYTHON=/path/to/python`, or include it in the main
suite with `make check WITH_TRITON=1 PYTHON=/path/to/python`. The runner uses an
independent CPU double reference, tests two consecutive calls, input preservation
and output guards, and checks all seven autotune candidates on representative
cases. `make sanitize-gemm-triton` selects tail, misalignment and K=0 cases for
memcheck/racecheck/synccheck; `WITH_TRITON=1` also includes it in `make sanitize`.
See [Triton GEMM](gemm-triton.md) for the separate performance comparison target
and its Python submission overhead.

## Architecture and device selection

```bash
make -j2 NVCC_ARCH=sm_75
CUDA_VISIBLE_DEVICES=0 make check NVCC_ARCH=sm_75
```

A T4 needs `sm_75`; A100/A800 use `sm_80`. Architecture-specific output directories
prevent accidental reuse of a binary for a different GPU. Set `CUDA_VISIBLE_DEVICES`
to select an available physical device; tests use logical device 0.

The build workflow in `.github/workflows/build.yml` has no GPU. It checks compilation
and linking only. A successful workflow is not evidence of numerical correctness
or T4 runtime validation.

## Measurement scope

- Matrix-vector and GEMM benchmarks warm up and report the median of five batches
  of repeated launches. They reuse input buffers and exclude host/device copies.
  GEMM timing uses `beta=0` to prevent repeated accumulation into C.
  `make check-gemm` validates scalar, tiled, all ten WMMA variants, and cuBLAS;
  `make run-gemm-compare GEMM_ARGS="1024 1024 1024 100"` compares their timings.
  The 52-case GEMM suite also checks misaligned A/B base pointers on complete
  tiles, one/two/three full K chunks, multiple output blocks, independently partial
  M/N/K dimensions, K=0, and unaligned C with aligned A/B. Wide 1024x2048 outputs
  with K=32/96 cover larger block/warp tile builds, nontrivial alpha/beta, and
  aligned/unaligned output stores. Additional two/four/five/seven/eight-chunk
  cases cover short prologues, ring wraparound, and drain for multistage input
  buffers. Three/five/seven BK=64 chunks also exercise ring wraparound/drain,
  nontrivial alpha/beta, and unaligned C. The 96x128 tile is checked at its
  192-block automatic-dispatch boundary with one/three K chunks, alpha/beta,
  and unaligned C; forced BM=96 builds also cover a non-64-row five-chunk case.
  Alpha/beta checks vary each scalar independently and initialize C with NaNs
  for beta=0 on native, tail, K=0, and unaligned-output paths, including beta=-0.
  The reference ignores old C when beta=0.
  Four additional 256x256 cases cover one/four/five/seven K chunks, general
  alpha/beta, NaN old C, and unaligned C for larger tiles. The forced
  `gemm_wmma_tiled_pipeline_large_dynamic_test` runs the same suite using
  a 128x256 block, 32x64 warp, four stages, and 96 KiB dynamic shared memory.
  It is included in `check-gemm`, `check`, and all three sanitizer tools;
  it is separate from the fourteen default benchmark implementations.
  The scheduled version adds forced 128x128/BK32 three- and four-stage tests,
  and a 96 KiB 128x256/BK64 two-stage test with 16 warps. Five further cases
  cover BK64 one/three/nine-chunk drains and the two automatic grid bands.
  The shared GEMM suite has 57 cases; all forced tests participate in check
  and memcheck/racecheck/synccheck. Large-shape optional `*_perf` targets use
  `benchmarks/gemm.cu` with a cuBLAS FP32-reduction reference; they supplement
  the independent CPU checks rather than replacing them.
  `sanitize` runs memcheck,
  racecheck, and synccheck over the complete suite for all nine pipelined versions.
  All use 256-byte-aligned benchmark buffers, with output guards and a separate
  unaligned correctness case. cuBLAS handle creation happens before timing.
  CUDA-event batches include any host submission gaps between GPU operations.
- Attention, softmax, and convolution benchmarks also report wall-clock timings.
  A wrapper may include allocation, deallocation, or synchronization; read the
  printed fields and wrapper before comparing with a kernel-only number.
- Sum compares the manual, Cooperative Groups, and CUB wrappers. CUB caches its
  workspace; the CG implementation allocates per call. These are wrapper timings,
  not a controlled measurement of reduction instructions alone.
- Maximum and sum use simple reference inputs and are basic benchmark checks, not
  exhaustive numerical validation suites.

Record GPU model, toolkit, compiler flags, input shapes, repetitions, cache reuse,
and any competing GPU work with benchmark results. Small kernels can be dominated
by launch overhead; large inputs can be dominated by memory bandwidth. Recheck
correctness whenever an optimization changes accumulation order or precision.

## Example custom runs

```bash
build/sm_80/reduce_bench 1048576 20
build/sm_80/max_bench 1048576 20
build/sm_80/softmax_bench 1025 10 3
build/sm_80/attention_bench 17 33 16 10 3
build/sm_80/conv2d_bench 17 35 3 5 10 3
build/sm_80/conv3d_bench 9 11 13 3 3 3 10 3
build/sm_80/mat_vec_mul_bench --check-only
build/sm_80/gemm_bench --check-only
build/sm_80/gauss_blur_test 17 35 2 4
build/sm_80/mse_test 50000000
make run-swiglu
make run-interleave
make run-conv1d
```

Known empty-input conventions differ: categorical cross entropy leaves output
unchanged; MSE writes zero; GEMM and blur with empty output shapes do no work.
Tests document these explicitly rather than inventing a universal tensor contract.
