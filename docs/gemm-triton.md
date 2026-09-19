# Triton GEMM integration

`src/gemm.triton.py` implements row-major FP16 A[M,K], B[K,N], C[M,N], FP32
accumulation, and `C = alpha*A*B + beta*C`. Its existing operator is unchanged
by this integration. Inputs must be contiguous CUDA tensors with nonnegative
M/N/K and matching storage. Tails and unaligned base pointers are supported.
M/N=0 is a no-op, K=0 scales C, and beta=0 (including negative zero) ignores
old C. The autotuner's `restore_value=["C"]` preserves the original C during
candidate evaluation, which matters when beta is nonzero. See the official
[autotune documentation](https://triton-lang.org/main/python-api/generated/triton.autotune.html).

## Correctness

Use a Python environment containing CUDA-enabled PyTorch and Triton. The initial
validation environment uses PyTorch 2.5.1+cu121 and Triton 3.1.0 on an A800;
the C++ targets use CUDA Toolkit 12.6. No global Python or driver changes are
needed. The default CUDA-only build keeps Python packages optional.

```sh
# Activate the desired Python environment first, or set PYTHON=/path/to/python.
CUDA_VISIBLE_DEVICES=0 make check-gemm-triton
CUDA_VISIBLE_DEVICES=0 make check WITH_TRITON=1
CUDA_VISIBLE_DEVICES=0 make sanitize-gemm-triton
```

`tests/gemm_triton.py` reads the 57 existing cases through the C++ test binary's
GPU-free `--list-cases` option. There is no second copy of the case table. The
Python runner generates deterministic FP16 data and compares against CPU double
matrix multiplication, converting the result through FP32 to FP16. This has the
same reference/tolerance semantics as the CUDA tests; Python and C++ random
generators do not produce identical inputs. Every case runs twice and checks
`abs_error <= 0.01 + 0.01*abs(reference)`, input preservation and output guards.
Cases cover all three tails, unaligned A/B/C, general alpha/beta, NaN old C with
beta=0, zero dimensions, long K, and larger output grids.

The default `--all-configs` also directly tests each of the seven JIT candidate
configurations on seven representative cases (49 additional checks). This catches
incorrect candidates even when the autotuner would select a different one.
Public-solve checks retain real autotuning, so their first calls include internal
candidate timing; no performance comparison is inferred from correctness runs.
The sanitizer target directly checks all candidates on three selected cases
under memcheck, racecheck and synccheck. It uses `--fixed-only --all-configs`
to avoid repeatedly benchmarking candidates under instrumentation; ordinary
correctness checks exercise the real public-solve autotuning path.

```sh
# Focused checks; names come from --list-cases.
make check-gemm-triton TRITON_TEST_ARGS="--case multi-tile-tail --case ab10-nan-K0"
build/sm_80/gemm_cublas_bench --list-cases
```

`TRITON_SOURCE` selects another standalone version for checks, sanitizer runs
and benchmarks. The default remains `src/gemm.triton.py`. Versions
`src/gemm.triton.v2.py` and `src/gemm.triton.v3.py` expose the same `solve`
contract and are supported by the runner, including their 32 and 34 raw JIT
candidate configurations. V2 needs the newer custom `do_bench` autotune API;
V3 uses the CUDA-Graph tuning API also available in Triton 3.1. The v2/v3
comparison uses Triton 3.3.1 for all versions to keep the compiler consistent.

```sh
make check-gemm-triton TRITON_SOURCE=src/gemm.triton.v3.py
make sanitize-gemm-triton TRITON_SOURCE=src/gemm.triton.v3.py
```

## Performance comparison

```sh
# Default: twelve square/rectangular/short-K shapes, up to 8192 cubed.
CUDA_VISIBLE_DEVICES=0 make bench-gemm-triton-compare

# Repeated --shape selects a smaller or custom set.
CUDA_VISIBLE_DEVICES=0 make bench-gemm-triton-compare \
  TRITON_BENCH_ARGS="--shape 1024 1024 1024 --shape 4096 4096 4096"

# Exercise the comparison adapters and references without timing batches.
CUDA_VISIBLE_DEVICES=0 make bench-gemm-triton-compare \
  TRITON_BENCH_ARGS="--shape 65 129 67 --shape 1024 1024 128 --verify-only"

# Compare v2, v1, v3, CUDA schedule and cuBLAS in the same process.
CUDA_VISIBLE_DEVICES=0 make bench-gemm-triton-compare \
  TRITON_SOURCE=src/gemm.triton.v2.py \
  TRITON_BENCH_ARGS="--reference-source src/gemm.triton.py --reference-source src/gemm.triton.v3.py --rounds 4"
```

`--reference-source` may be repeated. CSV labels `triton`, `triton_reference`,
`triton_reference_2`, etc. map to the source paths recorded in the metadata JSON.
Each implementation has its own tuning cache; all measured calls use the same
buffers and scalar values. Changing the Triton compiler can change both selected
configurations and generated kernels, so rerun the reference version alongside
the new version rather than using earlier measurements as its baseline.

`benchmarks/gemm_triton.py` loads independent shared libraries built from the
unmodified CUDA solve entry points. The default is
`TRITON_CUDA_GEMM=gemm_wmma_tiled_pipeline_schedule`; another standalone GEMM
source can be selected with that variable. cuBLAS is always included as a baseline.
The adapters deliberately use logical device 0 and its default stream, matching
the native wrappers. They are comparison tools, not general multi-stream bindings.

All implementations reuse **the same** random FP16 A/B and guarded C,
with alpha=1/beta=0. All outputs are checked against `src/gemm_cublas.cu` with
FP32 accumulation and reduced-precision reduction disabled before timing. CPU
checks are a separate requirement; the performance reference does not replace
them. JIT, autotuning, native-handle creation, allocations, transfers, reference
generation and validation are outside timing. Triton's chosen configuration is
recorded for each shape.

Five warmups precede five CUDA-event batches of 100 calls. Two rounds reverse
implementation order; the output is the mean of each round's median. Configure
these with `--iterations`, `--batches`, and `--rounds` in `TRITON_BENCH_ARGS`.
The timed batches include Python/ctypes submission gaps. In particular, small
matrices can measure launch overhead more than GPU execution. These are warmed
API timings, not kernel-only measurements, and should not be mixed with the
historical standalone C++ measurements or NCU replay durations.

Results are written to `BIN_DIR/triton-comparison.csv`; a sibling JSON records
versions and methodology. PyTorch can load a different cuBLAS/CUDA runtime from
the toolkit used to compile the native shared libraries. The metadata records
the runtime, driver API and cuBLAS versions actually resolved in the Python
process. Record competing workloads and use an idle GPU for performance claims.
`--verify-only` writes no timing report. The initial 2026-09-19 pass performed
validation only. A subsequent run used relatively idle GPU 2; see the
[measured Triton/CUDA/cuBLAS comparison](gemm-triton-performance.md).
The subsequent [v1/v2/v3 comparison](gemm-triton-versions.md) reruns all three
Triton versions with one compiler and includes CUDA/cuBLAS, selected
configurations, sanitizer results and a separate kernel-duration profile.

## Validation recorded on 2026-09-19

On A800 with PyTorch 2.5.1+cu121 / Triton 3.1.0:

- `make check WITH_TRITON=1` passed the existing CUDA suite and all 106 Triton
  checks (57 exported cases plus 49 forced-candidate checks).
- `make sanitize-gemm-triton` passed 17 checks under each of memcheck, racecheck
  and synccheck, with zero reported errors/hazards.
- The three-way comparison's `--verify-only` mode passed 65x129x67,
  1024x1024x128, and 2048x2048x2048. Both custom implementations matched the
  cuBLAS reference exactly on those inputs, including output guards.
- Case export succeeded with no visible GPU and preserved negative-zero beta;
  invalid shape/repetition/case-name arguments returned nonzero as expected.

The Python comparison process resolved CUDA runtime 12.1 and cuBLAS 12.1.3,
despite compiling the native adapters with Toolkit 12.6. That initial validation
pass generated no timing batches or performance CSV. Ignored logs are under `build/sm_80/` as
`triton-make-check.log`, `triton-sanitize.log`, and `triton-compare-verify.log`.
