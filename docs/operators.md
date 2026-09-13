# Operator contracts and limitations

All tensor storage is contiguous and row-major unless noted otherwise. Callers
allocate device input/output buffers of the documented sizes. Do not alias output
with inputs unless an implementation explicitly documents support. Valid, finite
inputs and sizes that fit the kernels' integer indexing are the default contract.
These are learning implementations, not a checked tensor API.

Most wrappers use the default CUDA stream. Some allocate/free temporary storage or
synchronize internally. They are not promised to support CUDA Graph capture,
concurrent calls, or arbitrary user streams. Check CUDA errors and synchronize
before consuming outputs on the CPU.

## Reductions and scans

- `sum.cu`: float input/output, grid-stride accumulation and atomic block sums.
  The output is reset on every call. The vectorized input pointer must be
  16-byte aligned. FP32 accumulation can lose accuracy for long or ill-conditioned sums.
- `sum_cg.cu`: float input/output, double accumulation and a temporary double scalar.
  Explicit tile scratch supports the target architectures used by this project.
- `sum_cub.cu`: CUB baseline with cached temporary storage. The cache is intended
  for sequential calls on one device; it is not thread-safe or device-switch-safe.
- `max.cu`: float maximum; vectorized input requires 16-byte alignment. The empty
  reduction produces negative infinity. NaN behavior is not specified by the tests.
- `top_k.cu`: float input `[N]` to descending output `[k]`, preserving duplicates
  and input storage; `1 <= k <= N <= 100000000`, no NaNs. Four byte-wise radix
  selection passes find the kth value, then a fifth pass collects larger values.
  Only the k output values are sorted, using bitonic tiles and merges for k > 1024.
  Uses CUDA intrinsics supported by sm_75/sm_80, without CUB or Thrust. Temporary
  storage is constant for k <= 1024 and O(k) for larger outputs.
- `scan.cu`: inclusive float scan, using recursive block sums. Compile-tested;
  no maintained runtime test target yet.
- `dot.cu`: float dot product; both inputs require 16-byte alignment for `float4`
  loads. Compile-tested only. Empty-input handling and very large integer indices
  still need dedicated validation.
- `histogram.cu`: integer bins, ignoring values outside `[0, num_bins)`; shared-memory
  or global atomic path depending on histogram size. Compile-tested only.

## Dense operators

- `mat_add.cu`: elementwise addition of two float `[N,N]` matrices.
- `mat_copy.cu`: copy a float `[N,N]` matrix. N is the side length, not the
  element count. Both square-matrix examples require positive N and N*N fitting int.
- `mat_vec_mul.cu`: `A[M,N] * x[N] -> y[M]`, float. A warp per row is the default;
  the test also compares a block per row and different block sizes. `nnz` is a
  retained, unused parameter; this is dense storage, not CSR/COO sparse storage.
- `gemm.cu`, `gemm_tile.cu`, `gemm_wmma.cu`, `gemm_wmma_tiled.cu`, `gemm_cublas.cu`:
  row-major `C = alpha * A * B + beta * C`, with `A[M,K]`, `B[K,N]`, `C[M,N]`.
  Inputs and output use FP16 with FP32 accumulation. `C` is read only when
  `beta != 0`. Empty output dimensions are a no-op; K=0 scales C by beta.
  Variants use scalar arithmetic, shared-memory tiling, single-warp WMMA,
  multi-warp WMMA, and cuBLAS, respectively. Multi-warp WMMA defaults to a 64x64
  output block with four warps, a K chunk of 32, and 16 half elements of shared-row
  padding. Each warp computes 32x32 through four accumulator fragments. See
  [the experiment notes](gemm-wmma-tiling.md) for compile-time tuning parameters.
  The cuBLAS baseline uses `cublasGemmEx`, disallows reduced-precision
  reductions, and reuses a handle on one selected device per host thread with the
  default stream. It requires linking with `-lcublas`.
- `gemm_wmma_tiled_pipeline.cu`, `gemm_wmma_tiled_pipeline_aligned.cu`: the same
  FP16 GEMM contract with two input buffers and 16-byte asynchronous staging on
  sm_80+. Partial or unaligned input groups use scalar loads and zero-fill;
  sm_75 uses synchronous staging. The aligned version selects a specialization
  when M/N/K are positive multiples of BM/BN/BK and A/B are 16-byte aligned.
  This guarantees complete tiles and aligned row strides. C still needs only
  half alignment. Other inputs use the generic pipeline, including K=0. Both
  versions use `size_t` address arithmetic and flattened output tile grids.
  The aligned fast path uses XOR shared storage and explicit `ldmatrix`/`mma.sync`
  when BN/BK are powers of two, and retains full-tile WMMA otherwise. It writes
  packed `half2` only when C is four-byte aligned, with scalar stores otherwise.
  Default block/warp tiles remain 64x64/32x32. See
  [the shared-layout notes](gemm-wmma-shared-layout.md) for tuning and validation.
- `batched_mm.cu`: FP32 batched multiplication with A[BATCH,M,K], B[BATCH,K,N]
  and C[BATCH,M,N], using contiguous row-major storage without broadcasting or
  transposes. The current kernel adds into C; callers must clear C before each
  multiplication. Tests use positive dimensions and CPU double references.
- `softmax_3kernel.cu` / `softmax_4kernel.cu`: softmax over one float vector, with
  maximum subtraction for stability. Vectorized paths require 16-byte alignment.
- `mha.cu`: `softmax(Q * K^T / sqrt(d)) * V`, with `Q[M,d]`, `K[N,d]`, `V[N,d]`.
  Uses an intermediate `M*N` allocation. No masking, batching, or head dimension.

## Losses

- `cat_ce.cu`: mean categorical cross entropy from logits `[N,C]` and integer labels
  `[N]`. Requires `C > 0` and each label in `[0,C)`. Uses stable log-sum-exp.
  `N == 0` is currently a no-op; the output is unchanged. Each block atomically adds
  its mean contribution in FP32, so very large batches can accumulate rounding error.
- `mse.cu`: mean squared error between two float vectors. Uses two-stage FP64
  accumulation, then converts the final mean to float. `N <= 0` writes zero.
  Temporary block sums and an extra kernel trade speed for numerical accuracy.

## Spatial operators

- `conv1d.cu`: valid float cross-correlation, without padding or kernel reversal.
  Input length N and kernel length K produce N-K+1 outputs; `1 <= K <= N`.
- `conv2d.cu`: valid cross-correlation, with no kernel flip or padding. An input
  `[H,W]` and kernel `[KH,KW]` produce `[H-KH+1,W-KW+1]`.
- `conv3d.cu`: the analogous valid 3D operation, with depth/height/width ordering.
- `gauss_blur.cu`: same-sized output, zero padding, no kernel flip. The caller supplies
  the weights; they are not normalized by the operator. Odd and even kernels use
  anchor `(KH/2, KW/2)` with integer division. Symmetric Gaussian weights give the
  usual blur; boundary pixels can darken because the zero padding is not renormalized.

## Elementwise and data movement

- `relu.cu`: `max(x, 0)` for N float elements.
- `leaky_relu.cu`: `x` for positive inputs and `0.01*x` otherwise.
- `silu.cu`: `x * sigmoid(x)` for N float elements.
- `sigmoid.cu`: logistic sigmoid for N float elements.
- `swiglu.cu`: even-length input N, split into contiguous halves x and gate;
  returns N/2 values `silu(x[i]) * gate[i]`.
- `geglu.cu`: even-length input N, split into contiguous halves x and gate;
  returns N/2 values `x[i] * gelu(gate[i])`, using the erf form of GELU.
- `rgb2grayscale.cu`: interleaved float RGB input `[height,width,3]` produces
  `[height,width]` values `0.299*R + 0.587*G + 0.114*B`.
- `clip.cu`: clip N finite float values to `[lo,hi]`, where `lo <= hi`.
- `reverse.cu`: reverses N float elements in place. Reversing twice restores input.
- `interleave.cu`: two float inputs A[N], B[N] produce 2*N outputs in the order
  A[0], B[0], A[1], B[1], etc.
- `rainbow.cu`: interpret N signed 32-bit integers as unsigned bit patterns and
  apply four-byte FNV-1a hashing R times. Each round processes the least significant
  byte first; output is uint32. R=0 returns the original bit patterns.

These examples require positive lengths (and nonnegative R); empty-input behavior
is not part of their tests. Inputs stay unchanged except for the in-place reverse.

## Experiments

`experimental/top_k.cu` contains an empty entry point. It is a placeholder, not a
working implementation. Keep new incomplete exercises here until they have a
working build and a correctness test.
