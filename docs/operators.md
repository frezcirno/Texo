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
- `prefix_sum.cu`: inclusive float scan, using recursive block sums. Compile-tested;
  no maintained runtime test target yet.
- `dot.cu`: float dot product; both inputs require 16-byte alignment for `float4`
  loads. Compile-tested only. Empty-input handling and very large integer indices
  still need dedicated validation.
- `histogram.cu`: integer bins, ignoring values outside `[0, num_bins)`; shared-memory
  or global atomic path depending on histogram size. Compile-tested only.

## Dense operators

- `mat_vec_mul.cu`: `A[M,N] * x[N] -> y[M]`, float. A warp per row is the default;
  the test also compares a block per row and different block sizes. `nnz` is a
  retained, unused parameter; this is dense storage, not CSR/COO sparse storage.
- `gemm.cu`: `C = alpha * A * B + beta * C`, with `A[M,K]`, `B[K,N]`, `C[M,N]`.
  Inputs and output use FP16; products and accumulation use FP32. `C` is read only
  when `beta != 0`. Empty output dimensions are a no-op. There is no tiling or
  Tensor Core implementation yet.
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

- `conv2d.cu`: valid cross-correlation, with no kernel flip or padding. An input
  `[H,W]` and kernel `[KH,KW]` produce `[H-KH+1,W-KW+1]`.
- `conv3d.cu`: the analogous valid 3D operation, with depth/height/width ordering.
- `gauss_blur.cu`: same-sized output, zero padding, no kernel flip. The caller supplies
  the weights; they are not normalized by the operator. Odd and even kernels use
  anchor `(KH/2, KW/2)` with integer division. Symmetric Gaussian weights give the
  usual blur; boundary pixels can darken because the zero padding is not renormalized.

## Experiments

`experimental/top_k.cu` contains an empty entry point. It is a placeholder, not a
working implementation. Keep new incomplete exercises here until they have a
working build and a correctness test.
