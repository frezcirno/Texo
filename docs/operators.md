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
- `slice_sum.cu`: sum INT32 `input[S..E]` into separate `output[1]`, with
  **both endpoints inclusive**. The [Subarray Sum contract](https://leetgpu.com/challenges/subarray-sum)
  specifies N in [1,100000000], input values in [1,10], and
  `0 <= S <= E < N`; S=E returns one element and the largest sum fits INT32.
  Tests use aligned input allocations but arbitrary legal S, including all
  four offsets modulo four. A vectorized implementation must handle slice
  starts that are not 16-byte aligned. Tests use exact INT64 CPU references,
  unpadded input tails, input preservation and output guards. Empty slices,
  negative inputs and aliasing are outside the tested contract. Run
  `make run-slice-sum`; `SLICE_SUM_ARGS=--large` checks full/interior slices of
  100M-element arrays. See [testing notes](testing.md).
- `slice_sum2d.cu`: sum a rectangular region of contiguous row-major INT32
  `input[N,M]` into separate `output[1]`. Both row and column endpoints are
  inclusive; input rows remain M elements apart even when the selected
  rectangle is narrower. The [2D Subarray Sum contract](https://leetgpu.com/challenges/2d-subarray-sum)
  specifies N/M in [1,10000], values in [1,10], and nonempty regions within
  the matrix. Every possible sum fits INT32. Tests use an exact INT64 CPU
  reference, input preservation, output guards and separate overwrite/reuse
  checks. Run `make run-slice-sum2d`; `SLICE_SUM2D_ARGS=--large` selects
  full/interior regions of 10000x10000 matrices. See [testing notes](testing.md).
- `slice_sum3d.cu`: sum a cuboid in contiguous row-major INT32 `input[N,M,K]`
  into separate `output[1]`. Depth, row and column endpoints are all inclusive;
  input row/depth strides stay K and M*K regardless of the selected cuboid.
  The [3D Subarray Sum contract](https://leetgpu.com/challenges/3d-subarray-sum)
  gives each dimension in [1,500] and input values in [1,10], with nonempty
  regions inside the tensor. The maximum sum, 1,250,000,000, fits INT32.
  Tests use independent INT64 CPU sums, input preservation, output guards,
  and separate output-overwrite/reuse regressions. Run `make run-slice-sum3d`;
  `SLICE_SUM3D_ARGS=--large` selects full/interior regions of 500^3 tensors.
  See [testing notes](testing.md) for validation status.
- `count.cu`: count the INT32 elements equal to K in contiguous `input[N]`,
  writing one INT32 result to separate `output[1]`. The
  [LeetGPU contract](https://leetgpu.com/challenges/count-array-element) gives
  N in [1,100000000], input/K in [1,100000]; every possible count fits INT32.
  Inputs must be 16-byte aligned for the current `int4` loads; tests retain
  this alignment and leave scalar tails unpadded. Empty inputs and arbitrary
  unaligned input views are not covered. Baseline cases clear output before
  each call; separate regressions require overwriting nonzero/previous results.
  Run `make run-count`; `COUNT_ARGS=--large` checks 16,777,217/100M elements.
  See [testing notes](testing.md) for the initial output-reset failure.
- `count3d.cu`: count occurrences of P in contiguous INT32 `input[N,M,K]`,
  writing one INT32 result to separate `output[1]`. K is a dimension; P is the
  comparison value. The [LeetGPU contract](https://leetgpu.com/challenges/count-3d-array-element)
  gives each dimension in [1,1000] and input/P in [1,100]. The total count is
  at most 1,000,000,000 and fits INT32. Tests use 16-byte aligned input for the
  current int4 loads, positive dimensions, unpadded tails, exact CPU counts,
  unchanged-input/output-guard checks, and separate output-overwrite cases.
  Run `make run-count3d`; `COUNT3D_ARGS=--large` selects 500^3/1000^3 cases.
  See [testing notes](testing.md) for validation and large-test memory use.
- `max.cu`: float maximum; vectorized input requires 16-byte alignment. The empty
  reduction produces negative infinity. NaN behavior is not specified by the tests.
- `max_subarray_sum.cu`: maximum sum over all contiguous windows of **exactly**
  `window_size` INT32 elements, writing one result to separate `output[1]`.
  The [LeetGPU contract](https://leetgpu.com/challenges/max-subarray-sum) specifies
  N in [1,50000], input values in [-10,10], and `1 <= window_size <= N`.
  Windows are nonempty; the maximum may be negative, and every result fits
  INT32. This is a fixed-length window problem. Run `make run-max-subarray-sum`;
  `MAX_SUBARRAY_SUM_ARGS=--large` checks N=50000 with four window sizes.
  Tests use independent INT64 CPU window sums, exact comparisons, input
  preservation, output guards and explicit overwrite/reuse checks. All 80 quick
  and four large cases pass after resetting output before the atomic maximum.
  The official reference can incorrectly omit the first window; see the
  [reference investigation](max_subarray_sum_reference.md) and
  [testing notes](testing.md) for validation results.
- `top_k.cu`: float input `[N]` to descending output `[k]`, preserving duplicates
  and input storage; `1 <= k <= N <= 100000000`, no NaNs. Four byte-wise radix
  selection passes find the kth value, then a fifth pass collects larger values.
  Only the k output values are sorted, using bitonic tiles and merges for k > 1024.
  Uses CUDA intrinsics supported by sm_75/sm_80, without CUB or Thrust. Temporary
  storage is constant for k <= 1024 and O(k) for larger outputs.
- `scan.cu`: inclusive float scan, using recursive block sums. Compile-tested;
  no maintained runtime test target yet.
- `dot.cu`: [FP32 dot product](https://leetgpu.com/challenges/dot-product),
  A[N]/B[N] to one FP32 result, with N in [1,100000000]. Inputs require
  16-byte alignment for `float4` loads. The intended contract overwrites result;
  the current source uses atomic addition without clearing result, so nonzero
  initial output and repeated calls fail. `make run-dot` checks the contract
  with a CPU double reference; `DOT_ARGS=--large` checks maximum size, an odd
  tail and decimal accumulation accuracy under documented local tolerances.
- `dot_fp16.cu`: [FP16 dot product](https://leetgpu.com/challenges/fp16-dot-product),
  half A[N]/B[N] to one half result, with the same positive size range.
  Inputs require 4-byte alignment for `half2` loads. The source clears an FP32
  accumulator, combines block sums in FP32 and converts the final value to half.
  `make run-dot-fp16` checks tails, output overwrites, FP32 intermediates and FP16
  rounding. `DOT_FP16_ARGS=--large` includes the 100M-element decimal precision
  regression. Both operators share [tests/dot.cpp](../tests/dot.cpp); see
  [testing notes](testing.md) for accuracy thresholds and current failures.
- `histogram.cu`: integer bins, ignoring values outside `[0, num_bins)`; shared-memory
  or global atomic path depending on histogram size. Compile-tested only.

## Dense operators

- `mat_add.cu`: elementwise addition of two float `[N,N]` matrices.
- `mat_copy.cu`: copy a float `[N,N]` matrix. N is the side length, not the
  element count. Both square-matrix examples require positive N and N*N fitting int.
- `mat_pow.cu`: matrix power `output = input^P`, with separate contiguous FP32
  `[N,N]` input/output in row-major order. The
  [LeetGPU contract](https://leetgpu.com/challenges/matrix-power) specifies
  `1 <= N <= 1024`, `1 <= P <= 20`, and input elements in `[-10,10]`.
  P=1 copies the input; P=0, negative powers and aliasing are outside the tested
  contract. `make run-mat-pow` uses independent CPU double and analytic
  references, output guards, changed-input repeated calls and input preservation.
  `MAT_POW_ARGS=--large` selects dense analytic cases through N=1024 and P=20.
- `nn.cu`: nearest neighbor for FP32 `points[N,3]`, writing one INT32 index per
  point to separate `indices[N]` storage. The
  [LeetGPU contract](https://leetgpu.com/challenges/nearest-neighbor) requires a
  closest point other than the query itself, using Euclidean distance;
  squared distances give the same ordering. N is in [1,100000], with coordinates
  in [-1000,1000]. The statement does not specify tie-breaking, so tests accept
  any non-self index at the minimum distance. N=1 has no such index and no
  documented sentinel: its test checks storage safety only. Run `make run-nn`,
  or select individual/large cases via `NN_ARGS`; see [testing notes](testing.md).
- `batch_norm.cu`: training-style batch normalization of FP32 `input[N,C]`,
  computing statistics independently per column across N rows. Population
  variance divides by N, and `eps` is added inside the square root before
  applying `gamma[C]` and `beta[C]`. There are no running statistics or updates
  to input/parameters; output is a separate overwritten `[N,C]` array.
  The [LeetGPU contract](https://leetgpu.com/challenges/batch-normalization)
  specifies N in [1,10000], C in [1,1024], input in [-100,100], gamma in [0.1,10],
  beta in [-10,10], and eps=1e-5. N=1 and constant channels mathematically yield
  beta. Run `make run-batch-norm`; `BN_ARGS=--large` checks N=5000/10000,C=1024.
- `rms_norm.cu`: RMS normalization of a contiguous FP32 `input[N]` vector,
  writing a separate overwritten `output[N]`. The scalar gamma/beta parameters
  are passed by value. The intended formula is
  `gamma * input[i] / sqrt(sum(input[j]^2)/N + eps) + beta`, reducing over the
  whole vector without subtracting its mean. The
  [LeetGPU contract](https://leetgpu.com/challenges/rms-normalization) specifies
  N in [1,100000], input in [-100,100], gamma in [0.1,10], beta in [-10,10],
  and eps=1e-5. The current float4 loads require 16-byte-aligned input; tests
  preserve that alignment and cover scalar tails and unaligned output pointers.
  Empty inputs, nonfinite values, aliasing and other eps values are not covered.
  Run `make run-rms-norm`; `RMS_ARGS=--large` checks N=99999/100000.
  See [testing notes](testing.md) for validation results and known limitations.
- `group_norm.cu`: Group Normalization of contiguous FP32 `X[N,C,H,W]` in
  NCHW order, writing a separate overwritten `Y` of the same shape. Each
  `(batch, group)` has its own mean and population variance over
  `M=(C/G)*H*W` elements in C/G consecutive channels; statistics never mix
  batches. Apply `gamma[c]*(X-mean)/sqrt(variance+eps)+beta[c]` with per-channel
  FP32 gamma/beta arrays. The
  [LeetGPU contract](https://leetgpu.com/challenges/group-normalization) requires
  N in [1,32], C in [1,1024], H/W in [1,128], `1 <= G <= C`, `C % G == 0`,
  input in [-100,100], gamma in [0.1,10], beta in [-10,10], and eps=1e-5.
  G=1 normalizes each sample over C/H/W; G=C normalizes each channel over H/W.
  A group with one element or constant values mathematically outputs beta.
  Run `make run-group-norm`; `GN_ARGS=--large` checks the challenge performance
  shape and H/W=128. See [testing notes](testing.md) for validation results.
- `layer_norm.cu`: Layer Normalization of contiguous FP32 `input[N,C]`,
  writing a separate overwritten `output[N,C]`. Each row independently uses
  its C features to compute the mean and population variance (divide by C).
  Apply `weight[c]*(input-mean)/sqrt(variance+eps)+bias[c]`; the per-feature
  weight/bias arrays are shared across rows. The
  [LeetGPU contract](https://leetgpu.com/challenges/layer-normalization) specifies
  N in [1,65536], C in [1,4096], input in [-100,100], weight in [0.1,10],
  bias in [-10,10], and eps=1e-5. C=1 and constant rows mathematically yield
  bias. Run `make run-layer-norm`; `LN_ARGS=--large` checks N/C=65536/512
  (the stated performance shape) and 1024/4096 for correctness. The current
  implementation fails the long constant-decimal regression at the local
  numerical tolerance; see [testing notes](testing.md) for details.
- `max_pooling_2d.cu`: FP32 max pooling from contiguous `input[N,C,H,W]` to
  separate overwritten `output[N,C,H_out,W_out]` in NCHW order. Tests use
  `H_out = floor((H + 2*padding - kernel_size) / stride) + 1`, likewise for W,
  with input-window origins `(h_out*stride-padding, w_out*stride-padding)`.
  Input addressing uses the original H/W, independently of output dimensions.
  The [LeetGPU statement](https://leetgpu.com/challenges/2d-max-pooling) gives
  N in [1,100], C in [1,512], H/W in [1,1024], kernel/stride in [1,16], and
  padding in [0,16]. It does not explicitly define the padding value; local
  tests follow standard negative-infinity max-pool padding, with positive
  output dimensions and `padding <= kernel_size/2`. Empty outputs, all-padding
  windows, nonfinite inputs and aliasing are outside the tested contract.
  Run `make run-max-pooling-2d`; `POOL_ARGS=--large` selects N=4,k=3,s=2
  spatial/channel regressions. See [testing notes](testing.md) for limitations.
- `mv.cu`: `A[M,N] * x[N] -> y[M]`, float. `solve` uses a 256-thread block per
  row when N is at least 1024 (M <= 64), 2048 (65 <= M <= 1024), or 4096
  (M > 1024); otherwise it uses one warp per row, eight rows per block.
  These are A800 performance heuristics; other GPUs may favor other thresholds.
  Both paths overwrite y. M <= 0 is a no-op; N=0 writes zeros. M*N must fit int.
  The test compares both kernels with a CPU double reference. `nnz` is a retained,
  unused parameter; this is dense storage, not CSR/COO sparse storage.
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
- `gemm_wmma_tiled_pipeline_aligned_swizzled.cu`: the same FP16 contract and
  full/aligned dispatch conditions, with XOR storage and explicit `ldmatrix`/`mma.sync`
  when BN/BK are powers of two. It retains full-tile WMMA otherwise and writes
  packed `half2` only when C is four-byte aligned, with scalar stores otherwise.
  Default block/warp tiles are 64x64/32x32. See
  [the shared-layout notes](gemm-wmma-shared-layout.md) for tuning and validation.
- `gemm_wmma_tiled_pipeline_multistage.cu`: the same FP16 contract, with deeper
  input and operand buffering for full/aligned tiles. Generic inputs retain a
  fixed 64x64 WMMA pipeline. See the [multistage notes](gemm-wmma-multistage.md)
  for tile selection, alignment, and tuning limits.
- `gemm_wmma_tiled_pipeline_mainloop.cu`: the same FP16 contract, with a BK=64
  mainloop for small, complete, aligned output grids. Generic inputs
  retain the fixed 64x64 WMMA pipeline. See the
  [mainloop notes](gemm-wmma-mainloop.md) for dispatch and tuning.
- `gemm_wmma_tiled_pipeline_reuse.cu`: the same FP16 contract, adding a
  96x128 block / 48x64 warp tile on sufficiently large complete/aligned grids.
  See the [reuse-tile notes](gemm-wmma-reuse.md) for its dispatch boundaries
  and measured benefit from fewer A/B loads.
- `gemm_wmma_tiled_pipeline_epilogue.cu`: the same FP16 contract, with an
  alpha=1/beta=0 template specialization and a general alpha/beta path.
  See the [epilogue notes](gemm-wmma-epilogue.md) for dispatch and measurements.
- `gemm_wmma_tiled_pipeline_large.cu`: the same FP16 contract, allowing larger
  block/warp tiles with dynamic shared input storage above 48 KiB. Unsupported
  shared-memory requirements fall back to the generic pipeline. See the
  [large-tile notes](gemm-wmma-large.md) for configuration and experiments.
- `gemm_wmma_tiled_pipeline_schedule.cu`: the same FP16 contract, with distributed
  asynchronous copies, interleaved operand loads, and compact address generation.
  Measured A800 grid bands select a 128x128 tile with three or four stages;
  other shapes and general alpha/beta retain the previous dispatcher. See the
  [scheduling notes](gemm-wmma-schedule.md) for controls and measurements.
- `gemm.triton.py`: the same row-major FP16 A[M,K], B[K,N], C[M,N] contract,
  FP32 accumulation, and in-place `C = alpha*A*B + beta*C`. Pass contiguous CUDA
  tensors with the stated dimensions. Masks handle M/N/K tails and unaligned
  bases; M/N=0 does no work, K=0 scales C, and beta=0 skips reading old C.
  Seven tile/warp/stage configurations are autotuned. See the
  [Triton integration](gemm-triton.md) for checks and comparison scope.
- `gemm.triton.v2.py` / `gemm.triton.v3.py`: the same contiguous row-major FP16
  contract, with grouped tile order, bounded 32-bit addressing, scalar
  specialization, and a cached JIT launch after CUDA-Graph tuning into separate
  output storage. V2 has 32 candidates and uses newer Triton autotune APIs;
  V3 has 34 candidates, supports Triton 3.1's graph-tuning interface and
  synchronizes input production before first-use tuning. Use `TRITON_SOURCE`
  to select either version in the [shared checks](gemm-triton.md).
- `batched_mm.cu`: FP32 batched multiplication with A[BATCH,M,K], B[BATCH,K,N]
  and C[BATCH,M,N], using contiguous row-major storage without broadcasting or
  transposes. The current kernel computes `C += A*B`; clear C before a standalone
  multiplication, or retain it for accumulation. Tests use CPU double references,
  including consecutive accumulation and unaligned A/B/C. Zero BATCH/M/N is a
  no-op; K=0 does not read A/B and preserves finite C numerically. Dimensions
  are nonnegative and flattened offsets must fit int; input/output storage must
  not overlap.
- `batched_mm_fp16.cu`: the [FP16 batched matrix multiplication challenge](https://leetgpu.com/challenges/fp16-batched-matrix-multiplication)
  uses contiguous row-major half A[BATCH,M,K], B[BATCH,K,N] and C[BATCH,M,N].
  Its contract is `C = A*B`, with FP32 accumulation and a final FP16 conversion;
  the previous contents of C must not contribute. BATCH is 1..128 and M/N/K
  are 1..1024. The source overwrites C with the converted FP32 sum.
  `make run-batched-mm-fp16` checks
  the challenge contract with CPU references, FP16 rounding/FP32 accumulation
  regressions, batch isolation, pointer offsets and output guards.
- `mm_int8.cu`: contiguous row-major signed INT8 A[M,K], B[K,N] and C[M,N].
  Subtract the input zero points and accumulate products in INT64. Convert the
  completed dot product to FP32, multiply by scale_A then scale_B, and divide by
  scale_C. Round to nearest with halfway values rounded to even (`nearbyintf`),
  add zero_point_C, then clamp to [-128,127] and overwrite C. Scales are finite and
  positive, zero points are in [-128,127], and inputs must keep the intermediate
  FP32 arithmetic finite. M/N must be positive, K nonnegative, and storage must
  not overlap between inputs and output. K=0 fills C with zero_point_C without
  reading A/B. See `make run-mm-int8` for exact CPU-reference checks.
- `mc_int.cu`: Monte Carlo integration from supplied FP32 function values
  `y_samples[n_samples]`: the intended result is
  `sum(y_samples) * (b-a) / n_samples`, written to one FP32 output element.
  Tests use positive sample counts, finite values and `a < b`, with separate
  input/output storage and cudaMalloc-aligned inputs. They check the estimate
  against CPU double arithmetic, output overwrites/repeated calls, and input
  preservation. Run `make run-mc-int`; add `MC_INT_ARGS=--large` for the separate
  10-million/100-million-sample suite. See [testing notes](testing.md).
- `lr.cu`: binary logistic regression with contiguous FP32 X[samples,features],
  y[samples] in {0,1}, and overwritten beta[features]. Samples/features must be
  positive, X finite, and inputs/output must not overlap. No intercept is added;
  all coefficients are penalized. The assumed objective is **SUM** of binary
  cross entropy plus `1e-6/2 * ||beta||^2`. This regularization matches the supplied
  separable regression case but is not specified by the public LeetGPU statement;
  it is an explicit local assumption, not a verified platform contract.
  Set the `optimizer` string in `solve` to `"GD"` (default) or `"Newton"`;
  tests override it with the `LR_OPTIMIZER` compiler definition. Both use FP64
  training parameters and gradients, max-absolute column scaling, and `lambda * theta / scale^2`
  for the L2 gradient in scaled coordinates. Newton additionally adds
  `lambda / scale^2` to the Hessian diagonal before the Cholesky solve.
  A gradient-based step search halves steps that cross the directional minimum
  and doubles GD's next trial after acceptance; Newton tries a full step each
  iteration and damps it when needed. Stopping uses the regularized
  gradient in original coefficient coordinates; 100,000 updates/64 backtracks
  remain safety caps, so arbitrary ill-conditioned inputs are not guaranteed to
  converge within those caps. Input buffers are preserved. `make run-lr` runs
  both optimizers against an independent CPU Newton reference, including the
  supplied platform coefficients, zero/duplicate columns, scaling, tails, and
  repeated calls.
- `softmax_3kernel.cu` / `softmax_4kernel.cu`: softmax over one float vector, with
  maximum subtraction for stability. Vectorized paths require 16-byte alignment.
- `attention.cu`: `softmax(Q * K^T / sqrt(d)) * V`, with `Q[M,d]`, `K[N,d]`, `V[N,d]`.
  Uses an intermediate `M*N` allocation. No masking, batching, or head dimension.
- `alibi.cu`: the [ALiBi challenge](https://leetgpu.com/challenges/attention-with-linear-biases)
  computes `softmax(Q*K^T/sqrt(d) + alpha*(i-j)) * V` in FP32 with Q[M,d],
  K[N,d], V[N,d] and output[M,d]. Softmax is row-wise; the signed relative
  position has no absolute value or causal mask, and the bias is not divided
  by sqrt(d). M/N are 1..2048, d is 1..1024 and alpha is a float in [-1,1].
  Output is overwritten and inputs are preserved. Softmax subtracts the maximum
  of the complete biased scores; a row-constant bias shift also limits roundoff.
  `make run-alibi` checks formulas, tails, offsets, repeated calls and exponent
  overflow/underflow regressions against a CPU double reference;
  `ALIBI_ARGS=--large` exercises full dimensions with bounded scores and small
  slopes. See [testing notes](testing.md).

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
