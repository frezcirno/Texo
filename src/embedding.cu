#include <cuda_runtime.h>

template <typename T> __device__ T pow2(T x) { return x * x; }

template <typename T> __device__ inline T warp_sum(T val) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, off);
  }
  return val;
}

template <int BLOCK_SIZE, typename T> __device__ inline T block_sum(T val) {
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ T warp_sums[NUM_WARPS];
  size_t lane = threadIdx.x % 32;
  size_t warp_idx = threadIdx.x / 32;
  val = warp_sum(val);
  if (lane == 0) {
    warp_sums[warp_idx] = val;
  }
  __syncthreads();
  if (warp_idx == 0) {
    val = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
    val = warp_sum(val);
  }
  return val;
}

template <typename T> struct Vector4;
template <> struct Vector4<float> {
  using type = float4;
};
template <> struct Vector4<int> {
  using type = int4;
};
template <> struct Vector4<double> {
  using type = double4;
};

template <int BLOCK_SIZE, typename T>
__global__ void sum_kernel(const T *__restrict__ input, T *__restrict__ output,
                           int N, float eps) {
  int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int stride = gridDim.x * BLOCK_SIZE;

  using float4 = typename Vector4<T>::type;
  T sum = 0;

  int N4 = N / 4;
  const auto *input4 = reinterpret_cast<const float4 *>(input);
  for (int i = tid; i < N4; i += stride) {
    float4 v = input4[i];
    sum += pow2(v.x) + pow2(v.y) + pow2(v.z) + pow2(v.w);
  }
  int tail_base = N4 * 4;
  if (tail_base + tid < N) {
    sum += pow2(input[tail_base + tid]);
  }

  sum = block_sum<BLOCK_SIZE>(sum);
  if (threadIdx.x == 0) {
    atomicAdd(output, sum);
  }
}

template <typename T>
__global__ void scale_kernel(const T *__restrict__ input, // (N,)
                             T *__restrict__ rms,         // (1,)
                             float gamma, float beta,
                             T *__restrict__ output, // (N,)
                             int N, float eps) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
    return;
  output[tid] = gamma * input[tid] / sqrt(*rms / N + eps) + beta;
}

__global__ void embedding(const int *token_ids,             // (B, T)
                          const int *position_ids,          // (T,)
                          const float *token_embeddings,    // (V, D)
                          const float *position_embeddings, // (P, D)
                          const float *gamma,               // (D,)
                          const float *beta,                // (D,)
                          float *output,                    // (B, T, D)
                          const size_t B, const size_t T, const size_t V,
                          const size_t P, const size_t D, const float eps) {
  //
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= B * T)
    return;
  const size_t b = tid / T;
  const size_t t = tid % T;

  const int token_id = token_ids[b * T + t];
  const int position_id = position_ids[t];
  const size_t token_offset = static_cast<size_t>(token_id) * D;
  const size_t position_offset = static_cast<size_t>(position_id) * D;

  float mu = 0;
  for (int d = 0; d < D; d++) {
    float v = token_embeddings[token_offset + d] +
              position_embeddings[position_offset + d];
    mu += v;
  }
  mu /= D;

  float sigma2 = 0;
  for (int d = 0; d < D; d++) {
    float v = token_embeddings[token_offset + d] +
              position_embeddings[position_offset + d];
    sigma2 += pow2(v - mu);
  }
  sigma2 /= D;

  for (int d = 0; d < D; d++) {
    output[(b * T + t) * D + d] =
        gamma[d] *
            (token_embeddings[token_offset + d] +
             position_embeddings[position_offset + d] - mu) *
            rsqrt(sigma2 + eps) +
        beta[d];
  }
}

// token_ids, position_ids, token_embeddings, position_embeddings, gamma, beta,
// output are device pointers
extern "C" void solve(const int *token_ids,             // (B, T)
                      const int *position_ids,          // (T,)
                      const float *token_embeddings,    // (V, D)
                      const float *position_embeddings, // (P, D)
                      const float *gamma,               // (D,)
                      const float *beta,                // (D,)
                      float *output,                    // (B, T, D)
                      int B, int T, int V, int P, int D, float eps) {
  constexpr int BLOCK_SIZE = 256;
  const int num_blocks = (B * T + BLOCK_SIZE - 1) / BLOCK_SIZE;
  embedding<<<num_blocks, BLOCK_SIZE>>>(
      token_ids, position_ids, token_embeddings, position_embeddings, gamma,
      beta, output, B, T, V, P, D, eps);
}
