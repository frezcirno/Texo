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
  using T4 = typename Vector4<T>::type;

  int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int stride = gridDim.x * BLOCK_SIZE;

  T sum = 0;

  int N4 = N / 4;
  const auto *input4 = reinterpret_cast<const T4 *>(input);
  for (int i = tid; i < N4; i += stride) {
    T4 v = input4[i];
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

// input, gamma, beta, output are device pointers
extern "C" void solve(const float *input, // (N,)
                      float gamma, float beta,
                      float *output, // (N,)
                      int N, float eps) {
  float *rms;
  cudaMalloc(&rms, sizeof(float));
  cudaMemset(rms, 0, sizeof(float));

  // rms = sum(input^2)
  sum_kernel<256><<<(N + 255) / 256, 256>>>(input, rms, N, eps);

  // y = gamma * x / sqrt(rms/N + eps) + beta
  scale_kernel<<<(N + 255) / 256, 256>>>(input, rms, gamma, beta, output, N,
                                         eps);

  cudaFree(rms);
}
