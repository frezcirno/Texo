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
__global__ void sum_kernel(const T *__restrict__ input1, // (N, C)
                           const T *__restrict__ input2, // (N, C)
                           T *__restrict__ output,       // (N,)
                           const size_t N, const size_t C) {
  const size_t n = blockIdx.y;
  const size_t tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const size_t stride = gridDim.x * BLOCK_SIZE;

  T sum = 0;

  for (int i = tid; i < C; i += stride) {
    sum += pow2(input1[n * C + i] + input2[n * C + i]);
  }

  sum = block_sum<BLOCK_SIZE>(sum);
  if (threadIdx.x == 0) {
    atomicAdd(&output[n], sum);
  }
}

template <typename T>
__global__ void scale_kernel(const T *__restrict__ input1, // (N, C)
                             const T *__restrict__ input2, // (N, C)
                             const T *__restrict__ weight, // (C,)
                             T *__restrict__ rms,          // (N,)
                             T *__restrict__ output,       // (N, C)
                             const size_t N, const size_t C, float eps) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= C)
    return;
  const size_t n = blockIdx.y;
  output[n * C + tid] = weight[tid] *
                        (input1[n * C + tid] + input2[n * C + tid]) /
                        sqrt(rms[n] / C + eps);
}
// x, residual, weight, out are device pointers
extern "C" void solve(const float *x,        // (N, C)
                      const float *residual, // (N, C)
                      const float *weight,   // (C,)
                      float *out,            // (N, C)
                      int N, int C, float eps) {
  float *rms; // (N,)
  cudaMalloc(&rms, N * sizeof(float));
  cudaMemset(rms, 0, N * sizeof(float));

  // rms = sum(input^2)
  sum_kernel<256><<<dim3((C + 255) / 256, N), 256>>>(x, residual, rms, N, C);

  // y = gamma * x / sqrt(rms/N + eps) + beta
  scale_kernel<<<dim3((C + 255) / 256, N), 256>>>(x, residual, weight, rms, out,
                                                  N, C, eps);

  cudaFree(rms);
}
