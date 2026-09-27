#include <cuda_runtime.h>
#include <math.h>

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

template <int BLOCK_SIZE>
__global__ void sum_kernel(const float *__restrict__ input,
                           float *__restrict__ output, double alpha, int N) {
  const size_t tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const size_t stride = gridDim.x * BLOCK_SIZE;

  double sum = 0.0f;

  int N4 = N / 4;
  const float4 *input4 = reinterpret_cast<const float4 *>(input);
  for (int i = tid; i < N4; i += stride) {
    float4 v = input4[i];
    sum += alpha * (v.x + v.y + v.z + v.w);
  }
  int tail_base = N4 * 4;
  if (tail_base + tid < N) {
    sum += alpha * input[tail_base + tid];
  }

  sum = block_sum<BLOCK_SIZE>(sum);
  if (threadIdx.x == 0) {
    atomicAdd(output, sum);
  }
}

// Monte Carlo Integration
// y_samples, result are device pointers
extern "C" void solve(const float *y_samples, // (n_samples,)
                      float *result,          // (1,)
                      float a, float b, int n_samples) {
  // sum(y_samples) * (b - a) / n_samples
  sum_kernel<256><<<(n_samples + 255) / 256, 256>>>(
      y_samples, result, (b - a) / n_samples, n_samples);
}
