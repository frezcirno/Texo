#include <cuda_runtime.h>

__device__ float pow2(float x) { return x * x; }

template <typename T> __device__ inline T warp_sum(T x) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    x += __shfl_down_sync(0xFFFFFFFF, x, off);
  }
  return x;
}

template <size_t BLOCK_SIZE, typename T> __device__ T block_sum(T x) {
  __shared__ T warp_sums[BLOCK_SIZE / 32];
  const size_t lane = threadIdx.x % 32;
  const size_t warp = threadIdx.x / 32;
  x = warp_sum(x);
  if (lane == 0) {
    warp_sums[warp] = x;
  }
  __syncthreads();
  if (warp == 0) {
    x = (lane < BLOCK_SIZE / 32) ? warp_sums[lane] : 0;
    x = warp_sum(x);
  }
  return x;
}

template <size_t BLOCK_SIZE>
__global__ void layer_norm_kernel(const float *__restrict__ input,  // (N, C)
                                  const float *__restrict__ weight, // (C,)
                                  const float *__restrict__ bias,   // (C,)
                                  float *__restrict__ output,       // (N, C)
                                  int N, int C, float eps) {
  const size_t n = blockIdx.x;
  // mu, sigma = f(input)
  // xhat = norm(input)
  // y = gamma * xhat + beta

  float mu = 0;
  for (int c = threadIdx.x; c < C; c += BLOCK_SIZE) {
    mu += input[(n * C) + c];
  }
  mu = block_sum<BLOCK_SIZE>(mu);
  __shared__ float g_mu;
  if (threadIdx.x == 0) {
    g_mu = mu / C;
  }
  __syncthreads();
  mu = g_mu;

  float sigma2 = 0;
  for (int c = threadIdx.x; c < C; c += BLOCK_SIZE) {
    sigma2 += pow2(input[(n * C) + c] - mu);
  }
  sigma2 = block_sum<BLOCK_SIZE>(sigma2);
  __shared__ float g_sigma2;
  if (threadIdx.x == 0) {
    g_sigma2 = sigma2 / C;
  }
  __syncthreads();
  sigma2 = g_sigma2;

  for (int c = threadIdx.x; c < C; c += BLOCK_SIZE) {
    output[(n * C) + c] =
        weight[c] * (input[(n * C) + c] - mu) / sqrt(sigma2 + eps) + bias[c];
  }
}

// X, gamma, beta, Y are device pointers
extern "C" void solve(const float *__restrict__ input,  // (N, C)
                      const float *__restrict__ weight, // (C,)
                      const float *__restrict__ bias,   // (C,)
                      float *__restrict__ output,       // (N, C)
                      int N, int C, float eps) {
  // mu, sigma = f(input)
  // xhat = norm(input)
  // y = gamma * xhat + beta
  // one group one block
  layer_norm_kernel<256><<<N, 256>>>(input, weight, bias, output, N, C, eps);
}
