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
__global__ void group_norm_kernel(const float *__restrict__ X, // (N, C, H, W)
                                  const float *__restrict__ gamma, // (C,)
                                  const float *__restrict__ beta,  // (C,)
                                  float *__restrict__ output, // (N, C, H, W)
                                  int N, int C, int H, int W, int G,
                                  float eps) {
  const size_t group = blockIdx.x % G;
  const size_t n = blockIdx.x / G;
  const size_t c_start = group * C / G;
  const size_t c_end = (group + 1) * C / G;
  const size_t work = (C / G) * H * W;
  const size_t work_start = threadIdx.x * work / BLOCK_SIZE;
  const size_t work_end = (threadIdx.x + 1) * work / BLOCK_SIZE;
  // mu, sigma = f(input)
  // xhat = norm(input)
  // y = gamma * xhat + beta

  float mu = 0;
  for (int i = work_start; i < work_end; i++) {
    const size_t w = i % W;
    const size_t h = i / W % H;
    const size_t c = i / W / H + c_start;
    mu += X[(((n * C) + c) * W + w) * H + h];
  }
  mu = block_sum<BLOCK_SIZE>(mu);
  __shared__ float g_mu;
  if (threadIdx.x == 0) {
    g_mu = mu / work;
  }
  __syncthreads();
  mu = g_mu;

  float sigma2 = 0;
  for (int i = work_start; i < work_end; i++) {
    const size_t w = i % W;
    const size_t h = i / W % H;
    const size_t c = i / W / H + c_start;
    sigma2 += pow2(X[(((n * C) + c) * W + w) * H + h] - mu);
  }
  sigma2 = block_sum<BLOCK_SIZE>(sigma2);
  __shared__ float g_sigma2;
  if (threadIdx.x == 0) {
    g_sigma2 = sigma2 / work;
  }
  __syncthreads();
  sigma2 = g_sigma2;

  for (int i = work_start; i < work_end; i++) {
    const size_t w = i % W;
    const size_t h = i / W % H;
    const size_t c = i / W / H + c_start;
    output[(((n * C) + c) * W + w) * H + h] =
        gamma[c] * (X[(((n * C) + c) * W + w) * H + h] - mu) /
            sqrt(sigma2 + eps) +
        beta[c];
  }
}

// X, gamma, beta, Y are device pointers
extern "C" void solve(const float *X,     // (N, C, H, W)
                      const float *gamma, // (C,)
                      const float *beta,  // (C,)
                      float *Y,           // (N, C, H, W)
                      int N, int C, int H, int W, int G, float eps) {
  // mu, sigma = f(input)
  // xhat = norm(input)
  // y = gamma * xhat + beta
  // one group one block
  group_norm_kernel<256><<<N * G, 256>>>(X, gamma, beta, Y, N, C, H, W, G, eps);
}
