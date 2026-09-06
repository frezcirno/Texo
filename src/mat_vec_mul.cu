#include <cuda_runtime.h>

__device__ inline float warp_sum(float val) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, off);
  }
  return val;
}

template <int BLOCK_SIZE> __device__ inline float block_sum(float val) {
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ float warp_sums[NUM_WARPS];
  int lane = threadIdx.x % 32;
  int warp_idx = threadIdx.x / 32;
  val = warp_sum(val);
  if (lane == 0)
    warp_sums[warp_idx] = val;
  __syncthreads();
  if (warp_idx == 0) {
    val = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
    val = warp_sum(val);
  }
  return val;
}

template <int BLOCK_SIZE>
__global__ void mat_vec_mul(const float *__restrict__ A, // (M, N)
                            const float *__restrict__ B, // (N,)
                            float *__restrict__ output,  // (M,)
                            int M, int N, int nnz) {
  const int row = blockIdx.x;
  float sum = 0.0f;

  // 同一个 block 的线程分担这一行的列
  for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
    sum += A[row * N + i] * B[i];
  }

  sum = block_sum<BLOCK_SIZE>(sum);

  if (threadIdx.x == 0) {
    output[row] = sum;
  }
}

__global__ void mat_vec_mul_warp(const float *__restrict__ A,
                                 const float *__restrict__ x,
                                 float *__restrict__ y, int M, int N) {

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x / 32;
  const int row = blockIdx.x * warps_per_block + warp;

  // 同一个 warp 的 row 相同，因此整个 warp 一起退出
  if (row >= M) {
    return;
  }

  float sum = 0.0f;
  for (int i = lane; i < N; i += 32) {
    sum += A[row * N + i] * x[i];
  }

  sum = warp_sum(sum);

  if (lane == 0) {
    y[row] = sum;
  }
}

// A, x, y are device pointers
extern "C" void solve(const float *A, // (M, N)
                      const float *x, // (N,)
                      float *y,       // (M,)
                      int M, int N, int nnz) {
  if (M <= 0) {
    return;
  }
  // The test also benchmarks mat_vec_mul<256> (one block per row).
  mat_vec_mul_warp<<<1 + (M - 1) / 8, 256>>>(A, x, y, M, N);
}
