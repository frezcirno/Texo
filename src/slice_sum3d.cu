#include <cuda_runtime.h>

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
__global__ void sum_kernel(const int *__restrict__ input, // (N, M, K)
                           int *__restrict__ output,      // (1,)
                           const dim3 size, const dim3 stride) {
  const size_t tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const size_t tstride = gridDim.x * BLOCK_SIZE;
  const size_t N = size.z * size.y * size.x;

  int sum = 0;

  for (int i = tid; i < N; i += tstride) {
    const dim3 pos(i % size.x, (i / size.x) % size.y, i / size.x / size.y);
    sum += input[pos.z * stride.z + pos.y * stride.y + pos.x];
  }

  sum = block_sum<BLOCK_SIZE>(sum);
  if (threadIdx.x == 0) {
    atomicAdd(output, sum);
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int *input, // (N, M, K)
                      int *output,      // (1,)
                      int N, int M, int K, int S_DEP, int E_DEP, int S_ROW,
                      int E_ROW, int S_COL, int E_COL) {
  const dim3 size(E_COL - S_COL + 1, E_ROW - S_ROW + 1, E_DEP - S_DEP + 1);
  const dim3 stride(1, K, M * K);
  sum_kernel<256><<<(size.x * size.y * size.z + 255) / 256, 256>>>(
      input + S_DEP * stride.z + S_ROW * stride.y + S_COL, output, size,
      stride);
}
