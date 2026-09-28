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
__global__ void count_kernel(const int *input, // (N,)
                             int *output,      // (1,)
                             int N, int K) {
  //
  int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int stride = gridDim.x * BLOCK_SIZE;

  int sum = 0;

  int N4 = N / 4;
  const int4 *input4 = reinterpret_cast<const int4 *>(input);
  for (int i = tid; i < N4; i += stride) {
    int4 v = input4[i];
    sum += (v.x == K) + (v.y == K) + (v.z == K) + (v.w == K);
  }
  int tail_base = N4 * 4;
  if (tail_base + tid < N) {
    sum += input[tail_base + tid] == K;
  }

  sum = block_sum<BLOCK_SIZE>(sum);
  if (threadIdx.x == 0) {
    atomicAdd(output, sum);
  }
}

// input, output are device pointers
extern "C" void solve(const int *input, // (N, M)
                      int *output,      // (1,)
                      int N, int M, int K) {
  const size_t NN = N * M;
  count_kernel<256><<<(NN + 255) / 256, 256>>>(input, output, NN, K);
}
