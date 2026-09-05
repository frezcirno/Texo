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
__global__ void sum(const float *__restrict__ input, float *__restrict__ output,
                    int N) {
  int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int stride = gridDim.x * BLOCK_SIZE;

  float sum = 0.0f;

  int N4 = N / 4;
  const float4 *input4 = reinterpret_cast<const float4 *>(input);
  for (int i = tid; i < N4; i += stride) {
    float4 v = input4[i];
    sum += v.x + v.y + v.z + v.w;
  }
  int tail_base = N4 * 4;
  if (tail_base + tid < N) {
    sum += input[tail_base + tid];
  }

  sum = block_sum<BLOCK_SIZE>(sum);
  if (threadIdx.x == 0)
    atomicAdd(output, sum);
}

extern "C" void solve(const float *input, float *output, int N) {
  constexpr int BLOCK_SIZE = 256;
  // Few hundred blocks: enough to fill A800's 108 SMs, few enough to keep
  // atomicAdd contention on the single output negligible.
  constexpr int GRID_SIZE = 432;
  sum<BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE>>>(input, output, N);
}
