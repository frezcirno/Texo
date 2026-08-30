#include <cuda_runtime.h>

#include "sum_utils.cuh"

template <int BLOCK_SIZE>
__global__ void reduce_v2_kernel(const float *__restrict__ input,
                                 float *__restrict__ output, int N) {
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

  sum = block_reduce_sum<BLOCK_SIZE>(sum);
  if (threadIdx.x == 0)
    atomicAdd(output, sum);
}

extern "C" void reduce_my_v2(const float *input, float *output, int N) {
  cudaMemset(output, 0, sizeof(float));
  constexpr int BLOCK_SIZE = 256;
  // Few hundred blocks: enough to fill A800's 108 SMs, few enough to keep
  // atomicAdd contention on the single output negligible.
  constexpr int GRID_SIZE = 432;
  reduce_v2_kernel<BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE>>>(input, output, N);
}
