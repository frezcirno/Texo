#include <cuda_runtime.h>

#include "sum_utils.cuh"

template <int BLOCK_SIZE>
__device__ inline void grid_reduce_sum(float val, float *out) {
  val = block_reduce_sum<BLOCK_SIZE>(val);
  if (threadIdx.x == 0) {
    atomicAdd(out, val);
  }
}

__global__ void reduction(const float *input, float *output, int N) {
  int index = blockIdx.x * 1024 + threadIdx.x;
  float val = (index < N) ? input[index] : 0.0f;
  grid_reduce_sum<1024>(val, output);
}

// input, output are device pointers
extern "C" void reduce_my(const float *input, float *output, int N) {
  cudaMemset(output, 0, sizeof(float));
  dim3 gridDim((N + 1023) / 1024, 1, 1);
  dim3 blockDim(1024, 1, 1);
  reduction<<<gridDim, blockDim>>>(input, output, N);
}
