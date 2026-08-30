#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

template <int BLOCK_SIZE>
__global__ void reduce_cg_kernel(const float *__restrict__ input, 
                                 double *__restrict__ accumulator, int N) {
  cg::thread_block block = cg::this_thread_block();

  cg::thread_block_tile<BLOCK_SIZE> tile =
      cg::tiled_partition<BLOCK_SIZE>(block);

  int tid = blockIdx.x * BLOCK_SIZE + tile.thread_rank();
  int stride = gridDim.x * BLOCK_SIZE;

  double sum = 0.0;

  // 标量读取仍然是合并访存，并且没有 float4 对齐要求
  for (int i = tid; i < N; i += stride) {
    sum += static_cast<double>(input[i]);
  }

  sum = cg::reduce(tile, sum, cg::plus<double>());

  if (tile.thread_rank() == 0) {
    atomicAdd(accumulator, sum);
  }
}

__global__ void convert_result_kernel(const double *accumulator,
                                      float *output) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    output[0] = static_cast<float>(accumulator[0]);
  }
}

extern "C" void solve(const float *input, float *output, int N) {
  constexpr int BLOCK_SIZE = 256;
  constexpr int NUM_BLOCKS = 432;

  double *accumulator = nullptr;

  cudaMalloc(reinterpret_cast<void **>(&accumulator), sizeof(double));

  cudaMemset(accumulator, 0, sizeof(double));

  reduce_cg_kernel<BLOCK_SIZE>
      <<<NUM_BLOCKS, BLOCK_SIZE>>>(input, accumulator, N);

  convert_result_kernel<<<1, 1>>>(accumulator, output);

  cudaFree(accumulator);
}