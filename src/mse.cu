#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

template <size_t BLOCK_SIZE>
__global__ void mse_kernel(const float *predictions, const float *targets,
                           double *block_sums, int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  double result =
      tid >= N ? 0.0 : double(predictions[tid]) - double(targets[tid]);
  result *= result;

  __shared__ cg::block_tile_memory<BLOCK_SIZE> scratch;
  auto block = cg::this_thread_block(scratch);
  auto tile = cg::tiled_partition<BLOCK_SIZE>(block);
  result = cg::reduce(tile, result, cg::plus<double>());

  if (threadIdx.x == 0)
    block_sums[blockIdx.x] = result;
}

// 分层求和降低舍入误差
template <size_t BLOCK_SIZE>
__global__ void finish_mse(const double *block_sums, float *mse, int num_blocks,
                           int N) {
  double sum = 0.0;
  for (int i = threadIdx.x; i < num_blocks; i += BLOCK_SIZE)
    sum += block_sums[i];

  __shared__ cg::block_tile_memory<BLOCK_SIZE> scratch;
  auto block = cg::this_thread_block(scratch);
  auto tile = cg::tiled_partition<BLOCK_SIZE>(block);
  sum = cg::reduce(tile, sum, cg::plus<double>());
  if (threadIdx.x == 0)
    mse[0] = sum / N;
}

// predictions, targets, mse are device pointers
extern "C" void solve(const float *predictions, const float *targets,
                      float *mse, int N) {
  if (N <= 0) {
    cudaMemsetAsync(mse, 0, sizeof(float));
    return;
  }
  const int num_blocks = 1 + (N - 1) / 256;
  double *block_sums = nullptr;
  cudaMalloc(&block_sums, size_t(num_blocks) * sizeof(double));
  mse_kernel<256><<<num_blocks, 256>>>(predictions, targets, block_sums, N);
  finish_mse<256><<<1, 256>>>(block_sums, mse, num_blocks, N);
  cudaFree(block_sums);
}
