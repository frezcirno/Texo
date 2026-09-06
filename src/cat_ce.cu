#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

__global__ void loss_kernel(const float *__restrict__ logits,    // (N, C)
                            const int *__restrict__ true_labels, // (N,)
                            float *__restrict__ item_loss,       // (N,)
                            int N, int C) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
    return;
  float max_logit = -INFINITY;
  for (int i = 0; i < C; ++i)
    max_logit = fmaxf(max_logit, logits[tid * C + i]);

  float sum_exp = 0.0f;
  for (int i = 0; i < C; ++i)
    sum_exp += expf(logits[tid * C + i] - max_logit);

  float true_logits = logits[tid * C + true_labels[tid]];
  item_loss[tid] = logf(sum_exp) + (max_logit - true_logits);
}

namespace cg = cooperative_groups;

template <size_t BLOCK_SIZE>
__global__ void sum_kernel(const float *__restrict__ item_loss, // (N,)
                           float *__restrict__ loss,            // (1,)
                           int N, int C) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto my_loss = tid >= N ? 0 : item_loss[tid];
  __shared__ cg::block_tile_memory<BLOCK_SIZE> scratch;
  auto block = cg::this_thread_block(scratch);
  auto tile = cg::tiled_partition<BLOCK_SIZE>(block);
  auto all_loss = cg::reduce(tile, my_loss, cg::plus<float>());
  if (threadIdx.x == 0)
    atomicAdd(loss, all_loss / N);
}

// logits, true_labels, loss are device pointers
extern "C" void solve(const float *logits,    // (N, C)
                      const int *true_labels, // (N,)
                      float *loss,            // (1,)
                      int N, int C) {
  if (N == 0)
    return;
  float *item_loss;
  cudaMalloc(&item_loss, N * sizeof(float));
  cudaMemsetAsync(loss, 0, sizeof(float));
  loss_kernel<<<(N + 255) / 256, 256>>>(logits, true_labels, item_loss, N, C);
  sum_kernel<256><<<(N + 255) / 256, 256>>>(item_loss, loss, N, C);
  cudaFree(item_loss);
}
