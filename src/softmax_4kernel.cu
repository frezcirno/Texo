#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

__device__ inline float warp_max(float val) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    val = max(val, __shfl_down_sync(0xffffffff, val, off));
  }
  return val;
}

template <int BLOCK_SIZE> __device__ inline float block_max(float val) {
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ float warp_maxes[NUM_WARPS];
  int lane = threadIdx.x % 32;
  int warp_idx = threadIdx.x / 32;
  val = warp_max(val);
  if (lane == 0) {
    warp_maxes[warp_idx] = val;
  }
  __syncthreads();
  if (warp_idx == 0) {
    val = (lane < NUM_WARPS) ? warp_maxes[lane] : -INFINITY;
    val = warp_max(val);
  }
  return val;
}

__device__ inline float atomic_max_float(float *addr, float val) {
  int *addr_as_int = reinterpret_cast<int *>(addr);
  int old = *addr_as_int;
  int assumed;
  do {
    assumed = old;
    if (val <= __int_as_float(assumed))
      break;
    old = atomicCAS(addr_as_int, assumed, __float_as_int(val));
  } while (assumed != old);
  return __int_as_float(old);
}

template <int BLOCK_SIZE>
__global__ void max_kernel(const float *__restrict__ input,
                           float *__restrict__ output, int N) {
  int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int stride = gridDim.x * BLOCK_SIZE;

  float max_res = -INFINITY;

  int N4 = N / 4;
  const float4 *input4 = reinterpret_cast<const float4 *>(input);
  for (int i = tid; i < N4; i += stride) {
    float4 v = input4[i];
    max_res = max(max_res, v.x);
    max_res = max(max_res, v.y);
    max_res = max(max_res, v.z);
    max_res = max(max_res, v.w);
  }
  int tail_base = N4 * 4;
  if (tail_base + tid < N) {
    max_res = max(max_res, input[tail_base + tid]);
  }

  max_res = block_max<BLOCK_SIZE>(max_res);
  if (threadIdx.x == 0)
    atomic_max_float(output, max_res);
}

__global__ void exp_kernel(const float *__restrict__ input,
                           const float *__restrict__ maximum,
                           float *__restrict__ output, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
    return;
  output[tid] = expf(input[tid] - maximum[0]);
}

template <int BLOCK_SIZE>
__global__ void sum_kernel(const float *__restrict__ output,
                           float *__restrict__ total, int N) {
  __shared__ cg::block_tile_memory<BLOCK_SIZE> scratch;
  auto block = cg::this_thread_block(scratch);
  auto tile = cg::tiled_partition<BLOCK_SIZE>(block);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float local_sum = tid < N ? output[tid] : 0.0f;
  local_sum = cg::reduce(tile, local_sum, cg::plus<float>());
  if (tile.thread_rank() == 0) {
    atomicAdd(total, local_sum);
  }
}

template <int BLOCK_SIZE>
__global__ void normalize_kernel(float *__restrict__ output,
                                 const float *__restrict__ total, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
    return;
  output[tid] = output[tid] / total[0];
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, float *output, int N) {
  if (N <= 0)
    return;
  constexpr int BLOCK_SIZE = 256;

  float *maximum;
  cudaMalloc(&maximum, sizeof(float));
  float neg_inf = -INFINITY;
  cudaMemcpy(maximum, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);

  int blocksPerGridN4 = (N / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
  if (blocksPerGridN4 == 0) {
    blocksPerGridN4 = 1;
  }
  max_kernel<BLOCK_SIZE><<<blocksPerGridN4, BLOCK_SIZE>>>(input, maximum, N);

  int blocksPerGrid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  exp_kernel<<<blocksPerGrid, BLOCK_SIZE>>>(input, maximum, output, N);

  cudaFree((void *)maximum);

  float *total;
  cudaMalloc(&total, sizeof(float));
  cudaMemset(total, 0, sizeof(float));

  sum_kernel<BLOCK_SIZE><<<blocksPerGrid, BLOCK_SIZE>>>(output, total, N);

  normalize_kernel<BLOCK_SIZE><<<blocksPerGrid, BLOCK_SIZE>>>(output, total, N);

  cudaFree((void *)total);

  cudaDeviceSynchronize();
}
