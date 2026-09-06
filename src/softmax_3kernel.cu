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

template <int BLOCK_SIZE>
__global__ void exp_sum_kernel(const float *__restrict__ input,
                               const float *__restrict__ maximum,
                               float *__restrict__ output,
                               float *__restrict__ total, int N) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<BLOCK_SIZE>(block);

  int N4 = N / 4;
  const float4 *input4 = reinterpret_cast<const float4 *>(input);
  float4 *output4 = reinterpret_cast<float4 *>(output);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  float local_sum = 0.0f;

  for (int i = tid; i < N4; i += stride) {
    float4 x = input4[i];

    float4 y;
    y.x = __expf(x.x - *maximum);
    y.y = __expf(x.y - *maximum);
    y.z = __expf(x.z - *maximum);
    y.w = __expf(x.w - *maximum);

    output4[i] = y;
    local_sum += y.x + y.y + y.z + y.w;
  }

  int tail_base = N4 * 4;
  int tail_index = tail_base + tid;
  if (tail_index < N) {
    float value = __expf(input[tail_index] - *maximum);
    output[tail_index] = value;
    local_sum += value;
  }

  local_sum = cg::reduce(tile, local_sum, cg::plus<float>());

  if (tile.thread_rank() == 0) {
    atomicAdd(total, local_sum);
  }
}

template <int BLOCK_SIZE>
__global__ void normalize_kernel(float *__restrict__ output,
                                 const float *__restrict__ total, int N) {
  __shared__ float inverse_total;

  int N4 = N / 4;
  float4 *output4 = reinterpret_cast<float4 *>(output);

  if (threadIdx.x == 0) {
    inverse_total = 1.0f / total[0];
  }
  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = tid; i < N4; i += stride) {
    float4 value = output4[i];
    value.x *= inverse_total;
    value.y *= inverse_total;
    value.z *= inverse_total;
    value.w *= inverse_total;
    output4[i] = value;
  }

  int tail_index = N4 * 4 + tid;
  if (tail_index < N) {
    output[tail_index] *= inverse_total;
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, float *output, int N) {
  if (N <= 0)
    return;

  constexpr int BLOCK_SIZE = 256;
  constexpr int MAX_BLOCKS = 432;

  int max_blocks = (N / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
  max_blocks = max_blocks < 1 ? 1 : max_blocks;
  max_blocks = max_blocks > MAX_BLOCKS ? MAX_BLOCKS : max_blocks;

  int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  blocks = blocks > MAX_BLOCKS ? MAX_BLOCKS : blocks;

  float *maximum_and_total;
  cudaMalloc(&maximum_and_total, 2 * sizeof(float));
  cudaMemcpyAsync(&maximum_and_total[0], input, sizeof(float),
                  cudaMemcpyDeviceToDevice);
  cudaMemsetAsync(&maximum_and_total[1], 0, sizeof(float));

  max_kernel<BLOCK_SIZE>
      <<<max_blocks, BLOCK_SIZE>>>(input, &maximum_and_total[0], N);
  exp_sum_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
      input, &maximum_and_total[0], output, &maximum_and_total[1], N);
  normalize_kernel<BLOCK_SIZE>
      <<<blocks, BLOCK_SIZE>>>(output, &maximum_and_total[1], N);

  cudaFree((void *)maximum_and_total);

  cudaDeviceSynchronize();
}
