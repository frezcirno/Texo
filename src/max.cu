#include <cuda_runtime.h>
#include <math.h>

__device__ inline float warp_reduce_max(float val) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    val = max(val, __shfl_down_sync(0xffffffff, val, off));
  }
  return val;
}

template <int BLOCK_SIZE> __device__ inline float block_reduce_max(float val) {
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ float warp_maxes[NUM_WARPS];
  int lane = threadIdx.x % 32;
  int warp_idx = threadIdx.x / 32;
  val = warp_reduce_max(val);
  if (lane == 0) {
    warp_maxes[warp_idx] = val;
  }
  __syncthreads();
  if (warp_idx == 0) {
    val = (lane < NUM_WARPS) ? warp_maxes[lane] : -INFINITY;
    val = warp_reduce_max(val);
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
__global__ void _max_v2_kernel(const float *__restrict__ input,
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

  max_res = block_reduce_max<BLOCK_SIZE>(max_res);
  if (threadIdx.x == 0)
    atomic_max_float(output, max_res);
}

extern "C" void max_kernel(const float *input, float *output, int N) {
  const float neg_inf = -INFINITY;
  cudaMemcpy(output, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
  constexpr int BLOCK_SIZE = 256;
  // Few hundred blocks: enough to fill A800's 108 SMs, few enough to keep
  // atomic contention on the single output negligible.
  constexpr int GRID_SIZE = 432;
  _max_v2_kernel<BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE>>>(input, output, N);
}
