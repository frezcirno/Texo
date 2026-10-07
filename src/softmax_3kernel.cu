#include <cmath>
#include <cuda_runtime.h>

template <typename T> struct Add {
  __host__ __device__ static constexpr T identity() { return T(0); }
  __device__ static T apply(T a, T b) { return a + b; }
  __device__ static void atomic_apply(T *address, T value) {
    atomicAdd(address, value);
  }
};
template <typename T> struct Max {
  __host__ __device__ static T identity() { return -INFINITY; }
  __device__ static T apply(T a, T b) { return max(a, b); }
  __device__ static void atomic_apply(T *address, T value) {
    atomicMax(address, value);
  }
};

template <template <typename> class Op = Max, typename T>
__device__ inline T warp_reduce(T val) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    val = Op<T>::apply(val, __shfl_down_sync(0xffffffff, val, off));
  }
  return val;
}

template <int BLOCK_SIZE, template <typename> class Op = Max, typename T>
__device__ inline T block_reduce(T val) {
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ T warp_sums[NUM_WARPS];
  const int lane = threadIdx.x % 32;
  const int warp_idx = threadIdx.x / 32;
  val = warp_reduce<Op, T>(val);
  if (lane == 0) {
    warp_sums[warp_idx] = val;
  }
  __syncthreads();
  if (warp_idx == 0) {
    val = (lane < NUM_WARPS) ? warp_sums[lane] : Op<T>::identity();
    val = warp_reduce<Op, T>(val);
  }
  return val;
}

__device__ inline void atomic_max_float(float *addr, float val) {
  if (val >= 0) {
    atomicMax((int *)addr, __float_as_int(val));
  } else {
    atomicMin((unsigned *)addr, __float_as_uint(val));
  }
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

  max_res = block_reduce<BLOCK_SIZE, Max>(max_res);
  if (threadIdx.x == 0)
    atomic_max_float(output, max_res);
}

template <int BLOCK_SIZE>
__global__ void exp_sum_kernel(const float *__restrict__ input,
                               const float *__restrict__ maximum,
                               float *__restrict__ output,
                               float *__restrict__ total, int N) {
  int N4 = N / 4;
  const float4 *input4 = reinterpret_cast<const float4 *>(input);
  float4 *output4 = reinterpret_cast<float4 *>(output);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  float local_sum = 0.0f;
  const float m = *maximum;

  for (int i = tid; i < N4; i += stride) {
    float4 x = input4[i];

    float4 y;
    y.x = __expf(x.x - m);
    y.y = __expf(x.y - m);
    y.z = __expf(x.z - m);
    y.w = __expf(x.w - m);

    output4[i] = y;
    local_sum += y.x + y.y + y.z + y.w;
  }

  int tail_base = N4 * 4;
  int tail_index = tail_base + tid;
  if (tail_index < N) {
    float value = __expf(input[tail_index] - m);
    output[tail_index] = value;
    local_sum += value;
  }

  local_sum = block_reduce<BLOCK_SIZE, Add>(local_sum);

  if (threadIdx.x == 0) {
    atomicAdd(total, local_sum);
  }
}

template <int BLOCK_SIZE>
__global__ void normalize_kernel(float *__restrict__ output,
                                 const float *__restrict__ total, int N) {
  const float inverse_total = 1.0f / total[0];

  int N4 = N / 4;
  float4 *output4 = reinterpret_cast<float4 *>(output);

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

__device__ float g_stats[2];

__global__ void init_kernel(float *stats) {
  stats[0] = -INFINITY;
  stats[1] = 0;
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

  static float *stats = nullptr;
  if (!stats) {
    cudaGetSymbolAddress((void **)&stats, g_stats);
  }

  init_kernel<<<1, 1>>>(stats);
  max_kernel<BLOCK_SIZE><<<max_blocks, BLOCK_SIZE>>>(input, &stats[0], N);
  exp_sum_kernel<BLOCK_SIZE>
      <<<max_blocks, BLOCK_SIZE>>>(input, &stats[0], output, &stats[1], N);
  normalize_kernel<BLOCK_SIZE>
      <<<max_blocks, BLOCK_SIZE>>>(output, &stats[1], N);
}
