#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

template <int BLOCK_SIZE>
__global__ void qkt_kernel(const float *__restrict__ Q, // (M, d)
                           const float *__restrict__ K, // (N, d)
                           float *__restrict__ qkt,     // (M, N)
                           int M, int d, int N) {
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidy >= M || tidx >= N)
    return;

  for (int i = 0; i < d; i++) {
    qkt[tidy * N + tidx] += Q[tidy * d + i] * K[tidx * d + i];
  }
}

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
__global__ void max_kernel(const float *__restrict__ input, // (N,)
                           float *__restrict__ output,      // (1,)
                           int N) {
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
__global__ void exp_sum_kernel(float *__restrict__ io,            // (N,)
                               const float *__restrict__ maximum, // (1,)
                               float *__restrict__ total,         // (1,)
                               int d, int N) {
  __shared__ float sqrtd;
  if (threadIdx.x == 0) {
    sqrtd = sqrtf(static_cast<float>(d));
  }
  __syncthreads();

  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<BLOCK_SIZE>(block);

  int N4 = N / 4;
  auto *io4 = reinterpret_cast<float4 *>(io);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  float local_sum = 0.0f;

  for (int i = tid; i < N4; i += stride) {
    float4 x = io4[i];

    float4 y;
    y.x = __expf((x.x - *maximum) * sqrtd);
    y.y = __expf((x.y - *maximum) * sqrtd);
    y.z = __expf((x.z - *maximum) * sqrtd);
    y.w = __expf((x.w - *maximum) * sqrtd);

    io4[i] = y;
    local_sum += y.x + y.y + y.z + y.w;
  }

  int tail_base = N4 * 4;
  int tail_index = tail_base + tid;
  if (tail_index < N) {
    float value = __expf((io[tail_index] - *maximum) * sqrtd);
    io[tail_index] = value;
    local_sum += value;
  }

  local_sum = cg::reduce(tile, local_sum, cg::plus<float>());

  if (tile.thread_rank() == 0) {
    atomicAdd(total, local_sum);
  }
}

template <int BLOCK_SIZE>
__global__ void normalize_kernel(float *__restrict__ output,      // (N,)
                                 const float *__restrict__ total, // (1,)
                                 int N) {
  __shared__ float inverse_total;
  if (threadIdx.x == 0) {
    inverse_total = 1.0f / total[0];
  }
  __syncthreads();

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

template <int BLOCK_SIZE>
__global__ void softmax_sqrtd_kernel(float *__restrict__ input, // qkt (M, N)
                                     float *maximum_and_total,  // (2,)
                                     int M, int d, int N) {}

template <int BLOCK_SIZE>
__global__ void v_kernel(const float *__restrict__ qkt, // (M, N)
                         const float *__restrict__ V,   // (N, d)
                         float *__restrict__ output,    // (M, d)
                         int M, int d, int N) {
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidy >= M || tidx >= d)
    return;

  for (int i = 0; i < N; i++) {
    output[tidy * d + tidx] += qkt[tidy * N + i] * V[i * d + tidx];
  }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float *__restrict__ Q, const float *__restrict__ K,
                      const float *__restrict__ V, float *__restrict__ output,
                      int M, int N, int d) {
  constexpr int BLOCK_SIZE = 256;
  constexpr int GRID_SIZE = 128;

  float *qkt;
  cudaMalloc(&qkt, M * N * sizeof(float));
  cudaMemset(qkt, 0, M * N * sizeof(float));

  float *maximum_and_total;
  cudaMalloc(&maximum_and_total, 2 * sizeof(float));
  cudaMemcpyAsync(&maximum_and_total[0], qkt, sizeof(float),
                  cudaMemcpyDeviceToDevice);
  cudaMemsetAsync(&maximum_and_total[1], 0, sizeof(float));

  qkt_kernel<BLOCK_SIZE><<<dim3((M + 15) / 16, (N + 15) / 16), dim3(16, 16)>>>(
      Q, K, qkt, M, d, N);

  //   softmax_sqrtd_kernel<BLOCK_SIZE>
  //       <<<128, BLOCK_SIZE>>>(qkt, maximum_and_total, M, d, N);
  {
    constexpr int MAX_BLOCKS = 432;
    const int MN = M * N;
    const int N = MN;

    int max_blocks = (N / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    max_blocks = max_blocks < 1 ? 1 : max_blocks;
    max_blocks = max_blocks > MAX_BLOCKS ? MAX_BLOCKS : max_blocks;

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks = blocks > MAX_BLOCKS ? MAX_BLOCKS : blocks;

    max_kernel<BLOCK_SIZE>
        <<<max_blocks, BLOCK_SIZE>>>(qkt, &maximum_and_total[0], N);
    exp_sum_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
        qkt, &maximum_and_total[0], &maximum_and_total[1], d, N);
    normalize_kernel<BLOCK_SIZE>
        <<<blocks, BLOCK_SIZE>>>(qkt, &maximum_and_total[1], N);
  }

  v_kernel<BLOCK_SIZE><<<dim3((M + 15) / 16, (N + 15) / 16), dim3(16, 16)>>>(
      qkt, V, output, M, d, N);

  cudaFree((void *)maximum_and_total);
  cudaFree((void *)qkt);
  cudaDeviceSynchronize();
}
