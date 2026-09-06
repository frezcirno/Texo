#include <cmath>
#include <cuda_runtime.h>

__device__ inline float warp_max(float val) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    val = max(val, __shfl_down_sync(0xffffffff, val, off));
  }
  return val;
}

__device__ inline float warp_sum(float value) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, off);
  }
  return value;
}

template <int BLOCK_SIZE> __device__ inline float block_max(float val) {
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ float warp_maxes[NUM_WARPS];

  int lane = threadIdx.x % 32;
  int warp_idx = threadIdx.x / 32;

  val = warp_max(val);

  if (lane == 0)
    warp_maxes[warp_idx] = val;

  __syncthreads();

  if (warp_idx == 0) {
    val = (lane < NUM_WARPS) ? warp_maxes[lane] : -INFINITY;
    val = warp_max(val);
  }

  return val;
}

template <int BLOCK_SIZE> __device__ inline float block_sum(float value) {
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ float warp_sums[NUM_WARPS];

  int lane = threadIdx.x % 32;
  int warp = threadIdx.x / 32;

  value = warp_sum(value);

  if (lane == 0)
    warp_sums[warp] = value;

  __syncthreads();

  if (warp == 0) {
    value = lane < NUM_WARPS ? warp_sums[lane] : 0.0f;
    value = warp_sum(value);
  }

  return value;
}

template <int BLOCK_SIZE>
__global__ void qkt_kernel(const float *__restrict__ Q, // (M, d)
                           const float *__restrict__ K, // (N, d)
                           float *__restrict__ qkt,     // (M, N)
                           int M, int d, int N) {
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidy >= M || tidx >= N)
    return;

  float result = 0.0f;
  for (int i = 0; i < d; i++) {
    result += Q[tidy * d + i] * K[tidx * d + i];
  }
  qkt[tidy * N + tidx] = result;
}

template <int BLOCK_SIZE>
__global__ void max_kernel(const float *__restrict__ input, // (M, N)
                           float *__restrict__ maximum,     // (M,)
                           int M, int N) {
  int row = blockIdx.x;
  if (row >= M)
    return;

  float local_max = -INFINITY;

  for (int i = threadIdx.x; i < N; i += BLOCK_SIZE)
    local_max = max(local_max, input[row * N + i]);

  local_max = block_max<BLOCK_SIZE>(local_max);

  if (threadIdx.x == 0)
    maximum[row] = local_max;
}

template <int BLOCK_SIZE>
__global__ void exp_sum_kernel(float *__restrict__ io,            // (M, N)
                               const float *__restrict__ maximum, // (M,)
                               float *__restrict__ total,         // (M,)
                               int d, int N) {
  __shared__ float inverse_sqrtd;
  if (threadIdx.x == 0) {
    inverse_sqrtd = rsqrtf(static_cast<float>(d));
  }
  __syncthreads();

  int row = blockIdx.x;
  float local_sum = 0.0f;

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float y = __expf((io[row * N + i] - maximum[row]) * inverse_sqrtd);
    io[row * N + i] = y;
    local_sum += y;
  }

  local_sum = block_sum<BLOCK_SIZE>(local_sum);

  if (threadIdx.x == 0)
    total[row] = local_sum;
}

template <int BLOCK_SIZE>
__global__ void normalize_kernel(float *__restrict__ output,      // (M, N)
                                 const float *__restrict__ total, // (M,)
                                 int N) {
  int row = blockIdx.x;

  __shared__ float inverse_total;
  if (threadIdx.x == 0)
    inverse_total = 1.0f / total[row];
  __syncthreads();

  int tid = threadIdx.x;
  int stride = blockDim.x;

  for (int i = tid; i < N; i += stride)
    output[row * N + i] *= inverse_total;
}

template <int BLOCK_SIZE>
__global__ void v_kernel(const float *__restrict__ qkt, // (M, N)
                         const float *__restrict__ V,   // (N, d)
                         float *__restrict__ output,    // (M, d)
                         int M, int d, int N) {
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidy >= M || tidx >= d)
    return;

  float result = 0.0f;
  for (int i = 0; i < N; i++) {
    result += qkt[tidy * N + i] * V[i * d + tidx];
  }
  output[tidy * d + tidx] = result;
}

// Q, K, V, output are device pointers
extern "C" void solve(const float *__restrict__ Q, const float *__restrict__ K,
                      const float *__restrict__ V, float *__restrict__ output,
                      int M, int N, int d) {
  constexpr int BLOCK_SIZE = 256;

  float *qkt; // (M, N)
  cudaMalloc(&qkt, M * N * sizeof(float));

  float *maximum;
  cudaMalloc(&maximum, 2 * M * sizeof(float));
  float *total = &maximum[M];

  qkt_kernel<BLOCK_SIZE><<<dim3((N + 15) / 16, (M + 15) / 16), dim3(16, 16)>>>(
      Q, K, qkt, M, d, N);

  max_kernel<BLOCK_SIZE><<<M, BLOCK_SIZE>>>(qkt, maximum, M, N);
  exp_sum_kernel<BLOCK_SIZE><<<M, BLOCK_SIZE>>>(qkt, maximum, total, d, N);
  normalize_kernel<BLOCK_SIZE><<<M, BLOCK_SIZE>>>(qkt, total, N);

  v_kernel<BLOCK_SIZE><<<dim3((d + 15) / 16, (M + 15) / 16), dim3(16, 16)>>>(
      qkt, V, output, M, d, N);

  cudaFree((void *)maximum);
  cudaFree((void *)qkt);
  cudaDeviceSynchronize();
}
