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

  const size_t lane = threadIdx.x % 32;
  const size_t warp_idx = threadIdx.x / 32;

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

  const size_t lane = threadIdx.x % 32;
  const size_t warp = threadIdx.x / 32;

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
__global__ void qkt_kernel(const float *__restrict__ Q, // (M, H, D)
                           const float *__restrict__ K, // (N, H, D)
                           float *__restrict__ qkt,     // (H, M, N)
                           const size_t M, const size_t N, const size_t D,
                           const size_t H) {
  const size_t kn = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t qm = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t h = blockIdx.z * blockDim.z + threadIdx.z;

  if (qm >= M || kn >= N || h >= H)
    return;

  float result = 0.0f;
  for (int i = 0; i < D; i++) {
    result += Q[qm * H * D + h * D + i] * K[kn * H * D + h * D + i];
  }
  qkt[(h * M + qm) * N + kn] = result;
}

template <int BLOCK_SIZE>
__global__ void max_kernel(const float *__restrict__ input, // (H, M, N)
                           float *__restrict__ maximum,     // (H, M)
                           const size_t H, const size_t M, const size_t N) {
  const size_t hm = blockIdx.x;
  const size_t h = hm / M;
  const size_t m = hm % M;

  float local_max = -INFINITY;

  for (int n = threadIdx.x; n < N; n += BLOCK_SIZE) {
    local_max = max(local_max, input[(h * M + m) * N + n]);
  }

  local_max = block_max<BLOCK_SIZE>(local_max);

  if (threadIdx.x == 0) {
    maximum[hm] = local_max;
  }
}

template <int BLOCK_SIZE>
__global__ void exp_sum_kernel(float *__restrict__ qkt,           // (H, M, N)
                               const float *__restrict__ maximum, // (H, M)
                               float *__restrict__ total,         // (H, M)
                               const size_t M, const size_t N, const size_t H,
                               const size_t D) {
  const float inverse_sqrtd = rsqrtf(static_cast<float>(D));

  const size_t hm = blockIdx.x;
  const size_t h = hm / M;
  const size_t m = hm % M;
  float local_sum = 0.0f;

  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    float y = __expf((qkt[(h * M + m) * N + n] - maximum[hm]) * inverse_sqrtd);
    qkt[(h * M + m) * N + n] = y;
    local_sum += y;
  }

  local_sum = block_sum<BLOCK_SIZE>(local_sum);

  if (threadIdx.x == 0) {
    total[hm] = local_sum;
  }
}

template <int BLOCK_SIZE>
__global__ void normalize_kernel(float *__restrict__ output,      // (H, M, N)
                                 const float *__restrict__ total, // (H, M)
                                 const size_t H, const size_t M,
                                 const size_t N) {
  const size_t hm = blockIdx.x;
  const size_t h = hm / M;
  const size_t m = hm % M;

  const float inverse_total = 1.0f / total[hm];

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  for (int n = tid; n < N; n += stride) {
    output[(h * M + m) * N + n] *= inverse_total;
  }
}

template <int BLOCK_SIZE>
__global__ void v_kernel(const float *__restrict__ qkt, // (H, M, N)
                         const float *__restrict__ V,   // (N, H, D)
                         float *__restrict__ output,    // (M, H, D)
                         const size_t M, const size_t N, const size_t H,
                         const size_t D) {
  const size_t vd = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t qm = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t h = blockIdx.z * blockDim.z + threadIdx.z;

  if (qm >= M || vd >= D || h >= H)
    return;

  float result = 0.0f;
  for (int vn = 0; vn < N; vn++) {
    result += qkt[(h * M + qm) * N + vn] * V[(vn * H + h) * D + vd];
  }
  output[qm * H * D + h * D + vd] = result;
}

// Q, K, V, output are device pointers
extern "C" void solve(const float *Q, // (M, H, D)
                      const float *K, // (N, H, D)
                      const float *V, // (N, H, D)
                      float *output,  // (M, H, D)
                      int M, int N, int H, int D) {
  constexpr int BLOCK_SIZE = 256;

  float *qkt; // (H, M, N)
  cudaMalloc(&qkt, H * M * N * sizeof(float));

  qkt_kernel<BLOCK_SIZE>
      <<<dim3((N + 7) / 8, (M + 7) / 8, (H + 3) / 4), dim3(8, 8, 4)>>>(
          Q, K, qkt, M, N, D, H);

  float *maximum; // (H, M)
  float *total;   // (H, M)
  cudaMalloc(&maximum, H * M * sizeof(float));
  cudaMalloc(&total, H * M * sizeof(float));

  max_kernel<BLOCK_SIZE><<<H * M, BLOCK_SIZE>>>(qkt, maximum, H, M, N);
  exp_sum_kernel<BLOCK_SIZE>
      <<<H * M, BLOCK_SIZE>>>(qkt, maximum, total, M, N, H, D);
  normalize_kernel<BLOCK_SIZE><<<H * M, BLOCK_SIZE>>>(qkt, total, H, M, N);

  v_kernel<BLOCK_SIZE>
      <<<dim3((D + 7) / 8, (M + 7) / 8, (H + 3) / 4), dim3(8, 8, 4)>>>(
          qkt, V, output, M, N, H, D);

  cudaFree((void *)maximum);
  cudaFree((void *)total);
  cudaFree((void *)qkt);
  cudaDeviceSynchronize();
}
