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
__global__ void qkt_kernel(const float *__restrict__ Q, // (N, d*h)
                           const float *__restrict__ K, // (N, d*h)
                           float *__restrict__ qkt,     // (h, N, N)
                           const size_t N, const size_t D, const size_t H) {
  const size_t kn = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t qn = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t h = blockIdx.z * blockDim.z + threadIdx.z;

  if (qn >= N || kn >= N || h >= H)
    return;

  float result = 0.0f;
  for (int i = 0; i < D; i++) {
    result += Q[qn * D * H + h * D + i] * K[kn * D * H + h * D + i];
  }
  qkt[(h * N + qn) * N + kn] = result;
}

template <int BLOCK_SIZE>
__global__ void max_kernel(const float *__restrict__ input, // (h, N, N)
                           float *__restrict__ maximum,     // (h, N)
                           const size_t H, const size_t N) {
  const size_t hn = blockIdx.x;

  float local_max = -INFINITY;

  for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
    local_max = max(local_max, input[hn * N + i]);
  }

  local_max = block_max<BLOCK_SIZE>(local_max);

  if (threadIdx.x == 0) {
    maximum[hn] = local_max;
  }
}

template <int BLOCK_SIZE>
__global__ void exp_sum_kernel(float *__restrict__ io,            // (h, N, N)
                               const float *__restrict__ maximum, // (h, N)
                               float *__restrict__ total,         // (h, N)
                               const size_t d, const size_t H, const size_t N) {
  const float inverse_sqrtd = rsqrtf(static_cast<float>(d));

  const size_t hn = blockIdx.x;
  float local_sum = 0.0f;

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float y = __expf((io[hn * N + i] - maximum[hn]) * inverse_sqrtd);
    io[hn * N + i] = y;
    local_sum += y;
  }

  local_sum = block_sum<BLOCK_SIZE>(local_sum);

  if (threadIdx.x == 0) {
    total[hn] = local_sum;
  }
}

template <int BLOCK_SIZE>
__global__ void normalize_kernel(float *__restrict__ output,      // (h, N, N)
                                 const float *__restrict__ total, // (h, N)
                                 const size_t H, const size_t N) {
  const size_t hn = blockIdx.x;

  const float inverse_total = 1.0f / total[hn];

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  for (int i = tid; i < N; i += stride) {
    output[hn * N + i] *= inverse_total;
  }
}

template <int BLOCK_SIZE>
__global__ void v_kernel(const float *__restrict__ qkt, // (h, N, N)
                         const float *__restrict__ V,   // (N, d*h)
                         float *__restrict__ output,    // (N, d*h)
                         const size_t N, const size_t D, const size_t H) {
  const size_t vd = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t qn = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t h = blockIdx.z * blockDim.z + threadIdx.z;

  if (qn >= N || vd >= D || h >= H)
    return;

  float result = 0.0f;
  for (int vn = 0; vn < N; vn++) {
    result += qkt[(h * N + qn) * N + vn] * V[vn * D * H + h * D + vd];
  }
  output[qn * D * H + h * D + vd] = result;
}

// Q, K, V, output are device pointers
extern "C" void solve(const float *Q, // (N, d*h)
                      const float *K, // (N, d*h)
                      const float *V, // (N, d*h)
                      float *output,  // (N, d*h)
                      int N, int d_model, int h) {
  constexpr int BLOCK_SIZE = 256;
  const size_t D = d_model / h;
  const size_t H = h;

  float *qkt; // (h, N, N)
  cudaMalloc(&qkt, H * N * N * sizeof(float));

  qkt_kernel<BLOCK_SIZE>
      <<<dim3((N + 7) / 8, (N + 7) / 8, (H + 3) / 4), dim3(8, 8, 4)>>>(
          Q, K, qkt, N, D, H);

  float *maximum; // (h, N)
  float *total;   // (h, N)
  cudaMalloc(&maximum, H * N * sizeof(float));
  cudaMalloc(&total, H * N * sizeof(float));

  max_kernel<BLOCK_SIZE><<<H * N, BLOCK_SIZE>>>(qkt, maximum, H, N);
  exp_sum_kernel<BLOCK_SIZE>
      <<<H * N, BLOCK_SIZE>>>(qkt, maximum, total, D, H, N);
  normalize_kernel<BLOCK_SIZE><<<H * N, BLOCK_SIZE>>>(qkt, total, H, N);

  v_kernel<BLOCK_SIZE>
      <<<dim3((D + 7) / 8, (N + 7) / 8, (H + 3) / 4), dim3(8, 8, 4)>>>(
          qkt, V, output, N, D, H);

  cudaFree((void *)maximum);
  cudaFree((void *)total);
  cudaFree((void *)qkt);
  cudaDeviceSynchronize();
}
