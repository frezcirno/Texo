#include <cuda_runtime.h>

template <typename T> __device__ T pow2(T x) { return x * x; }

template <typename T> __device__ inline T warp_sum(T val) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, off);
  }
  return val;
}

template <int BLOCK_SIZE, typename T> __device__ inline T block_sum(T val) {
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ T warp_sums[NUM_WARPS];
  size_t lane = threadIdx.x % 32;
  size_t warp_idx = threadIdx.x / 32;
  val = warp_sum(val);
  if (lane == 0) {
    warp_sums[warp_idx] = val;
  }
  __syncthreads();
  if (warp_idx == 0) {
    val = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
    val = warp_sum(val);
  }
  return val;
}

template <typename T> struct Vector4;
template <> struct Vector4<float> {
  using type = float4;
};
template <> struct Vector4<int> {
  using type = int4;
};
template <> struct Vector4<double> {
  using type = double4;
};

__global__ void fused_kernel(const float *x, const float *residual,
                             const float *weight, float *out, int N, int C,
                             float eps) {
  int n = blockIdx.x;
  int t = threadIdx.x;

  float sum = 0.f;
  for (int j = t; j < C; j += blockDim.x) {
    float z = x[n * C + j] + residual[n * C + j];
    sum += z * z;
  }

  sum = block_sum<256>(sum);

  // block_sum 的总和只在第 0 个线程可靠；放到共享内存，
  // 让整个 CTA 后续都能使用。
  __shared__ float inv_rms;
  if (t == 0)
    inv_rms = rsqrtf(sum / C + eps);
  __syncthreads();

  for (int j = t; j < C; j += blockDim.x) {
    float z = x[n * C + j] + residual[n * C + j];
    out[n * C + j] = z * inv_rms * weight[j];
  }
}

// x, residual, weight, out are device pointers
extern "C" void solve(const float *x,        // (N, C)
                      const float *residual, // (N, C)
                      const float *weight,   // (C,)
                      float *out,            // (N, C)
                      int N, int C, float eps) {
  fused_kernel<<<N, 256>>>(x, residual, weight, out, N, C, eps);
}
