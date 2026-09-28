#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

template <typename T> struct Vector2;
template <> struct Vector2<float> {
  using type = float2;
};
template <> struct Vector2<int> {
  using type = int2;
};
template <> struct Vector2<double> {
  using type = double2;
};
template <> struct Vector2<half> {
  using type = half2;
};

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

template <int BLOCK_SIZE, typename T, typename Tout>
__global__ void dot(const T *__restrict__ A, const T *__restrict__ B,
                    Tout *__restrict__ output, int N) {
  using T2 = typename Vector2<T>::type;
  const size_t tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const size_t stride = gridDim.x * BLOCK_SIZE;

  Tout sum = 0;

  size_t N2 = N / 2;
  const auto *A2 = reinterpret_cast<const T2 *>(A);
  const auto *B2 = reinterpret_cast<const T2 *>(B);
  for (int i = tid; i < N2; i += stride) {
    T2 v = A2[i];
    T2 w = B2[i];
    sum += Tout(v.x) * Tout(w.x) + Tout(v.y) * Tout(w.y);
  }
  int tail_base = N2 * 2;
  if (tail_base + tid < N) {
    sum += Tout(A[tail_base + tid]) * Tout(B[tail_base + tid]);
  }

  sum = block_sum<BLOCK_SIZE>(sum);
  if (threadIdx.x == 0) {
    atomicAdd(output, sum);
  }
}

template <typename T1, typename T2>
__global__ void memcpy_kernel(T1 *dst, const T2 *src, const size_t N) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
    return;
  dst[tid] = src[tid];
}

// A, B, result are device pointers
extern "C" void solve(const half *A, const half *B, half *result, int N) {
  float *buffer;
  cudaMalloc(&buffer, sizeof(float));
  cudaMemset(buffer, 0, sizeof(float));

  constexpr size_t BLOCK_SIZE = 256;
  dot<BLOCK_SIZE>
      <<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(A, B, buffer, N);

  memcpy_kernel<<<1, 1>>>(result, buffer, 1);

  cudaFree(buffer);
}
