#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>

template <typename T> struct Add {
  __host__ __device__ static constexpr T identity() { return T(0); }
  __device__ static T apply(T a, T b) { return a + b; }
  __device__ static void atomic_apply(T *address, T value) {
    atomicAdd(address, value);
  }
};
template <typename T> struct Max {
  __host__ __device__ static constexpr T identity() {
    return std::numeric_limits<T>::has_infinity
               ? -std::numeric_limits<T>::infinity()
               : std::numeric_limits<T>::lowest();
  }
  __device__ static T apply(T a, T b) { return max(a, b); }
  __device__ static void atomic_apply(T *address, T value) {
    atomicMax(address, value);
  }
};
template <typename T> struct Min {
  __host__ __device__ static constexpr T identity() {
    return std::numeric_limits<T>::has_infinity
               ? std::numeric_limits<T>::infinity()
               : std::numeric_limits<T>::max();
  }
  __device__ static T apply(T a, T b) { return min(a, b); }
  __device__ static void atomic_apply(T *address, T value) {
    atomicMin(address, value);
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
  const size_t lane = threadIdx.x % 32;
  const size_t warp_idx = threadIdx.x / 32;
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

template <int BLOCK_SIZE, template <typename> class Op = Max, typename T>
__global__ void sum_kernel(const T *__restrict__ input, T *__restrict__ output,
                           const size_t N) {
  using T4 = typename Vector4<T>::type;

  const size_t tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const size_t stride = gridDim.x * BLOCK_SIZE;

  T sum = Op<T>::identity();

  const size_t N4 = N / 4;
  const auto *input4 = reinterpret_cast<const T4 *>(input);
  for (size_t i = tid; i < N4; i += stride) {
    T4 v = input4[i];
    sum = Op<T>::apply(sum, v.x);
    sum = Op<T>::apply(sum, v.y);
    sum = Op<T>::apply(sum, v.z);
    sum = Op<T>::apply(sum, v.w);
  }
  for (size_t i = N4 * 4 + tid; i < N; i += stride) {
    sum = Op<T>::apply(sum, input[i]);
  }

  sum = block_reduce<BLOCK_SIZE, Op>(sum);
  if (threadIdx.x == 0) {
    Op<T>::atomic_apply(output, sum);
  }
}

extern "C" void solve(const float *input, float *output, int N) {
  cudaMemsetAsync(output, 0, sizeof(float));
  if (N <= 0)
    return;
  constexpr int BLOCK_SIZE = 256;
  // Few hundred blocks: enough to fill A800's 108 SMs, few enough to keep
  // atomicAdd contention on the single output negligible.
  constexpr int GRID_SIZE = 432;
  sum_kernel<BLOCK_SIZE, Add><<<GRID_SIZE, BLOCK_SIZE>>>(input, output, N);
}
