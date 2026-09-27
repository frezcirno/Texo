#include <cuda_runtime.h>
#include <math.h>

template <typename T> __device__ inline T warp_max(T val) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    val = max(val, __shfl_down_sync(0xffffffff, val, off));
  }
  return val;
}

template <int BLOCK_SIZE, typename T> __device__ inline T block_max(T val) {
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ T warp_maxes[NUM_WARPS];
  size_t lane = threadIdx.x % 32;
  size_t warp_idx = threadIdx.x / 32;
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

template <typename T> __device__ T atomic_max(T *addr, T val);

template <> __device__ float atomic_max<float>(float *addr, float val) {
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

template <> __device__ double atomic_max<double>(double *addr, double val) {
  unsigned long long *addr_as_int =
      reinterpret_cast<unsigned long long *>(addr);
  unsigned long long old = *addr_as_int;
  unsigned long long assumed;
  do {
    assumed = old;
    if (val <= __longlong_as_double(assumed))
      break;
    old = atomicCAS(addr_as_int, assumed, __double_as_longlong(val));
  } while (assumed != old);
  return __longlong_as_double(old);
}

template <int BLOCK_SIZE, typename T>
__global__ void max_2d_kernel(const T *__restrict__ input, // (M, N)
                              T *__restrict__ output,      // (N,)
                              int M, int N) {
  const size_t col = blockIdx.x;
  T max_res = -INFINITY;

  for (int i = threadIdx.x; i < M; i += BLOCK_SIZE) {
    max_res = max(max_res, input[i * N + col]);
  }

  max_res = block_max<BLOCK_SIZE>(max_res);
  if (threadIdx.x == 0) {
    output[col] = max_res;
  }
}

extern "C" void solve(const float *input, // (M, N)
                      float *output,      // (N,)
                      int M, int N) {
  max_2d_kernel<256><<<N, 256>>>(input, output, M, N);
}
