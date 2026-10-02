#include <cuda_runtime.h>

#include <math.h>

template <int BLOCK_SIZE>
__device__ inline float block_max(float value, float *warp_values) {
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = max(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  if (lane == 0)
    warp_values[warp] = value;
  __syncthreads();

  if (warp == 0) {
    value = lane < NUM_WARPS ? warp_values[lane] : -INFINITY;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value = max(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    if (lane == 0)
      warp_values[0] = value;
  }
  __syncthreads();
  value = warp_values[0];
  __syncthreads();
  return value;
}

template <int BLOCK_SIZE>
__device__ inline float block_sum(float value, float *warp_values) {
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  if (lane == 0)
    warp_values[warp] = value;
  __syncthreads();

  if (warp == 0) {
    value = lane < NUM_WARPS ? warp_values[lane] : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset);
    }
    if (lane == 0)
      warp_values[0] = value;
  }
  __syncthreads();
  value = warp_values[0];
  __syncthreads();
  return value;
}

// One CTA scans the vector in BLOCK_SIZE-element tiles. For each tile B, keep
// the running pair (m, l), where
//   m = max seen x_i,       l = sum seen exp(x_i - m).
// If the next tile has (m_B, l_B), the stable online-softmax recurrence is
//   m' = max(m, m_B)
//   l' = l * exp(m - m') + l_B * exp(m_B - m').
// At the end, softmax_i = exp(x_i - m) / l.
template <size_t BLOCK_SIZE>
__global__ void online_softmax_kernel(const float *__restrict__ input,
                                      float *__restrict__ output, const size_t N) {
  const size_t tid = threadIdx.x;
  __shared__ float warp_values[BLOCK_SIZE / 32];
  float running_max = -INFINITY;
  float running_sum = 0.0f;

  for (int base = 0; base < N; base += BLOCK_SIZE) {
    const int index = base + tid;
    const float x = index < N ? input[index] : -INFINITY;

    const float tile_max = block_max<BLOCK_SIZE>(x, warp_values);
    const float tile_exp = index < N ? exp(x - tile_max) : 0.0f;
    const float tile_sum = block_sum<BLOCK_SIZE>(tile_exp, warp_values);

    const float new_max = max(running_max, tile_max);
    running_sum = running_sum * exp(running_max - new_max) +
                  tile_sum * exp(tile_max - new_max);
    running_max = new_max;
  }

  // Every lane has the same final (running_max, running_sum), so the output
  // pass needs no additional synchronization or global reduction.
  for (int base = 0; base < N; base += BLOCK_SIZE) {
    const int index = base + tid;
    if (index < N) {
      output[index] = exp(input[index] - running_max) / running_sum;
    }
  }
}

// input and output are device pointers (i.e. pointers to memory on the GPU).
extern "C" void solve(const float *input, float *output, int N) {
  if (N <= 0)
    return;

  constexpr int BLOCK_SIZE = 256;
  online_softmax_kernel<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(input, output, N);
}
