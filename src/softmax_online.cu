#include <cuda_runtime.h>

template <typename T> struct Add {
  __host__ __device__ static constexpr T identity() { return T(0); }
  __device__ static T apply(T a, T b) { return a + b; }
};

template <typename T> struct Max {
  __host__ __device__ static constexpr T identity() { return -INFINITY; }
  __device__ static T apply(T a, T b) { return max(a, b); }
};

template <template <typename> class Op, typename T>
__device__ inline T warp_reduce(T value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = Op<T>::apply(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

template <int BLOCK_SIZE, template <typename> class Op, typename T>
__device__ inline T block_reduce(T value) {
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ T warp_values[NUM_WARPS];
  __shared__ T block_value;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_reduce<Op, T>(value);
  if (lane == 0) {
    warp_values[warp] = value;
  }
  __syncthreads();

  if (warp == 0) {
    value = lane < NUM_WARPS ? warp_values[lane] : Op<T>::identity();
    value = warp_reduce<Op, T>(value);
    if (lane == 0) {
      block_value = value;
    }
  }
  __syncthreads();
  return block_value;
}

struct SoftmaxPartial {
  float max_value;
  float exp_sum;
};

// Small vectors avoid scratch allocation and extra kernel launches.
template <int BLOCK_SIZE>
__global__ void online_softmax_single_kernel(const float *__restrict__ input,
                                             float *__restrict__ output,
                                             int N) {
  float running_max = -INFINITY;
  float running_sum = 0.0f;

  for (int base = 0; base < N; base += BLOCK_SIZE) {
    const int index = base + threadIdx.x;
    const float x = index < N ? input[index] : -INFINITY;
    const float tile_max = block_reduce<BLOCK_SIZE, Max>(x);
    const float tile_exp = index < N ? __expf(x - tile_max) : 0.0f;
    const float tile_sum = block_reduce<BLOCK_SIZE, Add>(tile_exp);

    const float next_max = max(running_max, tile_max);
    running_sum = running_sum * __expf(running_max - next_max) +
                  tile_sum * __expf(tile_max - next_max);
    running_max = next_max;
  }

  for (int base = 0; base < N; base += BLOCK_SIZE) {
    const int index = base + threadIdx.x;
    if (index < N) {
      output[index] = __expf(input[index] - running_max) / running_sum;
    }
  }
}

// Each CTA computes the stable (max, sum(exp(x - max))) pair for one tile.
template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void softmax_partial_kernel(const float *__restrict__ input,
                                       SoftmaxPartial *__restrict__ partials,
                                       int N) {
  const int base = blockIdx.x * TILE_SIZE;
  float local_max = -INFINITY;
  for (int offset = threadIdx.x; offset < TILE_SIZE; offset += BLOCK_SIZE) {
    const int index = base + offset;
    if (index < N) {
      local_max = max(local_max, input[index]);
    }
  }
  const float tile_max = block_reduce<BLOCK_SIZE, Max>(local_max);

  float local_sum = 0.0f;
  for (int offset = threadIdx.x; offset < TILE_SIZE; offset += BLOCK_SIZE) {
    const int index = base + offset;
    if (index < N) {
      local_sum += __expf(input[index] - tile_max);
    }
  }
  const float tile_sum = block_reduce<BLOCK_SIZE, Add>(local_sum);

  if (threadIdx.x == 0) {
    partials[blockIdx.x] = {tile_max, tile_sum};
  }
}

// Merge tile pairs in a numerically stable way. Each lane handles a strided
// subset of tiles before the block combines the resulting maximum and sum.
template <int BLOCK_SIZE>
__global__ void
softmax_finalize_kernel(const SoftmaxPartial *__restrict__ partials,
                        int num_tiles, float *__restrict__ stats) {
  float local_max = -INFINITY;
  for (int tile = threadIdx.x; tile < num_tiles; tile += BLOCK_SIZE) {
    local_max = max(local_max, partials[tile].max_value);
  }
  const float global_max = block_reduce<BLOCK_SIZE, Max>(local_max);

  float local_sum = 0.0f;
  for (int tile = threadIdx.x; tile < num_tiles; tile += BLOCK_SIZE) {
    const SoftmaxPartial partial = partials[tile];
    local_sum += partial.exp_sum * __expf(partial.max_value - global_max);
  }
  const float global_sum = block_reduce<BLOCK_SIZE, Add>(local_sum);

  if (threadIdx.x == 0) {
    stats[0] = global_max;
    stats[1] = global_sum;
  }
}

template <int BLOCK_SIZE>
__global__ void softmax_output_kernel(const float *__restrict__ input,
                                      float *__restrict__ output, int N,
                                      const float *__restrict__ stats) {
  const float max_value = stats[0];
  const float inverse_sum = 1.0f / stats[1];
  const size_t stride = static_cast<size_t>(gridDim.x) * BLOCK_SIZE;
  for (size_t index =
           static_cast<size_t>(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
       index < static_cast<size_t>(N); index += stride) {
    output[index] = __expf(input[index] - max_value) * inverse_sum;
  }
}

// input and output are device pointers (i.e. pointers to memory on the GPU).
extern "C" void solve(const float *input, float *output, int N) {
  if (N <= 0) {
    return;
  }

  constexpr int BLOCK_SIZE = 256;
  constexpr int SMALL_INPUT_LIMIT = 4096;
  if (N <= SMALL_INPUT_LIMIT) {
    online_softmax_single_kernel<BLOCK_SIZE>
        <<<1, BLOCK_SIZE>>>(input, output, N);
    return;
  }

  constexpr int TILE_SIZE = 4096;
  constexpr int MAX_OUTPUT_BLOCKS = 432;
  const int num_tiles = (N - 1) / TILE_SIZE + 1;
  const int needed_output_blocks = (N - 1) / BLOCK_SIZE + 1;
  const int output_blocks = needed_output_blocks < MAX_OUTPUT_BLOCKS
                                ? needed_output_blocks
                                : MAX_OUTPUT_BLOCKS;

  SoftmaxPartial *partials = nullptr;
  float *stats = nullptr;
  if (cudaMalloc(&partials, static_cast<size_t>(num_tiles) *
                                sizeof(SoftmaxPartial)) != cudaSuccess ||
      cudaMalloc(&stats, 2 * sizeof(float)) != cudaSuccess) {
    if (partials != nullptr) {
      cudaFree(partials);
    }
    return;
  }

  softmax_partial_kernel<BLOCK_SIZE, TILE_SIZE>
      <<<num_tiles, BLOCK_SIZE>>>(input, partials, N);
  softmax_finalize_kernel<BLOCK_SIZE>
      <<<1, BLOCK_SIZE>>>(partials, num_tiles, stats);
  softmax_output_kernel<BLOCK_SIZE>
      <<<output_blocks, BLOCK_SIZE>>>(input, output, N, stats);

  cudaFree(stats);
  cudaFree(partials);
}
