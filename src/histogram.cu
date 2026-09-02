#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>

template <int BLOCK_SIZE>
__global__ void histogram_kernel(const int *__restrict__ input, // (N,)
                                 int *__restrict__ histogram,   // (num_bins,)
                                 int N, int num_bins) {
  extern __shared__ int local_hist[];
  for (int bin = threadIdx.x; bin < num_bins; bin += BLOCK_SIZE)
    local_hist[bin] = 0;

  __syncthreads();

  int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int stride = gridDim.x * BLOCK_SIZE;

  for (int i = tid; i < N; i += stride) {
    int bin = input[i];
    if (bin >= 0 && bin < num_bins) {
      atomicAdd(&local_hist[bin], 1);
    }
  }

  __syncthreads();

  for (int bin = threadIdx.x; bin < num_bins; bin += BLOCK_SIZE) {
    int count = local_hist[bin];
    if (count != 0)
      atomicAdd(&histogram[bin], count);
  }
}

template <int BLOCK_SIZE>
__global__ void histogram_global_kernel(const int *__restrict__ input,
                                        int *__restrict__ histogram, int N,
                                        int num_bins) {
  int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int stride = gridDim.x * BLOCK_SIZE;

  for (int i = tid; i < N; i += stride) {
    int bin = input[i];
    if (bin >= 0 && bin < num_bins)
      atomicAdd(&histogram[bin], 1);
  }
}

// input, histogram are device pointers
extern "C" void solve(const int *input, // (N,)
                      int *histogram,   // (num_bins,)
                      int N, int num_bins) {
  if (histogram == nullptr || num_bins <= 0)
    return;

  const size_t histogram_bytes = static_cast<size_t>(num_bins) * sizeof(int);
  if (cudaMemset(histogram, 0, histogram_bytes) != cudaSuccess || N <= 0 ||
      input == nullptr)
    return;

  constexpr int BLOCK_SIZE = 256;
  int device = 0;
  int sm_count = 0;
  int max_shared_bytes = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                             device) != cudaSuccess ||
      cudaDeviceGetAttribute(&max_shared_bytes,
                             cudaDevAttrMaxSharedMemoryPerBlock,
                             device) != cudaSuccess)
    return;

  const int input_blocks = (N - 1) / BLOCK_SIZE + 1;
  const int blocks = std::min(input_blocks, 4 * sm_count);
  if (histogram_bytes <= static_cast<size_t>(max_shared_bytes)) {
    histogram_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE, histogram_bytes>>>(
        input, histogram, N, num_bins);
  } else {
    histogram_global_kernel<BLOCK_SIZE>
        <<<blocks, BLOCK_SIZE>>>(input, histogram, N, num_bins);
  }
}
