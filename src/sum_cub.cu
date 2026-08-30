#include <cub/cub.cuh>
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void reduce_cub(const float *input, float *output, int N) {
  // Cache temp workspace across calls so the benchmark measures the kernel,
  // not cudaMalloc. Resized only when N grows.
  static void *d_temp = nullptr;
  static size_t temp_bytes = 0;
  static int last_N = 0;

  if (N != last_N) {
    size_t needed = 0;
    cub::DeviceReduce::Sum(nullptr, needed, input, output, N);
    if (needed > temp_bytes) {
      if (d_temp)
        cudaFree(d_temp);
      cudaMalloc(&d_temp, needed);
      temp_bytes = needed;
    }
    last_N = N;
  }

  cub::DeviceReduce::Sum(d_temp, temp_bytes, input, output, N);
}
