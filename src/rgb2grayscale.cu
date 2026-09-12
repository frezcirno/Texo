#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(const float *input, float *output,
                                        int width, int height) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int N = width * height;
  if (tid >= N)
    return;
  output[tid] = 0.299 * input[3 * tid] + 0.587 * input[3 * tid + 1] +
                0.114 * input[3 * tid + 2];
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int width,
                      int height) {
  int total_pixels = width * height;
  int threadsPerBlock = 256;
  int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

  rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output,
                                                              width, height);
  cudaDeviceSynchronize();
}
