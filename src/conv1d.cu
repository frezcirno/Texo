#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float *input, const float *kernel,
                                      float *output, int input_size,
                                      int kernel_size) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= input_size - kernel_size + 1)
    return;
  float result = 0;
  for (int i = 0; i < kernel_size; i++) {
    result += input[tid + i] * kernel[i];
  }
  output[tid] = result;
}

// input, kernel, output are device pointers (i.e. pointers to memory on the
// GPU)
extern "C" void solve(const float *input, const float *kernel, float *output,
                      int input_size, int kernel_size) {
  convolution_1d_kernel<<<input_size - kernel_size + 1 + 255 / 256, 256>>>(
      input, kernel, output, input_size, kernel_size);
  cudaDeviceSynchronize();
}
