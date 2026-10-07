#include <cuda_runtime.h>

constexpr int BLOCK = 1024;
constexpr int MAX_K = 2048;

__global__ void conv1d_kernel(const float *__restrict__ input,
                              const float *__restrict__ kernel,
                              float *__restrict__ output, int input_size,
                              int kernel_size) {
  __shared__ float tile[BLOCK + MAX_K - 1];
  __shared__ float k_s[MAX_K];
  const int base = blockIdx.x * BLOCK;
  const int out_size = input_size - kernel_size + 1;

  for (int i = threadIdx.x; i < BLOCK + kernel_size - 1; i += BLOCK)
    tile[i] = base + i < input_size ? input[base + i] : 0.f;

  for (int i = threadIdx.x; i < kernel_size; i += BLOCK)
    k_s[i] = kernel[i];

  __syncthreads();

  const int tid = base + threadIdx.x;
  if (tid >= out_size)
    return;

  float result = 0;
  for (int i = 0; i < kernel_size; i++)
    result += tile[threadIdx.x + i] * k_s[i];
  output[tid] = result;
}

extern "C" void solve(const float *input, const float *kernel, float *output,
                      int input_size, int kernel_size) {
  const int out_size = input_size - kernel_size + 1;
  conv1d_kernel<<<(out_size + BLOCK - 1) / BLOCK, BLOCK>>>(
      input, kernel, output, input_size, kernel_size);
}
