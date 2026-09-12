#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float *input, float *output, int halfN) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= halfN)
    return;
  float x1 = input[tid];
  float x2 = input[halfN + tid];
  output[tid] = x2 * x1 / (1 + exp(-x1));
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N) {
  int halfN = N / 2;
  int threadsPerBlock = 256;
  int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

  swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
  cudaDeviceSynchronize();
}
