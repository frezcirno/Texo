#include <cuda_runtime.h>

__global__ void geglu_kernel(const float *input, float *output, int halfN) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= halfN)
    return;
  auto x1 = input[tid];
  auto x2 = input[halfN + tid];
  output[tid] = x1 * x2 * 0.5 * (1 + erf(x2 / sqrtf(2)));
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N) {
  int halfN = N / 2;
  int threadsPerBlock = 256;
  int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

  geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
  cudaDeviceSynchronize();
}
