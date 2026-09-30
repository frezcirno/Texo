#include <cuda_runtime.h>

template <typename T> __host__ __device__ T silu(T x) {
  return x / (1 + exp(-x));
}

__global__ void silu_kernel(const float *input, float *output, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
    return;
  output[tid] = silu(input[tid]);
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
  cudaDeviceSynchronize();
}
