#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float *X, float *Y, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
    return;
  Y[tid] = 1 / (1 + exp(-X[tid]));
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *X, float *Y, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
  cudaDeviceSynchronize();
}
