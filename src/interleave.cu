#include <cuda_runtime.h>

__global__ void interleave_kernel(const float *A, const float *B, float *output,
                                  int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= 2 * N)
    return;
  output[tid] = tid % 2 == 0 ? A[tid / 2] : B[tid / 2];
}

// A, B, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *output, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (2 * N + threadsPerBlock - 1) / threadsPerBlock;

  interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output, N);
  cudaDeviceSynchronize();
}
