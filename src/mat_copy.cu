#include <cuda_runtime.h>

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, float *B, int N) {
  cudaMemcpy(B, A, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
  // cudaDeviceSynchronize();
}
