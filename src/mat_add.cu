#include <cuda_runtime.h>

__global__ void matrix_add(const float *__restrict__ A,
                           const float *__restrict__ B, float *__restrict__ C,
                           int N) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= N || y >= N)
    return;
  auto i = y * N + x;
  C[i] = A[i] + B[i];
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int N) {
  matrix_add<<<dim3((N + 31) / 32, (N + 7) / 8), dim3(32, 8)>>>(A, B, C, N);
  cudaDeviceSynchronize();
}
