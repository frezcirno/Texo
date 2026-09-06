#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void gemm(const half *__restrict__ A, // (M, K)
                     const half *__restrict__ B, // (K, N)
                     half *__restrict__ C,       // (M, N)
                     int M, int N, int K, float alpha, float beta) {
  //
  const int cx = blockIdx.x * blockDim.x + threadIdx.x;
  const int cy = blockIdx.y * blockDim.y + threadIdx.y;
  if (cy >= M || cx >= N)
    return;
  half sum = 0;
  for (int i = 0; i < K; i++) {
    sum += A[cy * K + i] * B[i * N + cx];
  }
  C[cy * N + cx] = alpha * float(sum) + beta * float(C[cy * N + cx]);
}

// A, B, and C are device pointers
extern "C" void solve(const half *A, // (M, K)
                      const half *B, // (K, N)
                      half *C,       // (M, N)
                      int M, int N, int K, float alpha, float beta) {

  gemm<<<dim3((N + 15) / 16, (M + 15) / 16), //
         dim3(16, 16)>>>(A, B, C, M, N, K, alpha, beta);
}
