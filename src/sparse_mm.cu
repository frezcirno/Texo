#include <cuda_runtime.h>

__global__ void gemm(const float *__restrict__ A, // (M, N)
                     const float *__restrict__ B, // (N, K)
                     float *__restrict__ C,       // (M, K)
                     const size_t M, const size_t N, const size_t K) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (y >= M || x >= K)
    return;
  float sum = 0.0f;
  for (int i = 0; i < N; i++) {
    sum += float(A[y * N + i]) * float(B[i * K + x]);
  }
  C[y * K + x] = sum;
}

// A, B, and C are device pointers
extern "C" void solve(const float *A, // (M, N)
                      const float *B, // (N, K)
                      float *C,       // (M, K)
                      int M, int N, int K, int nnz) {

  (void)nnz;
  if (M <= 0 || K <= 0)
    return;
  gemm<<<dim3((K + 15) / 16, (M + 15) / 16), //
         dim3(16, 16)>>>(A, B, C, M, N, K);
}
