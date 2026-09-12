#include <cuda_runtime.h>

__global__ void batched_mm(const float *__restrict__ A, // (BATCH, M, K)
                           const float *__restrict__ B, // (BATCH, K, N)
                           float *__restrict__ C,       // (BATCH, M, N)
                           int BATCH, int M, int N, int K) {
  const int cx = blockIdx.x * blockDim.x + threadIdx.x;
  const int cy = blockIdx.y * blockDim.y + threadIdx.y;
  const int cz = blockIdx.z * blockDim.z + threadIdx.z;
  if (cy >= M || cx >= N || cz >= BATCH)
    return;
  float sum = 0.0f;
  for (int i = 0; i < K; i++) {
    sum += A[cz * M * K + cy * K + i] * B[cz * K * N + i * N + cx];
  }
  const size_t index = cz * M * N + cy * N + cx;
  C[index] += sum;
}

// A, B, and C are device pointers
extern "C" void solve(const float *A, // (BATCH, M, K)
                      const float *B, // (BATCH, K, N)
                      float *C,       // (BATCH, M, N)
                      int BATCH, int M, int N, int K) {
  if (M <= 0 || N <= 0 || BATCH <= 0)
    return;
  batched_mm<<<dim3((N + 3) / 4, (M + 7) / 8, (BATCH + 7) / 8), //
               dim3(4, 8, 8)>>>(A, B, C, BATCH, M, N, K);
}
