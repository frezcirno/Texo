#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void batched_mm(const half *__restrict__ A, // (BATCH, M, K)
                           const half *__restrict__ B, // (BATCH, K, N)
                           half *__restrict__ C,       // (BATCH, M, N)
                           size_t BATCH, size_t M, size_t N, size_t K) {
  const size_t cx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t cy = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t cz = blockIdx.z * blockDim.z + threadIdx.z;
  if (cy >= M || cx >= N || cz >= BATCH)
    return;
  float sum = 0.0f;
  for (int i = 0; i < K; i++) {
    sum +=
        float(A[cz * M * K + cy * K + i]) * float(B[cz * K * N + i * N + cx]);
  }
  C[cz * M * N + cy * N + cx] += sum;
}

// A, B, and C are device pointers
extern "C" void solve(const half *A, // (BATCH, M, K)
                      const half *B, // (BATCH, K, N)
                      half *C,       // (BATCH, M, N)
                      int BATCH, int M, int N, int K) {
  if (M <= 0 || N <= 0 || BATCH <= 0)
    return;
  batched_mm<<<dim3((N + 3) / 4, (M + 7) / 8, (BATCH + 7) / 8), //
               dim3(4, 8, 8)>>>(A, B, C, BATCH, M, N, K);
}
