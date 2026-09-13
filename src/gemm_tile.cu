#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr int TILE_SIZE = 16;

__global__ void gemm(const half *__restrict__ A, // (M, K)
                     const half *__restrict__ B, // (K, N)
                     half *__restrict__ C,       // (M, N)
                     int M, int N, int K, float alpha, float beta) {
  __shared__ half A_tile[TILE_SIZE][TILE_SIZE];
  __shared__ half B_tile[TILE_SIZE][TILE_SIZE];
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  float sum = 0;
  for (int tb = 0; tb < K; tb += TILE_SIZE) {
    const int kx = tb + threadIdx.x;
    const int ky = tb + threadIdx.y;
    A_tile[threadIdx.y][threadIdx.x] =
        (m >= M || kx >= K) ? half(0) : A[m * K + kx];
    B_tile[threadIdx.y][threadIdx.x] =
        (n >= N || ky >= K) ? half(0) : B[ky * N + n];
    __syncthreads();
    for (int k = 0; k < TILE_SIZE; k++) {
      sum += float(A_tile[threadIdx.y][k]) * float(B_tile[k][threadIdx.x]);
    }
    __syncthreads();
  }
  if (m < M && n < N) {
    const size_t index = m * N + n;
    C[index] = alpha * sum + (beta == 0.0f ? 0.0f : beta * float(C[index]));
  }
}

// A, B, and C are device pointers
extern "C" void solve(const half *A, // (M, K)
                      const half *B, // (K, N)
                      half *C,       // (M, N)
                      int M, int N, int K, float alpha, float beta) {

  if (M <= 0 || N <= 0)
    return;
  gemm<<<dim3((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE), //
         dim3(TILE_SIZE, TILE_SIZE)>>>(A, B, C, M, N, K, alpha, beta);
}
