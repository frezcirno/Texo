#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr int TILE_SIZE = 16;

__global__ void gemm(const half *__restrict__ A, // (M, K)
                     const half *__restrict__ B, // (K, N)
                     half *__restrict__ C,       // (M, N)
                     int M, int N, int K, float alpha, float beta) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
  __shared__ float B_tile[TILE_SIZE][TILE_SIZE];
  float sum = 0;
  for (int tile = 0; tile < K; tile += TILE_SIZE) {
    const int kx = tile + threadIdx.x;
    const int ky = tile + threadIdx.y;
    A_tile[threadIdx.y][threadIdx.x] =
        (m >= M || kx >= K) ? 0 : float(A[m * K + kx]);
    B_tile[threadIdx.y][threadIdx.x] =
        (n >= N || ky >= K) ? 0 : float(B[ky * N + n]);
    __syncthreads();
    for (int k = 0; k < TILE_SIZE; k++) {
      sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
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
