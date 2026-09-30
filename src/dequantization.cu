#include <cuda_runtime.h>

__global__ void
dequantization_kernel(const float *__restrict__ X, // (M, N)
                      const float *__restrict__ S, // (ceil(M/T), ceil(N/T))
                      float *__restrict__ Y,       // (M, N)
                      const size_t M, const size_t N, const size_t TILE_SIZE) {
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= N || y >= M)
    return;

  Y[y * N + x] =
      X[y * N + x] *
      S[(y / TILE_SIZE) * ((N + TILE_SIZE - 1) / TILE_SIZE) + x / TILE_SIZE];
}

// X, S, Y are device pointers
extern "C" void solve(const float *X, // (M, N)
                      const float *S, // (ceil(M/T), ceil(N/T))
                      float *Y,       // (M, N)
                      int M, int N, int TILE_SIZE) {
  dequantization_kernel<<<dim3((N + 15) / 16, (M + 15) / 16), dim3(16, 16)>>>(
      X, S, Y, M, N, TILE_SIZE);
}
