#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

constexpr int TILE_SIZE = 16;

// Exactly one warp per block; one block computes a 16 x 16 output tile.
__global__ void gemm(const half *A, const half *B, half *C, int M, int N, int K,
                     float alpha, float beta) {
  __shared__ __align__(32) half A_tile[TILE_SIZE][TILE_SIZE];
  __shared__ __align__(32) half B_tile[TILE_SIZE][TILE_SIZE];
  __shared__ __align__(32) float sum[TILE_SIZE][TILE_SIZE];

  const int n0 = blockIdx.x * TILE_SIZE;
  const int m0 = blockIdx.y * TILE_SIZE;

  wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half,
                 wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half,
                 wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  for (int tile = 0; tile < K; tile += TILE_SIZE) {
    // 256 elements / 32 lanes = 8 elements per lane, for each input tile.
    for (int i = threadIdx.x; i < 256; i += 32) {
      const int y = i / TILE_SIZE, x = i % TILE_SIZE;
      const int m = m0 + y, n = n0 + x;
      const int kx = tile + x, ky = tile + y;
      A_tile[y][x] = (m >= M || kx >= K) ? 0 : float(A[m * K + kx]);
      B_tile[y][x] = (ky >= K || n >= N) ? 0 : float(B[ky * N + n]);
    }
    __syncthreads();
    wmma::load_matrix_sync(a_frag, &A_tile[0][0], TILE_SIZE);
    wmma::load_matrix_sync(b_frag, &B_tile[0][0], TILE_SIZE);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    __syncthreads();
  }

  // Convert the opaque fragment mapping into ordinary row-major coordinates.
  wmma::store_matrix_sync(&sum[0][0], acc, TILE_SIZE, wmma::mem_row_major);
  __syncthreads();

  for (int i = threadIdx.x; i < 256; i += 32) {
    const int y = i / TILE_SIZE, x = i % TILE_SIZE;
    const int m = m0 + y, n = n0 + x;
    if (m < M && n < N) {
      const size_t index = m * N + n;
      C[index] =
          alpha * sum[y][x] + (beta == 0.0f ? 0.0f : beta * float(C[index]));
    }
  }
}

extern "C" void solve(const half *A, const half *B, half *C, int M, int N,
                      int K, float alpha, float beta) {
  if (M <= 0 || N <= 0)
    return;
  // 32 thread per block, process 16 * 16 tile
  gemm<<<dim3((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE),
         32>>>(A, B, C, M, N, K, alpha, beta);
}
