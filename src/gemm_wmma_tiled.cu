#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

// BM x BN: output per block; BK: inputs staged per K iteration.
// WM x WN: output per warp, made up of 16 x 16 accumulator fragments.
template <int BM, int BN, int BK, int WM, int WN, int SKEW = 0>
__global__ void gemm_wmma_tiled(const half* __restrict__ A,
                               const half* __restrict__ B,
                               half* __restrict__ C, int M, int N, int K,
                               float alpha, float beta) {
  static_assert(BM % WM == 0 && BN % WN == 0, "Whole warp tiles per block");
  static_assert(WM % 16 == 0 && WN % 16 == 0 && BK % 16 == 0,
                "WMMA dimensions must be multiples of 16");
  static_assert(SKEW >= 0 && SKEW % 16 == 0, "Preserve 32-byte row alignment");
  constexpr int WARPS = (BM / WM) * (BN / WN);
  static_assert(WARPS > 0 && WARPS <= 32, "Valid CUDA block size");
  constexpr int THREADS = WARPS * 32;
  constexpr int ROW_FRAGS = WM / 16, COL_FRAGS = WN / 16;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int warp_row = (warp / (BN / WN)) * WM;
  const int warp_col = (warp % (BN / WN)) * WN;
  const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;

  // Optional row padding changes the shared-memory bank mapping.
  __shared__ __align__(32) half a_tile[BM][BK + SKEW];
  __shared__ __align__(32) half b_tile[BK][BN + SKEW];
  // Each warp owns a separate output scratch tile, reused for its fragments.
  __shared__ __align__(32) float output[WARPS][16][16];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float>
      acc[ROW_FRAGS][COL_FRAGS];
#pragma unroll
  for (int i = 0; i < ROW_FRAGS; ++i)
#pragma unroll
    for (int j = 0; j < COL_FRAGS; ++j)
      wmma::fill_fragment(acc[i][j], 0.0f);

  for (int base = 0; base < K; base += BK) {
    // All warps cooperate on loads; even threads outside C load or zero-pad.
    for (int i = tid; i < BM * BK; i += THREADS) {
      const int row = i / BK, k = i % BK;
      a_tile[row][k] = (row0 + row < M && base + k < K)
          ? A[size_t(row0 + row) * K + base + k] : __float2half(0.0f);
    }
    for (int i = tid; i < BK * BN; i += THREADS) {
      const int k = i / BN, col = i % BN;
      b_tile[k][col] = (base + k < K && col0 + col < N)
          ? B[size_t(base + k) * N + col0 + col] : __float2half(0.0f);
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; k += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
          a_frag[ROW_FRAGS];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
          b_frag[COL_FRAGS];
#pragma unroll
      for (int i = 0; i < ROW_FRAGS; ++i)
        wmma::load_matrix_sync(a_frag[i], &a_tile[warp_row + i * 16][k], BK + SKEW);
#pragma unroll
      for (int j = 0; j < COL_FRAGS; ++j)
        wmma::load_matrix_sync(b_frag[j], &b_tile[k][warp_col + j * 16], BN + SKEW);
#pragma unroll
      for (int i = 0; i < ROW_FRAGS; ++i)
#pragma unroll
        for (int j = 0; j < COL_FRAGS; ++j)
          wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
    }
    __syncthreads(); // All warps finish reading before overwriting A/B tiles.
  }

#pragma unroll
  for (int i = 0; i < ROW_FRAGS; ++i) {
#pragma unroll
    for (int j = 0; j < COL_FRAGS; ++j) {
      wmma::store_matrix_sync(&output[warp][0][0], acc[i][j], 16,
                              wmma::mem_row_major);
      __syncwarp();
      for (int e = lane; e < 256; e += 32) {
        const int row = e / 16, col = e % 16;
        const int m = row0 + warp_row + i * 16 + row;
        const int n = col0 + warp_col + j * 16 + col;
        if (m < M && n < N) {
          const size_t index = size_t(m) * N + n;
          C[index] = __float2half(alpha * output[warp][row][col]
              + (beta == 0.0f ? 0.0f : beta * __half2float(C[index])));
        }
      }
      __syncwarp(); // This warp must finish reading before reusing its scratch.
    }
  }
}

template <int BM, int BN, int BK, int WM, int WN, int SKEW = 0>
void launch_gemm_wmma_tiled(const half* A, const half* B, half* C,
                            int M, int N, int K, float alpha, float beta) {
  constexpr int THREADS = (BM / WM) * (BN / WN) * 32;
  gemm_wmma_tiled<BM, BN, BK, WM, WN, SKEW>
      <<<dim3((N + BN - 1) / BN, (M + BM - 1) / BM), THREADS>>>(
          A, B, C, M, N, K, alpha, beta);
}

// Compile-time overrides for experiments, e.g. -DGEMM_BK=64.
#ifndef GEMM_BM
#define GEMM_BM 64
#endif
#ifndef GEMM_BN
#define GEMM_BN 64
#endif
#ifndef GEMM_BK
#define GEMM_BK 32
#endif
#ifndef GEMM_WM
#define GEMM_WM 32
#endif
#ifndef GEMM_WN
#define GEMM_WN 32
#endif
#ifndef GEMM_SKEW
#define GEMM_SKEW 16
#endif

extern "C" void solve(const half* A, const half* B, half* C,
                      int M, int N, int K, float alpha, float beta) {
  if (M <= 0 || N <= 0) return;
  launch_gemm_wmma_tiled<GEMM_BM, GEMM_BN, GEMM_BK, GEMM_WM, GEMM_WN, GEMM_SKEW>(
      A, B, C, M, N, K, alpha, beta);
}
