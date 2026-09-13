#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

// Each thread owns disjoint groups of eight half elements. Shared rows and
// group starts are 16-byte aligned; global row strides/pointers need not be.
template <int ROWS, int COLS, int STRIDE, int THREADS>
__device__ __forceinline__ void
stage_input_tile(half (&tile)[ROWS][STRIDE], const half *input, int rows,
                 int cols, size_t row0, size_t col0) {
  constexpr int PACK = 8;
  for (int pack = threadIdx.x; pack < ROWS * (COLS / PACK); pack += THREADS) {
    const int row = pack / (COLS / PACK);
    const int col = (pack % (COLS / PACK)) * PACK;
    const size_t global_row = row0 + row, global_col = col0 + col;
#if __CUDA_ARCH__ >= 800
    if (global_row < size_t(rows) && global_col + PACK <= size_t(cols)) {
      const half *src = input + global_row * cols + global_col;
      if ((reinterpret_cast<uintptr_t>(src) & 15) == 0) {
        __pipeline_memcpy_async(&tile[row][col], src, 16);
        continue;
      }
    }
#endif
    // Partial/unaligned groups, and pre-Ampere builds, use ordinary loads.
    // No source pointer is formed for an out-of-bounds row or column.
#pragma unroll
    for (int e = 0; e < PACK; ++e) {
      tile[row][col + e] =
          (global_row < size_t(rows) && global_col + e < size_t(cols))
              ? input[global_row * cols + global_col + e]
              : half(0.0f);
    }
  }
}

template <int BM, int BN, int BK, int WM, int WN, int SKEW>
__global__ void gemm_wmma_tiled_pipeline(const half *__restrict__ A,
                                         const half *__restrict__ B,
                                         half *__restrict__ C, int M, int N,
                                         int K, float alpha, float beta) {
  static_assert(BM > 0 && BN > 0 && BK > 0 && WM > 0 && WN > 0,
                "Tile dimensions must be positive");
  static_assert(BM % WM == 0 && BN % WN == 0, "Whole warp tiles per block");
  static_assert(WM % 16 == 0 && WN % 16 == 0 && BK % 16 == 0,
                "WMMA dimensions must be multiples of 16");
  static_assert(SKEW >= 0 && SKEW % 16 == 0, "Preserve 32-byte row alignment");
  constexpr int WARP_NUM = (BM / WM) * (BN / WN);
  static_assert(WARP_NUM > 0 && WARP_NUM <= 32, "Valid CUDA block size");
  constexpr int BLOCK_SIZE = WARP_NUM * 32;
  constexpr int WARP_FRAG_NUM_M = WM / 16, WARP_FRAG_NUM_N = WN / 16;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int warp_in_block_row = (warp / (BN / WN)) * WM;
  const int warp_in_block_col = (warp % (BN / WN)) * WN;

  __shared__ __align__(32) half A_tile[2][BM][BK + SKEW];
  __shared__ __align__(32) half B_tile[2][BK][BN + SKEW];
  __shared__ __align__(32) float output[WARP_NUM][16][16];

  const size_t tiles_m = (size_t(M) + BM - 1) / BM;
  const size_t tiles_n = (size_t(N) + BN - 1) / BN;
  const size_t tiles_k = (size_t(K) + BK - 1) / BK;
  // Flatten output tiles so tall matrices do not exceed the grid.y limit.
  for (size_t tile_id = blockIdx.x; tile_id < tiles_m * tiles_n;
       tile_id += gridDim.x) {
    const size_t row0 = (tile_id / tiles_n) * BM;
    const size_t col0 = (tile_id % tiles_n) * BN;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WARP_FRAG_NUM_M]
                                                            [WARP_FRAG_NUM_N];
#pragma unroll
    for (int i = 0; i < WARP_FRAG_NUM_M; ++i)
#pragma unroll
      for (int j = 0; j < WARP_FRAG_NUM_N; ++j)
        wmma::fill_fragment(acc[i][j], 0.0f);

    if (tiles_k > 0) {
      // Prologue: the first tile must be ready before starting WMMA.
      stage_input_tile<BM, BK, BK + SKEW, BLOCK_SIZE>(A_tile[0], A, M, K, row0,
                                                      0);
      stage_input_tile<BK, BN, BN + SKEW, BLOCK_SIZE>(B_tile[0], B, K, N, 0,
                                                      col0);
      __syncwarp(); // Reconverge after per-thread alignment/bounds branches.
      __pipeline_commit();
      __pipeline_wait_prior(0);
      __syncthreads();
    }

    for (size_t t = 0; t < tiles_k; ++t) {
      const int cur = int(t & 1), next = cur ^ 1;
      if (t + 1 < tiles_k) {
        const size_t tb = (t + 1) * BK;
        stage_input_tile<BM, BK, BK + SKEW, BLOCK_SIZE>(A_tile[next], A, M, K,
                                                        row0, tb);
        stage_input_tile<BK, BN, BN + SKEW, BLOCK_SIZE>(B_tile[next], B, K, N,
                                                        tb, col0);
        __syncwarp();
        __pipeline_commit();
      }

      // Copies into next run concurrently with WMMA reading cur.
#pragma unroll
      for (int k = 0; k < BK; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
            a_frag[WARP_FRAG_NUM_M];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
            b_frag[WARP_FRAG_NUM_N];
#pragma unroll
        for (int i = 0; i < WARP_FRAG_NUM_M; ++i)
          wmma::load_matrix_sync(a_frag[i],
                                 &A_tile[cur][warp_in_block_row + i * 16][k],
                                 BK + SKEW);
#pragma unroll
        for (int j = 0; j < WARP_FRAG_NUM_N; ++j)
          wmma::load_matrix_sync(b_frag[j],
                                 &B_tile[cur][k][warp_in_block_col + j * 16],
                                 BN + SKEW);
#pragma unroll
        for (int i = 0; i < WARP_FRAG_NUM_M; ++i)
#pragma unroll
          for (int j = 0; j < WARP_FRAG_NUM_N; ++j)
            wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
      }

      // Each thread waits for its own copies. The block barrier then ensures
      // all next data is ready AND every warp has stopped reading cur.
      __pipeline_wait_prior(0);
      __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < WARP_FRAG_NUM_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_FRAG_NUM_N; ++j) {
        wmma::store_matrix_sync(&output[warp][0][0], acc[i][j], 16,
                                wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
          const int row = e / 16, col = e % 16;
          const size_t m = row0 + warp_in_block_row + i * 16 + row;
          const size_t n = col0 + warp_in_block_col + j * 16 + col;
          if (m < size_t(M) && n < size_t(N)) {
            const size_t index = m * N + n;
            C[index] = alpha * output[warp][row][col] +
                       (beta == 0.0f ? 0.0f : beta * __half2float(C[index]));
          }
        }
        __syncwarp();
      }
    }
  }
}

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

// Row-major FP16 A[M,K], B[K,N], C[M,N]; FP32 accumulation and alpha/beta.
// Nonnegative dimensions; M/N=0 is a no-op, K=0 scales C by beta.
// A/B/C need only half alignment. sm_80+ uses async copies where possible;
// sm_75 builds use synchronous staging with the same numerical contract.
extern "C" void solve(const half *A, const half *B, half *C, int M, int N,
                      int K, float alpha, float beta) {
  if (M <= 0 || N <= 0)
    return;
  constexpr int THREADS = (GEMM_BM / GEMM_WM) * (GEMM_BN / GEMM_WN) * 32;
  const size_t tiles = ((size_t(M) + GEMM_BM - 1) / GEMM_BM) *
                       ((size_t(N) + GEMM_BN - 1) / GEMM_BN);
  const unsigned blocks = unsigned(tiles < 2147483647u ? tiles : 2147483647u);
  gemm_wmma_tiled_pipeline<GEMM_BM, GEMM_BN, GEMM_BK, GEMM_WM, GEMM_WN,
                           GEMM_SKEW>
      <<<blocks, THREADS>>>(A, B, C, M, N, K, alpha, beta);
}
