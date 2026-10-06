#include <cmath>
#include <cstdint>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

// Shared epilogue: dot is the exact zero-point-adjusted integer product.
__device__ __forceinline__ int8_t quantize(int64_t dot, float scale_A,
                                           float scale_B, float scale_C,
                                           int zero_point_C) {
  // Keep the FP32 operations in this order; precombining the scale ratio or
  // promoting it to double can change decisions near a half-integer boundary.
  float acc = float(dot) * scale_A * scale_B / scale_C;
  // Quantization uses round-to-nearest with halfway values rounded to even.
  acc = nearbyintf(acc) + zero_point_C;
  if (acc > 127)
    acc = 127;
  if (acc < -128)
    acc = -128;
  return int8_t(acc);
}

// Fallback for K >= 2^17, where raw INT8 products could overflow INT32.
__global__ void mm_int8_naive(const int8_t *__restrict__ A,
                              const int8_t *__restrict__ B,
                              int8_t *__restrict__ C, int M, int N, int K,
                              float scale_A, float scale_B, float scale_C,
                              int zero_point_A, int zero_point_B,
                              int zero_point_C) {
  const int cx = blockIdx.x * blockDim.x + threadIdx.x;
  const int cy = blockIdx.y * blockDim.y + threadIdx.y;
  if (cy >= M || cx >= N)
    return;
  int64_t bigacc = 0;
  for (int i = 0; i < K; i++) {
    bigacc += int64_t(A[size_t(cy) * K + i] - zero_point_A) *
              (B[size_t(i) * N + cx] - zero_point_B);
  }
  C[size_t(cy) * N + cx] =
      quantize(bigacc, scale_A, scale_B, scale_C, zero_point_C);
}

// row_sum[m] += sum_k A[m,k] over one K chunk per blockIdx.y.
__global__ void row_sums(const int8_t *__restrict__ A, int *__restrict__ sum,
                         int M, int K, int chunk) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int row = blockIdx.x * (blockDim.x / 32) + warp;
  if (row >= M)
    return;
  const int k0 = blockIdx.y * chunk, k1 = min(K, k0 + chunk);
  int acc = 0;
  for (int k = k0 + lane; k < k1; k += 32)
    acc += A[size_t(row) * K + k];
  for (int offset = 16; offset > 0; offset /= 2)
    acc += __shfl_down_sync(0xffffffff, acc, offset);
  if (lane == 0)
    atomicAdd(&sum[row], acc);
}

// col_sum[n] += sum_k B[k,n] over one K chunk per blockIdx.y.
__global__ void col_sums(const int8_t *__restrict__ B, int *__restrict__ sum,
                         int K, int N, int chunk) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= N)
    return;
  const int k0 = blockIdx.y * chunk, k1 = min(K, k0 + chunk);
  int acc = 0;
  for (int k = k0; k < k1; ++k)
    acc += B[size_t(k) * N + col];
  atomicAdd(&sum[col], acc);
}

// Stage a ROWS x COLS byte tile in 16-byte groups. Out-of-bounds bytes are
// zero, so they add nothing to the raw product sum.
template <int ROWS, int COLS, int STRIDE, int THREADS>
__device__ __forceinline__ void stage_tile(int8_t (&tile)[ROWS][STRIDE],
                                           const int8_t *input, int rows,
                                           int cols, int row0, int col0) {
  constexpr int PACK = 16;
  for (int pack = threadIdx.x; pack < ROWS * (COLS / PACK); pack += THREADS) {
    const int row = pack / (COLS / PACK);
    const int col = (pack % (COLS / PACK)) * PACK;
    const int global_row = row0 + row, global_col = col0 + col;
    if (global_row < rows && global_col + PACK <= cols) {
      const int8_t *src = input + size_t(global_row) * cols + global_col;
      if ((reinterpret_cast<uintptr_t>(src) & 15) == 0) {
#if __CUDA_ARCH__ >= 800
        __pipeline_memcpy_async(&tile[row][col], src, 16);
#else
        *reinterpret_cast<int4 *>(&tile[row][col]) =
            *reinterpret_cast<const int4 *>(src);
#endif
        continue;
      }
    }
    // Partial or unaligned groups use byte loads.
#pragma unroll
    for (int e = 0; e < PACK; ++e) {
      tile[row][col + e] =
          (global_row < rows && global_col + e < cols)
              ? input[size_t(global_row) * cols + global_col + e]
              : int8_t(0);
    }
  }
}

constexpr int BM = 128, BN = 128, BK = 64, WM = 64, WN = 32, SKEW = 16;
constexpr int WARPS = (BM / WM) * (BN / WN);
constexpr int THREADS = WARPS * 32;

// Raw A*B on INT8 tensor cores with INT32 accumulation (exact for K < 2^17),
// then the zero-point correction in INT64:
//   sum (a-za)(b-zb) = sum ab - zb*rowA - za*colB + K*za*zb
__global__ void __launch_bounds__(THREADS)
    mm_int8_wmma(const int8_t *__restrict__ A, const int8_t *__restrict__ B,
                 int8_t *__restrict__ C, const int *__restrict__ row_sum,
                 const int *__restrict__ col_sum, int M, int N, int K,
                 float scale_A, float scale_B, float scale_C, int zero_point_A,
                 int zero_point_B, int zero_point_C) {
  constexpr int FM = WM / 16, FN = WN / 16;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int warp_row = (warp / (BN / WN)) * WM;
  const int warp_col = (warp % (BN / WN)) * WN;

  __shared__ __align__(128) int8_t A_tile[2][BM][BK + SKEW];
  __shared__ __align__(128) int8_t B_tile[2][BK][BN + SKEW];
  __shared__ __align__(32) int output[WARPS][16][16];

  const int tiles_n = (N + BN - 1) / BN;
  const int row0 = (blockIdx.x / tiles_n) * BM;
  const int col0 = (blockIdx.x % tiles_n) * BN;
  const int tiles_k = (K + BK - 1) / BK;

  wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; ++i) {
#pragma unroll
    for (int j = 0; j < FN; ++j) {
      wmma::fill_fragment(acc[i][j], 0);
    }
  }

  if (tiles_k > 0) {
    stage_tile<BM, BK, BK + SKEW, THREADS>(A_tile[0], A, M, K, row0, 0);
    stage_tile<BK, BN, BN + SKEW, THREADS>(B_tile[0], B, K, N, 0, col0);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
  }

  for (int tk = 0; tk < tiles_k; ++tk) {
    const int cur = tk & 1, next = cur ^ 1;
    if (tk + 1 < tiles_k) {
      const int k0 = (tk + 1) * BK;
      stage_tile<BM, BK, BK + SKEW, THREADS>(A_tile[next], A, M, K, row0, k0);
      stage_tile<BK, BN, BN + SKEW, THREADS>(B_tile[next], B, K, N, k0, col0);
      __pipeline_commit();
    }
#pragma unroll
    for (int k = 0; k < BK; k += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major>
          a_frag[FM];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major>
          b_frag[FN];
#pragma unroll
      for (int i = 0; i < FM; ++i) {
        wmma::load_matrix_sync(a_frag[i],
                               reinterpret_cast<const signed char *>(
                                   &A_tile[cur][warp_row + i * 16][k]),
                               BK + SKEW);
      }
#pragma unroll
      for (int j = 0; j < FN; ++j) {
        wmma::load_matrix_sync(b_frag[j],
                               reinterpret_cast<const signed char *>(
                                   &B_tile[cur][k][warp_col + j * 16]),
                               BN + SKEW);
      }
#pragma unroll
      for (int i = 0; i < FM; ++i) {
#pragma unroll
        for (int j = 0; j < FN; ++j) {
          wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
      }
    }
    // All next copies have landed and every warp has finished reading cur.
    __pipeline_wait_prior(0);
    __syncthreads();
  }

  // Each lane writes eight consecutive outputs of a 16x16 fragment.
  const int r = lane / 2, c = (lane % 2) * 8;
  const int64_t offset = int64_t(K) * zero_point_A * zero_point_B;
#pragma unroll
  for (int i = 0; i < FM; ++i) {
    const int m = row0 + warp_row + i * 16 + r;
    const int64_t row_term =
        m < M ? offset - int64_t(zero_point_B) * row_sum[m] : 0;
#pragma unroll
    for (int j = 0; j < FN; ++j) {
      wmma::store_matrix_sync(&output[warp][0][0], acc[i][j], 16,
                              wmma::mem_row_major);
      __syncwarp();
      const int n0 = col0 + warp_col + j * 16 + c;
      if (m < M) {
        int8_t out[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int n = n0 + e;
          const int64_t dot = n < N ? output[warp][r][c + e] + row_term -
                                          int64_t(zero_point_A) * col_sum[n]
                                    : 0;
          out[e] = quantize(dot, scale_A, scale_B, scale_C, zero_point_C);
        }
        int8_t *dst = C + size_t(m) * N + n0;
        if (n0 + 8 <= N && (reinterpret_cast<uintptr_t>(dst) & 7) == 0) {
          *reinterpret_cast<int2 *>(dst) = *reinterpret_cast<int2 *>(out);
        } else {
          for (int e = 0; e < 8 && n0 + e < N; ++e) {
            dst[e] = out[e];
          }
        }
      }
      __syncwarp();
    }
  }
}

// A, B, C are device pointers
extern "C" void solve(const int8_t *A, // (M, K)
                      const int8_t *B, // (K, N)
                      int8_t *C,       // (M, N)
                      int M, int N, int K, float scale_A, float scale_B,
                      float scale_C, int zero_point_A, int zero_point_B,
                      int zero_point_C) {
  if (K >= (1 << 17)) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);
    mm_int8_naive<<<gridDim, blockDim>>>(A, B, C, M, N, K, scale_A, scale_B,
                                         scale_C, zero_point_A, zero_point_B,
                                         zero_point_C);
    return;
  }

  // Scratch for row/column sums, reused across calls.
  static int *sums = nullptr;
  static size_t capacity = 0;
  const size_t needed = size_t(M) + N;
  if (needed > capacity) {
    cudaFree(sums);
    cudaMalloc(&sums, needed * sizeof(int));
    capacity = needed;
  }

  int *row_sum = sums, *col_sum = sums + M;
  cudaMemsetAsync(sums, 0, needed * sizeof(int));

  constexpr int CHUNK = 256;
  const int chunks = (K + CHUNK - 1) / CHUNK;

  if (K > 0 && zero_point_B != 0) {
    row_sums<<<dim3((M + 7) / 8, chunks), 256>>>(A, row_sum, M, K, CHUNK);
  }

  if (K > 0 && zero_point_A != 0) {
    col_sums<<<dim3((N + 255) / 256, chunks), 256>>>(B, col_sum, K, N, CHUNK);
  }

  const unsigned tiles = unsigned((M + BM - 1) / BM) * ((N + BN - 1) / BN);
  mm_int8_wmma<<<tiles, THREADS>>>(A, B, C, row_sum, col_sum, M, N, K, scale_A,
                                   scale_B, scale_C, zero_point_A, zero_point_B,
                                   zero_point_C);
}
