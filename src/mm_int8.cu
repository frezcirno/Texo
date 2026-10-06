#include <cmath>
#include <cstdint>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <type_traits>

namespace wmma = nvcuda::wmma;

// Shared epilogue: dot is the exact zero-point-adjusted integer product,
// converted to FP32 by the caller (round to nearest).
__device__ __forceinline__ int8_t quantize(float dot, float scale_A,
                                           float scale_B, float scale_C,
                                           int zero_point_C) {
  // Keep the FP32 operations in this order; precombining the scale ratio or
  // promoting it to double can change decisions near a half-integer boundary.
  float acc = dot * scale_A * scale_B / scale_C;
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
  const int8_t *a = A + size_t(row) * K;
  // Whole aligned 16-byte groups use dp4a; the rest (or all of an unaligned
  // row) uses byte loads.
  const bool aligned = ((reinterpret_cast<uintptr_t>(a + k0)) & 15) == 0;
  const int vec_end = aligned ? k0 + (k1 - k0) / 16 * 16 : k0;
  int acc = 0;
  for (int k = k0 + lane * 16; k < vec_end; k += 32 * 16) {
    const int4 v = *reinterpret_cast<const int4 *>(a + k);
    acc = __dp4a(v.x, 0x01010101, acc);
    acc = __dp4a(v.y, 0x01010101, acc);
    acc = __dp4a(v.z, 0x01010101, acc);
    acc = __dp4a(v.w, 0x01010101, acc);
  }
  for (int k = vec_end + lane; k < k1; k += 32)
    acc += a[k];
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

// sm_80+ path: Bt[Np][Kp] = B^T, zero padded to whole tiles, so both MMA
// operands are K-contiguous and B can use ldmatrix like A. Also accumulates
// col_sum[n] = sum_k B[k,n].
constexpr int TT = 64;
__global__ void transpose_b(const int8_t *__restrict__ B,
                            int8_t *__restrict__ Bt, int *__restrict__ col_sum,
                            int K, int N, int Kp) {
  __shared__ int8_t tile[TT][TT + 4];
  const int k0 = blockIdx.y * TT, n0 = blockIdx.x * TT;
  {
    const int r = threadIdx.x / 4, c = (threadIdx.x % 4) * 16;
    const int k = k0 + r, n = n0 + c;
    const int8_t *src = B + size_t(k) * N + n;
    if (k < K && n + 16 <= N && (reinterpret_cast<uintptr_t>(src) & 15) == 0) {
      const int4 v = *reinterpret_cast<const int4 *>(src);
      const int8_t *bytes = reinterpret_cast<const int8_t *>(&v);
#pragma unroll
      for (int e = 0; e < 16; ++e)
        tile[r][c + e] = bytes[e];
    } else {
#pragma unroll
      for (int e = 0; e < 16; ++e)
        tile[r][c + e] = (k < K && n + e < N) ? src[e] : int8_t(0);
    }
  }
  __syncthreads();
  const int n = threadIdx.x / 4, c = (threadIdx.x % 4) * 16;
  int4 v;
  int8_t *bytes = reinterpret_cast<int8_t *>(&v);
  int sum = 0;
#pragma unroll
  for (int e = 0; e < 16; ++e) {
    bytes[e] = tile[c + e][n];
    sum += bytes[e];
  }
  *reinterpret_cast<int4 *>(Bt + size_t(n0 + n) * Kp + k0 + c) = v;
  // The four lanes of one column are adjacent.
  sum += __shfl_xor_sync(0xffffffff, sum, 1);
  sum += __shfl_xor_sync(0xffffffff, sum, 2);
  if (threadIdx.x % 4 == 0 && n0 + n < N)
    atomicAdd(&col_sum[n0 + n], sum);
}

#ifndef MMA_BM
#define MMA_BM 128
#endif
#ifndef MMA_BN
#define MMA_BN 128
#endif
#ifndef MMA_WM
#define MMA_WM 64
#endif
#ifndef MMA_WN
#define MMA_WN 32
#endif
#ifndef MMA_STAGES
#define MMA_STAGES 4
#endif
constexpr int MBM = MMA_BM, MBN = MMA_BN, MBK = 64, STAGES = MMA_STAGES;
constexpr int MWM = MMA_WM, MWN = MMA_WN;
constexpr int MWARPS = (MBM / MWM) * (MBN / MWN);
constexpr int MTHREADS = MWARPS * 32;
constexpr int STAGE_BYTES = (MBM + MBN) * MBK;
constexpr int MMA_SMEM = STAGES * STAGE_BYTES;
static_assert(MBK == 64, "swizzle assumes four 16-byte chunks per row");
static_assert(MBM * (MBN + 16) <= MMA_SMEM, "output staging fits");

// Rows are 64 bytes. XOR the 16-byte chunk with (row / 2) % 4 so the eight
// rows read by one ldmatrix phase cover all 32 banks.
__device__ __forceinline__ int swizzle(int row, int chunk) {
  return row * MBK + ((chunk ^ ((row >> 1) & 3)) << 4);
}

__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], unsigned addr) {
#if __CUDA_ARCH__ >= 800
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#endif
}

__device__ __forceinline__ void mma_s8(int (&c)[4], const unsigned (&a)[4],
                                       unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// NARROW: K <= 33025, so |dot| <= 255*255*K < 2^31 and the zero-point
// correction is exact in wrapping 32-bit arithmetic.
template <bool NARROW>
__global__ void __launch_bounds__(MTHREADS)
    mm_int8_mma(const int8_t *__restrict__ A, const int8_t *__restrict__ Bt,
                int8_t *__restrict__ C, const int *__restrict__ row_sum,
                const int *__restrict__ col_sum, int M, int N, int K, int Kp,
                float scale_A, float scale_B, float scale_C, int zero_point_A,
                int zero_point_B, int zero_point_C) {
#if __CUDA_ARCH__ >= 800
  extern __shared__ __align__(128) int8_t smem[];
  constexpr int MI = MWM / 16, NI = MWN / 8;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int warp_row = (warp / (MBN / MWN)) * MWM;
  const int warp_col = (warp % (MBN / MWN)) * MWN;
  const int tiles_n = (N + MBN - 1) / MBN;
  const int row0 = (blockIdx.x / tiles_n) * MBM;
  const int col0 = (blockIdx.x % tiles_n) * MBN;
  const int tiles_k = Kp / MBK;
  const unsigned smem_base = unsigned(__cvta_generic_to_shared(smem));

  auto load_stage = [&](int stage, int k0) {
    int8_t *a_tile = smem + stage * STAGE_BYTES;
    int8_t *b_tile = a_tile + MBM * MBK;
    for (int i = threadIdx.x; i < MBM * 4; i += MTHREADS) {
      const int row = i / 4, chunk = i % 4;
      const int m = row0 + row, k = k0 + chunk * 16;
      int8_t *dst = a_tile + swizzle(row, chunk);
      const int8_t *src = A + size_t(m) * K + k;
      if (m < M && k + 16 <= K &&
          (reinterpret_cast<uintptr_t>(src) & 15) == 0) {
        __pipeline_memcpy_async(dst, src, 16);
      } else {
        // K tail, rows past M, or an unaligned A: zero-filled byte loads.
#pragma unroll
        for (int e = 0; e < 16; ++e) {
          dst[e] = (m < M && k + e < K) ? src[e] : int8_t(0);
        }
      }
    }
    // Bt is padded to whole tiles and cudaMalloc-aligned.
    for (int i = threadIdx.x; i < MBN * 4; i += MTHREADS) {
      const int row = i / 4, chunk = i % 4;
      __pipeline_memcpy_async(b_tile + swizzle(row, chunk),
                              Bt + size_t(col0 + row) * Kp + k0 + chunk * 16,
                              16);
    }
  };

  int acc[MI][NI][4] = {};

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < tiles_k) {
      load_stage(s, s * MBK);
    }
    __pipeline_commit();
  }

  for (int kt = 0; kt < tiles_k; ++kt) {
    __pipeline_wait_prior(STAGES - 2);
    __syncthreads();
    // The stage being refilled was consumed in iteration kt-1; the barrier
    // above guarantees every warp has finished with it.
    const int fetch = kt + STAGES - 1;
    if (fetch < tiles_k) {
      load_stage(fetch % STAGES, fetch * MBK);
    }
    __pipeline_commit();

    const unsigned a_base = smem_base + (kt % STAGES) * STAGE_BYTES;
    const unsigned b_base = a_base + MBM * MBK;
#pragma unroll
    for (int kc = 0; kc < MBK / 16; kc += 2) {
      // ldmatrix lane -> row address: matrices are (rows 0-7, chunk kc),
      // (rows 8-15, kc), (rows 0-7, kc+1), (rows 8-15, kc+1) for A, which
      // is exactly the m16n8k32 A fragment order a0..a3.
      unsigned a[MI][4], b[NI][2];
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        const int row = warp_row + mi * 16 + (lane % 8) + ((lane >> 3) & 1) * 8;
        ldmatrix_x4(a[mi], a_base + swizzle(row, kc + (lane >> 4)));
      }
      // For B: (n 0-7, kc), (n 0-7, kc+1), (n 8-15, kc), (n 8-15, kc+1),
      // i.e. b0/b1 of two adjacent n8 tiles.
#pragma unroll
      for (int p = 0; p < NI / 2; ++p) {
        const int row = warp_col + p * 16 + (lane % 8) + (lane >> 4) * 8;
        unsigned r[4];
        ldmatrix_x4(r, b_base + swizzle(row, kc + ((lane >> 3) & 1)));
        b[2 * p][0] = r[0];
        b[2 * p][1] = r[1];
        b[2 * p + 1][0] = r[2];
        b[2 * p + 1][1] = r[3];
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
          mma_s8(acc[mi][ni], a[mi], b[ni][0], b[ni][1]);
        }
      }
    }
  }
  __pipeline_wait_prior(0);
  __syncthreads();

  // Accumulator fragment: c[h*2+j] is row g+8h, column 2t+j of the n8 tile.
  // Quantize in registers, then stage the INT8 tile for coalesced stores.
  constexpr int OUT_STRIDE = MBN + 16;
  int8_t *out = smem;
  const int g = lane / 4, t = lane % 4;
  using Wide = typename std::conditional<NARROW, unsigned, int64_t>::type;
  using Signed = typename std::conditional<NARROW, int, int64_t>::type;
  const Wide offset = Wide(K) * Wide(zero_point_A) * Wide(zero_point_B);
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = warp_row + mi * 16 + g + h * 8;
      const int m = row0 + row;
      const Wide row_term =
          m < M ? offset - Wide(zero_point_B) * Wide(row_sum[m]) : 0;
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) {
        // Loaded per use: keeping these live beside acc causes spills.
        const int n = col0 + warp_col + ni * 8 + t * 2;
        const Wide col_term0 =
            n < N ? Wide(zero_point_A) * Wide(col_sum[n]) : 0;
        const Wide col_term1 =
            n + 1 < N ? Wide(zero_point_A) * Wide(col_sum[n + 1]) : 0;
        char2 v;
        v.x = quantize(
            float(Signed(Wide(acc[mi][ni][h * 2]) + row_term - col_term0)),
            scale_A, scale_B, scale_C, zero_point_C);
        v.y = quantize(
            float(Signed(Wide(acc[mi][ni][h * 2 + 1]) + row_term - col_term1)),
            scale_A, scale_B, scale_C, zero_point_C);
        *reinterpret_cast<char2 *>(out + row * OUT_STRIDE + warp_col + ni * 8 +
                                   t * 2) = v;
      }
    }
  __syncthreads();
  for (int i = threadIdx.x; i < MBM * (MBN / 16); i += MTHREADS) {
    const int row = i / (MBN / 16), col = (i % (MBN / 16)) * 16;
    const int m = row0 + row, n = col0 + col;
    if (m >= M) {
      continue;
    }
    int8_t *dst = C + size_t(m) * N + n;
    const int8_t *src = out + row * OUT_STRIDE + col;
    if (n + 16 <= N && (reinterpret_cast<uintptr_t>(dst) & 15) == 0) {
      *reinterpret_cast<int4 *>(dst) = *reinterpret_cast<const int4 *>(src);
    } else {
      for (int e = 0; e < 16 && n + e < N; ++e) {
        dst[e] = src[e];
      }
    }
  }
#endif
}

// Grows a reusable device scratch buffer.
static void *scratch(size_t bytes) {
  static void *buffer = nullptr;
  static size_t capacity = 0;
  if (bytes > capacity) {
    cudaFree(buffer);
    cudaMalloc(&buffer, bytes);
    capacity = bytes;
  }
  return buffer;
}

static bool use_mma() {
  static int major = -1;
  if (major < 0) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (major >= 8) {
      cudaFuncSetAttribute(mm_int8_mma<true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           MMA_SMEM);
      cudaFuncSetAttribute(mm_int8_mma<false>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           MMA_SMEM);
    }
  }
  return major >= 8;
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

  constexpr int CHUNK = 256;
  const int chunks = (K + CHUNK - 1) / CHUNK;

  if (use_mma()) {
    // Scratch: Bt[Np][Kp], then row_sum[M] and col_sum[Np].
    const int Kp = (K + MBK - 1) / MBK * MBK;
    const int Np = (N + MBN - 1) / MBN * MBN;
    const size_t bt_bytes = size_t(Np) * Kp;
    int8_t *Bt = static_cast<int8_t *>(
        scratch(bt_bytes + (size_t(M) + Np) * sizeof(int)));
    int *row_sum = reinterpret_cast<int *>(Bt + bt_bytes),
        *col_sum = row_sum + M;
    cudaMemsetAsync(row_sum, 0, (size_t(M) + Np) * sizeof(int));

    if (K > 0) {
      transpose_b<<<dim3(Np / TT, Kp / TT), 256>>>(B, Bt, col_sum, K, N, Kp);
    }

    if (K > 0 && zero_point_B != 0) {
      row_sums<<<dim3((M + 7) / 8, chunks), 256>>>(A, row_sum, M, K, CHUNK);
    }

    const unsigned tiles = unsigned((M + MBM - 1) / MBM) * (Np / MBN);
    auto kernel = K <= 33025 ? mm_int8_mma<true> : mm_int8_mma<false>;
    kernel<<<tiles, MTHREADS, MMA_SMEM>>>(
        A, Bt, C, row_sum, col_sum, M, N, K, Kp, scale_A, scale_B, scale_C,
        zero_point_A, zero_point_B, zero_point_C);
    return;
  }

  // sm_75: WMMA path. Scratch holds row_sum[M] and col_sum[N].
  int *row_sum = static_cast<int *>(scratch((size_t(M) + N) * sizeof(int)));
  int *col_sum = row_sum + M;
  cudaMemsetAsync(row_sum, 0, (size_t(M) + N) * sizeof(int));

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
