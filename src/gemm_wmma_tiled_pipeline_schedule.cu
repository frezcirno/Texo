// Standalone successor to gemm_wmma_tiled_pipeline_large.cu.
// Controlled mainloop experiments: distributed copies, operand scheduling,
// and address generation. Tile overrides retain the previous contract.
#ifndef GEMM_SCHEDULE
#define GEMM_SCHEDULE 3
#endif
#ifndef GEMM_INTERLEAVE_LOADS
#define GEMM_INTERLEAVE_LOADS 1
#endif
#ifndef GEMM_ADDRESS_MODE
#define GEMM_ADDRESS_MODE (GEMM_SCHEDULE >= 3 ? 2 : 0)
#endif
#ifndef GEMM_AUTO_TILE
#if defined(GEMM_BM) || defined(GEMM_BN) || defined(GEMM_BK) ||                \
    defined(GEMM_WM) || defined(GEMM_WN) || defined(GEMM_STAGES)
#define GEMM_AUTO_TILE 0
#else
#define GEMM_AUTO_TILE 1
#endif
#endif

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

// The dispatcher guarantees complete tiles and 16-byte-aligned A/B pointers.
// Each thread computes its copy addresses once, then advances them by a K
// chunk. The fixed arrays and loops are unrolled into registers for the default
// tiles.
template <int ROWS, int COLS, int STRIDE, int THREADS> struct AlignedInputTile {
  static constexpr int PACK = 8;
  static constexpr int PACKS = ROWS * (COLS / PACK);
  static constexpr int COPIES = (PACKS + THREADS - 1) / THREADS;
  const half *source[COPIES];
  int destination[COPIES];

  __device__ __forceinline__ void init(const half *input, int cols, size_t row0,
                                       size_t col0) {
#pragma unroll
    for (int i = 0; i < COPIES; ++i) {
      const int pack = int(threadIdx.x) + i * THREADS;
      // This is a thread-work bound, not a matrix-tail check. It compiles out
      // when PACKS is divisible by THREADS, including the default
      // configuration.
      if (PACKS % THREADS == 0 || pack < PACKS) {
        const int row = pack / (COLS / PACK);
        const int col = (pack % (COLS / PACK)) * PACK;
        source[i] = input + (row0 + row) * size_t(cols) + col0 + col;
        destination[i] = row * STRIDE + col;
      }
    }
  }

  __device__ __forceinline__ void copy(half (&tile)[ROWS][STRIDE],
                                       size_t advance) {
#pragma unroll
    for (int i = 0; i < COPIES; ++i) {
      const int pack = int(threadIdx.x) + i * THREADS;
      if (PACKS % THREADS == 0 || pack < PACKS) {
        // Advance only when another valid K chunk is about to be copied.
        source[i] += advance;
        half *dst = &tile[0][0] + destination[i];
#if __CUDA_ARCH__ >= 800
        __pipeline_memcpy_async(dst, source[i], 16);
#else
#pragma unroll
        for (int e = 0; e < PACK; ++e)
          dst[e] = source[i][e];
#endif
      }
    }
  }
};

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

template <bool UNIT_ALPHA_ZERO_BETA, bool FULL_ALIGNED, int BM, int BN, int BK,
          int WM, int WN, int SKEW>
__global__ void gemm_wmma_tiled_pipeline_aligned(const half *__restrict__ A,
                                                 const half *__restrict__ B,
                                                 half *__restrict__ C, int M,
                                                 int N, int K, float alpha,
                                                 float beta) {
  static_assert(BM > 0 && BN > 0 && BK > 0 && WM > 0 && WN > 0,
                "Tile dimensions must be positive");
  static_assert(BM % WM == 0 && BN % WN == 0, "Whole warp tiles per block");
  static_assert(WM % 16 == 0 && WN % 16 == 0 && BK % 16 == 0,
                "WMMA dimensions must be multiples of 16");
  // Fragment origins are multiples of 16 rows/columns and remain 32-byte
  // aligned; WMMA's half stride and each async copy need 16-byte alignment.
  static_assert(SKEW >= 0 && SKEW % 8 == 0, "Preserve 16-byte row alignment");
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

    AlignedInputTile<BM, BK, BK + SKEW, BLOCK_SIZE> aligned_a;
    AlignedInputTile<BK, BN, BN + SKEW, BLOCK_SIZE> aligned_b;
    if (FULL_ALIGNED) {
      aligned_a.init(A, K, row0, 0);
      aligned_b.init(B, N, 0, col0);
    }
    const size_t b_advance = size_t(BK) * N;

    if (tiles_k > 0) {
      // Prologue: the first tile must be ready before starting WMMA.
      if (FULL_ALIGNED) {
        aligned_a.copy(A_tile[0], 0);
        aligned_b.copy(B_tile[0], 0);
      } else {
        stage_input_tile<BM, BK, BK + SKEW, BLOCK_SIZE>(A_tile[0], A, M, K,
                                                        row0, 0);
        stage_input_tile<BK, BN, BN + SKEW, BLOCK_SIZE>(B_tile[0], B, K, N, 0,
                                                        col0);
      }
      __syncwarp(); // Reconverge after per-thread alignment/bounds branches.
      __pipeline_commit();
      __pipeline_wait_prior(0);
      __syncthreads();
    }

    for (size_t t = 0; t < tiles_k; ++t) {
      const int cur = int(t & 1), next = cur ^ 1;
      if (t + 1 < tiles_k) {
        if (FULL_ALIGNED) {
          aligned_a.copy(A_tile[next], BK);
          aligned_b.copy(B_tile[next], b_advance);
        } else {
          const size_t tb = (t + 1) * BK;
          stage_input_tile<BM, BK, BK + SKEW, BLOCK_SIZE>(A_tile[next], A, M, K,
                                                          row0, tb);
          stage_input_tile<BK, BN, BN + SKEW, BLOCK_SIZE>(B_tile[next], B, K, N,
                                                          tb, col0);
        }
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
          if (FULL_ALIGNED || (m < size_t(M) && n < size_t(N))) {
            const size_t index = m * N + n;
            if (UNIT_ALPHA_ZERO_BETA) {
              C[index] = __float2half_rn(output[warp][row][col]);
            } else {
              C[index] = alpha * output[warp][row][col] +
                         (beta == 0.0f ? 0.0f : beta * __half2float(C[index]));
            }
          }
        }
        __syncwarp();
      }
    }
  }
}

// Permute whole eight-half packs; values inside a 16-byte copy stay contiguous.
// For power-of-two COLS >= 16 this XOR is a bijection: each source row bit is
// above its destination bit. Aligned 8x8 matrices span all eight 16-byte bank
// groups; the copy and matrix-load address maps use the same permutation.
template <int COLS>
__device__ __forceinline__ int swizzled_offset(int row, int col) {
  static_assert(COLS >= 16 && (COLS & (COLS - 1)) == 0,
                "Swizzled input width must be a power of two >= 16");
  return (row * COLS + col) ^ ((row & 7) * 8);
}

template <int ROWS, int COLS, int THREADS, int ADDRESS_MODE>
struct SwizzledInputTile {
  static constexpr int PACKS = ROWS * COLS / 8;
  static constexpr int COPIES = (PACKS + THREADS - 1) / THREADS;
  static constexpr bool COMPACT =
      ADDRESS_MODE >= 2 && THREADS % (COLS / 8) == 0;
  const half *source[COPIES];
  size_t row_step;
  int destination[COPIES];
  __device__ __forceinline__ void init(const half *input, int stride,
                                       size_t row0, size_t col0) {
    if (COMPACT)
      row_step = size_t(stride) * (THREADS / (COLS / 8));
#pragma unroll
    for (int i = 0; i < COPIES; ++i) {
      const int pack = int(threadIdx.x) + i * THREADS;
      if (PACKS % THREADS == 0 || pack < PACKS) {
        const int row = pack / (COLS / 8), col = (pack % (COLS / 8)) * 8;
        if (!COMPACT || i == 0)
          source[i] = input + (row0 + row) * size_t(stride) + col0 + col;
        destination[i] = swizzled_offset<COLS>(row, col);
      }
    }
  }
  __device__ __forceinline__ void copy(half *tile, size_t advance) {
#pragma unroll
    for (int i = 0; i < COPIES; ++i) {
      const int pack = int(threadIdx.x) + i * THREADS;
      if (PACKS % THREADS == 0 || pack < PACKS) {
        if (!COMPACT || i == 0)
          source[i] += advance;
        const half *src = COMPACT ? source[0] + i * row_step : source[i];
#if __CUDA_ARCH__ >= 800
        __pipeline_memcpy_async(tile + destination[i], src, 16);
#else
#pragma unroll
        for (int e = 0; e < 8; ++e)
          tile[destination[i] + e] = src[e];
#endif
      }
    }
  }
  // Each copy belongs to exactly one group. The unrolled caller supplies a
  // constant group, so inactive copies and their pointer updates disappear.
  template <int GROUPS>
  __device__ __forceinline__ void copy_group(half *tile, size_t advance,
                                            int group) {
#pragma unroll
    for (int i = 0; i < COPIES; ++i) {
      const int pack = int(threadIdx.x) + i * THREADS;
      if (i * GROUPS / COPIES == group &&
          (PACKS % THREADS == 0 || pack < PACKS)) {
        if (!COMPACT || i == 0)
          source[i] += advance;
        const half *src = COMPACT ? source[0] + i * row_step : source[i];
#if __CUDA_ARCH__ >= 800
        __pipeline_memcpy_async(tile + destination[i], src, 16);
#else
#pragma unroll
        for (int e = 0; e < 8; ++e)
          tile[destination[i] + e] = src[e];
#endif
      }
    }
  }
};

// Use the documented PTX fragment mapping, independently of WMMA's opaque
// fragment representation. Each register packs two FP16 values.
__device__ __forceinline__ void load_swizzled_a(uint32_t (&r)[4],
                                                const half *p) {
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(address)
               : "memory");
}
__device__ __forceinline__ void load_swizzled_b(uint32_t (&r)[4],
                                                const half *p) {
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(address)
      : "memory");
}
// Use the documented PTX fragment mapping, independently of WMMA's opaque
// fragment representation. Each register packs two FP16 values.
__device__ __forceinline__ void load_swizzled_a(uint32_t (&r)[4],
                                                uint32_t address) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(address)
               : "memory");
}
__device__ __forceinline__ void load_swizzled_b(uint32_t (&r)[4],
                                                uint32_t address) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(address)
      : "memory");
}
__device__ __forceinline__ void
mma_swizzled(float (&d)[4], const uint32_t (&a)[4], const uint32_t *b) {
#if __CUDA_ARCH__ >= 800
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                 "r"(b[1]));
#else
  // Turing supports K=8 for this instruction. Two operations cover K=16.
#pragma unroll
  for (int k = 0; k < 2; ++k)
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[2 * k]), "r"(a[2 * k + 1]), "r"(b[k]));
#endif
}

// The XOR affects only low lane/column bits. Whole 16-row fragments and
// whole stages advance above those bits; do shared address arithmetic in u32.
template <int BM, int BN, int BK, int ADDRESS_MODE>
struct ScheduledOperands {
  half (*at)[BM * BK];
  half (*bt)[BK * BN];
  int warp_m, warp_n, lane;
  uint32_t base_a, base_b, offset_a, offset_b;
  __device__ __forceinline__ void init(half (*a)[BM * BK], half (*b)[BK * BN],
                                       int m, int n, int l) {
    at = a; bt = b; warp_m = m; warp_n = n; lane = l;
    if (ADDRESS_MODE > 0) {
      base_a = static_cast<uint32_t>(__cvta_generic_to_shared(a));
      base_b = static_cast<uint32_t>(__cvta_generic_to_shared(b));
      offset_a = 2 * swizzled_offset<BK>(m + l % 16, (l / 16) * 8);
      offset_b = 2 * swizzled_offset<BN>(l % 16, n + (l / 16) * 8);
    }
  }
  __device__ __forceinline__ void load_a(uint32_t (&r)[4], int stage,
                                         int k, int i) const {
    if (ADDRESS_MODE > 0)
      load_swizzled_a(r, base_a + stage * (2 * BM * BK) +
          (offset_a ^ uint32_t(k * 32)) + i * (32 * BK));
    else
      load_swizzled_a(r, at[stage] + swizzled_offset<BK>(
          warp_m + i * 16 + lane % 16, k * 16 + (lane / 16) * 8));
  }
  __device__ __forceinline__ void load_b(uint32_t (&r)[4], int stage,
                                         int k, int j) const {
    if (ADDRESS_MODE > 0)
      load_swizzled_b(r, base_b + stage * (2 * BK * BN) +
          (offset_b ^ uint32_t(j * 32)) + k * (32 * BN));
    else
      load_swizzled_b(r, bt[stage] + swizzled_offset<BN>(
          k * 16 + lane % 16, warp_n + j * 16 + (lane / 16) * 8));
  }
};

// Give BK=64 more copy-address/operand registers than the BK=32 path.
// Four is a compiler register target, not a runtime residency promise.
#ifndef GEMM_MULTISTAGE_MIN_BLOCKS
#define GEMM_MULTISTAGE_MIN_BLOCKS 0
#endif
#ifndef GEMM_GROUP_M
#define GEMM_GROUP_M 1
#endif
#ifndef GEMM_REGISTER_PIPELINE
#define GEMM_REGISTER_PIPELINE 1
#endif

template <bool UNIT_ALPHA_ZERO_BETA, int BM, int BN, int BK, int WM, int WN,
          int STAGES, int SCHEDULE>
__global__ __launch_bounds__(
    (BM / WM) * (BN / WN) * 32,
    GEMM_MULTISTAGE_MIN_BLOCKS > 0
        ? GEMM_MULTISTAGE_MIN_BLOCKS
        : ((BM / WM) * (BN / WN) == 4
               ? (BM == 64 && BN == 64 ? (BK == 64 ? 4 : 5)
                                       : ((BM == 96 || BM == 128) && BN == 128 ? 2 : 3))
               : 2)) void gemm_wmma_tiled_pipeline_schedule(
    const half *__restrict__ A, const half *__restrict__ B,
    half *__restrict__ C, int M, int N, int K, float alpha, float beta) {
  static_assert(SCHEDULE >= 0 && SCHEDULE <= 3, "Schedule mode 0..3");
  constexpr int ADDRESS_MODE = SCHEDULE > 0 ? GEMM_ADDRESS_MODE : 0;
  static_assert(ADDRESS_MODE >= 0 && ADDRESS_MODE <= 2, "Address mode 0..2");
  static_assert(BM > 0 && BN > 0 && BK > 0 && WM > 0 && WN > 0,
                "Tile dimensions must be positive");
  static_assert(BM % WM == 0 && BN % WN == 0 && WM % 16 == 0 && WN % 16 == 0,
                "Whole 16x16 warp fragments and whole warp tiles required");
  constexpr int WARPS = (BM / WM) * (BN / WN), THREADS = WARPS * 32;
  static_assert(WARPS > 0 && WARPS <= 32, "Valid CUDA block size");
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int warp_m = (warp / (BN / WN)) * WM;
  const int warp_n = (warp % (BN / WN)) * WN;
  static_assert(STAGES >= 2 && STAGES <= 4, "Two to four input stages");
  constexpr size_t SHARED_BYTES = STAGES * (BM + BN) * BK * sizeof(half);
  constexpr bool DYNAMIC = SHARED_BYTES > 48 * 1024;
  static_assert(SHARED_BYTES <= 163 * 1024,
                "Input stages must fit the sm_80 opt-in shared memory limit");
  // Preserve the original static layout for small tiles. Unused static arrays
  // are removed in dynamic specializations; A and B each remain 32B aligned.
  __shared__ __align__(32) half at_static[DYNAMIC ? 1 : STAGES]
                                        [DYNAMIC ? 1 : BM * BK];
  __shared__ __align__(32) half bt_static[DYNAMIC ? 1 : STAGES]
                                        [DYNAMIC ? 1 : BK * BN];
  extern __shared__ __align__(32) half input_storage[];
  half (*at)[BM * BK] = reinterpret_cast<half (*)[BM * BK]>(
      DYNAMIC ? input_storage : &at_static[0][0]);
  half (*bt)[BK * BN] = reinterpret_cast<half (*)[BK * BN]>(
      DYNAMIC ? input_storage + STAGES * BM * BK : &bt_static[0][0]);
  const size_t tiles_n = size_t(N) / BN, tiles = (size_t(M) / BM) * tiles_n;
  for (size_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    static_assert(GEMM_GROUP_M > 0, "Positive output row group");
    const size_t group = tile / (GEMM_GROUP_M * tiles_n);
    const size_t first_m = group * GEMM_GROUP_M;
    const size_t remaining_m = size_t(M) / BM - first_m;
    const size_t group_m =
        remaining_m < GEMM_GROUP_M ? remaining_m : GEMM_GROUP_M;
    const size_t local = tile % (GEMM_GROUP_M * tiles_n);
    const size_t row0 = (first_m + local % group_m) * BM;
    const size_t col0 = (local / group_m) * BN;
    float acc[WM / 16][WN / 8][4];
#pragma unroll
    for (int i = 0; i < WM / 16; ++i)
#pragma unroll
      for (int j = 0; j < WN / 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          acc[i][j][e] = 0;
    SwizzledInputTile<BM, BK, THREADS, ADDRESS_MODE> a_copy;
    SwizzledInputTile<BK, BN, THREADS, ADDRESS_MODE> b_copy;
    a_copy.init(A, K, row0, 0);
    b_copy.init(B, N, 0, col0);
    const int chunks = K / BK;
    // Commit even empty drain groups. Keeping a fixed group distance lets
    // wait_prior(STAGES - 2) make the current tile ready for short K as well.
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (s < chunks) {
        a_copy.copy(at[s], s == 0 ? 0 : BK);
        b_copy.copy(bt[s], s == 0 ? 0 : size_t(BK) * N);
      }
      __syncwarp();
      __pipeline_commit();
    }
    __pipeline_wait_prior(STAGES - 2);
    __syncthreads();
    // Preserve the legacy loop's code generation as a compile-time branch,
    // including its BK=64 operand lifetimes. Mode 0 remains a direct control.
    if (SCHEDULE == 0) {
      int cur = 0, next = STAGES - 1;
      for (int t = 0; t < chunks; ++t) {
        // This stage is free: all warps finished reading it before the preceding
        // block barrier. Input pointers only advance for an in-bounds K chunk.
        if (t + STAGES - 1 < chunks) {
          a_copy.copy(at[next], BK);
          b_copy.copy(bt[next], size_t(BK) * N);
        }
        __syncwarp();
        __pipeline_commit();
        uint32_t a[2][WM / 16][4], b[2][WN / 16][4];
#if GEMM_REGISTER_PIPELINE
#pragma unroll
        for (int i = 0; i < WM / 16; ++i)
          load_swizzled_a(
              a[0][i], at[cur] + swizzled_offset<BK>(warp_m + i * 16 + lane % 16,
                                                     (lane / 16) * 8));
#pragma unroll
        for (int j = 0; j < WN / 16; ++j)
          load_swizzled_b(b[0][j], bt[cur] + swizzled_offset<BN>(
                                                 lane % 16, warp_n + j * 16 +
                                                                (lane / 16) * 8));
#endif
#pragma unroll
        for (int k = 0; k < BK / 16; ++k) {
#if GEMM_REGISTER_PIPELINE
          const int slot = k & 1;
          const int load_k = k + 1, load_slot = slot ^ 1;
#else
          const int slot = 0, load_k = k, load_slot = 0;
#endif
          if (load_k < BK / 16) {
#pragma unroll
            for (int i = 0; i < WM / 16; ++i)
              load_swizzled_a(
                  a[load_slot][i],
                  at[cur] + swizzled_offset<BK>(warp_m + i * 16 + lane % 16,
                                                load_k * 16 + (lane / 16) * 8));
#pragma unroll
            for (int j = 0; j < WN / 16; ++j)
              load_swizzled_b(b[load_slot][j],
                              bt[cur] + swizzled_offset<BN>(
                                            load_k * 16 + lane % 16,
                                            warp_n + j * 16 + (lane / 16) * 8));
          }
#pragma unroll
          for (int i = 0; i < WM / 16; ++i)
#pragma unroll
            for (int j = 0; j < WN / 8; ++j)
              mma_swizzled(acc[i][j], a[slot][i], &b[slot][j / 2][(j % 2) * 2]);
        }
        __pipeline_wait_prior(STAGES - 2);
        // All threads' next operands are ready; all warps have released cur.
        __syncthreads();
        cur = cur + 1 == STAGES ? 0 : cur + 1;
        next = next + 1 == STAGES ? 0 : next + 1;
      }
    } else {
      int cur = 0, next = STAGES - 1;
      constexpr bool CROSS_STAGE =
          SCHEDULE >= 2 && GEMM_REGISTER_PIPELINE && BK >= 32;
      constexpr bool INTERLEAVE = CROSS_STAGE && GEMM_INTERLEAVE_LOADS;
      ScheduledOperands<BM, BN, BK, ADDRESS_MODE> operands;
      operands.init(at, bt, warp_m, warp_n, lane);
      uint32_t a[2][WM / 16][4], b[2][WN / 16][4];
      if (CROSS_STAGE) {
#pragma unroll
        for (int i = 0; i < WM / 16; ++i)
          operands.load_a(a[0][i], 0, 0, i);
#pragma unroll
        for (int j = 0; j < WN / 16; ++j)
          operands.load_b(b[0][j], 0, 0, j);
      }
      for (int t = 0; t < chunks; ++t) {
        // This stage is free: all warps finished reading it before the preceding
        // block barrier. Input pointers only advance for an in-bounds K chunk.
        if ((SCHEDULE == 0 || BK < 32) && t + STAGES - 1 < chunks) {
          a_copy.copy(at[next], BK);
          b_copy.copy(bt[next], size_t(BK) * N);
        }
        if (SCHEDULE == 0 || BK < 32) {
          __syncwarp();
          __pipeline_commit();
        }
#if GEMM_REGISTER_PIPELINE
        if (!CROSS_STAGE) {
#pragma unroll
          for (int i = 0; i < WM / 16; ++i)
            operands.load_a(a[0][i], cur, 0, i);
#pragma unroll
          for (int j = 0; j < WN / 16; ++j)
            operands.load_b(b[0][j], cur, 0, j);
        }
#endif
#pragma unroll
        for (int k = 0; k < BK / 16; ++k) {
#if GEMM_REGISTER_PIPELINE
          const int slot = k & 1;
          const int load_k = k + 1, load_slot = slot ^ 1;
#else
          const int slot = 0, load_k = k, load_slot = 0;
#endif
          const bool prefetch = load_k < BK / 16 ||
              (CROSS_STAGE && t + 1 < chunks);
          const int operand_k = load_k < BK / 16 ? load_k : 0;
          if (prefetch && !INTERLEAVE) {
#pragma unroll
            for (int i = 0; i < WM / 16; ++i)
              operands.load_a(a[load_slot][i], cur, operand_k, i);
#pragma unroll
            for (int j = 0; j < WN / 16; ++j)
              operands.load_b(b[load_slot][j], cur, operand_k, j);
          }
#pragma unroll
          for (int i = 0; i < WM / 16; ++i) {
#pragma unroll
            for (int j = 0; j < WN / 8; ++j) {
              mma_swizzled(acc[i][j], a[slot][i], &b[slot][j / 2][(j % 2) * 2]);
              // Keep each B fragment's reuse across all M fragments. Only its
              // other register slot is filled, once per K group.
              if (INTERLEAVE && prefetch && i == 0 && j % 2 == 1)
                operands.load_b(b[load_slot][j / 2], cur, operand_k, j / 2);
            }
            if (INTERLEAVE && prefetch)
              operands.load_a(a[load_slot][i], cur, operand_k, i);
            // Finish issuing this stage before the final K group. Its commit
            // still represents a whole stage, including an empty drain group.
            if (SCHEDULE > 0 && BK >= 32 && k < BK / 16 - 1 &&
                t + STAGES - 1 < chunks) {
              constexpr int GROUPS = (BK / 16 - 1) * (WM / 16);
              const int group = k * (WM / 16) + i;
              a_copy.template copy_group<GROUPS>(at[next], BK, group);
              b_copy.template copy_group<GROUPS>(bt[next], size_t(BK) * N, group);
            }
          }
          if (SCHEDULE > 0 && BK >= 32 && k == BK / 16 - 2) {
            __syncwarp();
            __pipeline_commit();
            if (CROSS_STAGE) {
              // The last K group's operands are now in registers. All warps
              // can release cur before its MMA, then prefetch the next stage
              // into the alternate register slot during that final group.
              __pipeline_wait_prior(STAGES - 2);
              __syncthreads();
              cur = cur + 1 == STAGES ? 0 : cur + 1;
              next = next + 1 == STAGES ? 0 : next + 1;
            }
          }
        }
        if (!CROSS_STAGE) {
          __pipeline_wait_prior(STAGES - 2);
          // All threads' next operands are ready; all warps have released cur.
          __syncthreads();
          cur = cur + 1 == STAGES ? 0 : cur + 1;
          next = next + 1 == STAGES ? 0 : next + 1;
        }
      }
    }
    __pipeline_wait_prior(0);
    // Pack each lane's adjacent output values, then exchange the two N=8
    // fragments so a warp writes four complete 16-element rows. Direct stores
    // in the MMA register order would scatter each instruction over eight rows.
    const bool aligned_c = (reinterpret_cast<uintptr_t>(C) & 3) == 0;
    // Keep the general epilogue's original loop structure. The specialized
    // large-warp path visits all N fragments within an eight-row stripe;
    // the small-warp path converts each pair with the half2 intrinsic.
    if (UNIT_ALPHA_ZERO_BETA && WM >= 48) {
#pragma unroll
      for (int group = 0; group < (WM / 16) * (WN / 16) * 2; ++group) {
        const int i = group / (2 * (WN / 16));
        const int j = group % (WN / 16);
        const int half_rows = (group / (WN / 16)) % 2;
        uint32_t packed[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float x = acc[i][j * 2 + h][half_rows * 2];
          const float y = acc[i][j * 2 + h][half_rows * 2 + 1];
          packed[h] = uint32_t(__half_as_ushort(__float2half_rn(x))) |
                      (uint32_t(__half_as_ushort(__float2half_rn(y))) << 16);
        }
#pragma unroll
        for (int stripe = 0; stripe < 2; ++stripe) {
          const int source_lane = (stripe * 4 + lane / 8) * 4 + lane % 4;
          const uint32_t left =
              __shfl_sync(0xffffffff, packed[0], source_lane);
          const uint32_t right =
              __shfl_sync(0xffffffff, packed[1], source_lane);
          const uint32_t value = (lane / 4) % 2 ? right : left;
          const size_t row =
              row0 + warp_m + i * 16 + half_rows * 8 + stripe * 4 + lane / 8;
          const size_t col = col0 + warp_n + j * 16 + (lane % 8) * 2;
          const size_t index = row * N + col;
          __half2_raw pair;
          pair.x = static_cast<unsigned short>(value);
          pair.y = static_cast<unsigned short>(value >> 16);
          if (aligned_c) {
            *reinterpret_cast<half2 *>(C + index) = half2(pair);
          } else {
            C[index] = __ushort_as_half(pair.x);
            C[index + 1] = __ushort_as_half(pair.y);
          }
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < WM / 16; ++i)
#pragma unroll
        for (int j = 0; j < WN / 16; ++j)
#pragma unroll
          for (int half_rows = 0; half_rows < 2; ++half_rows) {
            uint32_t packed[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const size_t row =
                  row0 + warp_m + i * 16 + lane / 4 + half_rows * 8;
              const size_t col = col0 + warp_n + j * 16 + h * 8 + (lane % 4) * 2;
              const size_t index = row * N + col;
              // The true specialization contains no scaling or old-C loads.
              // C++14 folds this template-constant branch before code generation.
              const float x = UNIT_ALPHA_ZERO_BETA
                  ? acc[i][j * 2 + h][half_rows * 2]
                  : alpha * acc[i][j * 2 + h][half_rows * 2] +
                        (beta == 0.0f ? 0.0f : beta * __half2float(C[index]));
              const float y = UNIT_ALPHA_ZERO_BETA
                  ? acc[i][j * 2 + h][half_rows * 2 + 1]
                  : alpha * acc[i][j * 2 + h][half_rows * 2 + 1] +
                        (beta == 0.0f ? 0.0f : beta * __half2float(C[index + 1]));
              if (UNIT_ALPHA_ZERO_BETA) {
                const __half2_raw pair = __floats2half2_rn(x, y);
                packed[h] = uint32_t(pair.x) | (uint32_t(pair.y) << 16);
              } else {
                packed[h] = uint32_t(__half_as_ushort(__float2half_rn(x))) |
                            (uint32_t(__half_as_ushort(__float2half_rn(y))) << 16);
              }
            }
#pragma unroll
            for (int stripe = 0; stripe < 2; ++stripe) {
              const int source_lane = (stripe * 4 + lane / 8) * 4 + lane % 4;
              const uint32_t left =
                  __shfl_sync(0xffffffff, packed[0], source_lane);
              const uint32_t right =
                  __shfl_sync(0xffffffff, packed[1], source_lane);
              const uint32_t value = (lane / 4) % 2 ? right : left;
              const size_t row =
                  row0 + warp_m + i * 16 + half_rows * 8 + stripe * 4 + lane / 8;
              const size_t col = col0 + warp_n + j * 16 + (lane % 8) * 2;
              const size_t index = row * N + col;
              __half2_raw pair;
              pair.x = static_cast<unsigned short>(value);
              pair.y = static_cast<unsigned short>(value >> 16);
              if (aligned_c) {
                *reinterpret_cast<half2 *>(C + index) = half2(pair);
              } else {
                C[index] = __ushort_as_half(pair.x);
                C[index + 1] = __ushort_as_half(pair.y);
              }
            }
          }
    }
  }
}

#ifndef GEMM_STAGES
#define GEMM_STAGES 3
#endif
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
#ifndef GEMM_DYNAMIC_SHARED_LIMIT
// Zero uses the active device's limit; positive values impose a lower cap.
#define GEMM_DYNAMIC_SHARED_LIMIT 0
#endif

template <bool UNIT_ALPHA_ZERO_BETA, int BM, int BN, int BK, int WM, int WN,
          int STAGES, int SCHEDULE = GEMM_SCHEDULE>
static bool launch_schedule(const half *A, const half *B, half *C, int M, int N,
                            int K, float alpha, float beta) {
  constexpr int THREADS = (BM / WM) * (BN / WN) * 32;
  constexpr size_t SHARED_BYTES = STAGES * (BM + BN) * BK * sizeof(half);
  constexpr size_t DYNAMIC_BYTES = SHARED_BYTES > 48 * 1024 ? SHARED_BYTES : 0;
  static_assert(GEMM_DYNAMIC_SHARED_LIMIT >= 0, "Nonnegative shared memory cap");
  if (DYNAMIC_BYTES) {
    int device = 0, limit = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               device) != cudaSuccess)
      return false;
    // Unsupported devices retain the fixed generic fallback. Configure on
    // each launch so device/context changes require no host-side cache.
    if (int(DYNAMIC_BYTES) > limit ||
        (GEMM_DYNAMIC_SHARED_LIMIT > 0 &&
         DYNAMIC_BYTES > size_t(GEMM_DYNAMIC_SHARED_LIMIT)))
      return false;
    if (cudaFuncSetAttribute(
            gemm_wmma_tiled_pipeline_schedule<UNIT_ALPHA_ZERO_BETA, BM, BN, BK,
                                           WM, WN, STAGES, SCHEDULE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(DYNAMIC_BYTES)) != cudaSuccess)
      return false;
  }
  const size_t tiles = (size_t(M) / BM) * (size_t(N) / BN);
  const unsigned blocks = unsigned(tiles < 2147483647u ? tiles : 2147483647u);
  gemm_wmma_tiled_pipeline_schedule<UNIT_ALPHA_ZERO_BETA, BM, BN, BK, WM, WN,
                                   STAGES, SCHEDULE>
      <<<blocks, THREADS, DYNAMIC_BYTES>>>(A, B, C, M, N, K, alpha, beta);
  return true;
}

// Row-major FP16 A[M,K], B[K,N], C[M,N]; FP32 accumulation and alpha/beta.
// Nonnegative dimensions; M/N=0 is a no-op, K=0 scales C by beta.
// Fast tiles require complete dimensions and 16-byte-aligned A/B. C needs
// only half alignment; a misaligned half2 output uses scalar stores instead.
// Default tile selection uses 64x64, 64x128, 96x128, or 128x128 blocks and
// three/four input stages. Small output grids use BK=64 when K is a multiple of 64
// and has at least three chunks; other configurations retain BK=32.
// Explicit GEMM_BM/BN/BK/WM/WN/STAGES disables automatic
// selection. Unsupported layouts/tails use the fixed 64x64 generic WMMA
// pipeline. sm_75 uses synchronous copies and K=8 MMA; sm_80+ uses async copies
// and K=16.
// Larger explicit tiles use dynamic shared storage above 48 KiB, with opt-in
// on the active device. An unsupported size falls back to the generic kernel.
// The scheduled 128x128 path is selected only in measured A800 grid bands;
// other shapes and general alpha/beta retain the preceding dispatcher.
template <bool UNIT_ALPHA_ZERO_BETA>
static void solve_schedule(const half *A, const half *B, half *C, int M, int N,
                      int K, float alpha, float beta) {
  if (M <= 0 || N <= 0)
    return;
#if (GEMM_BK & (GEMM_BK - 1)) == 0 && (GEMM_BN & (GEMM_BN - 1)) == 0
  const bool full_aligned =
      K > 0 && M % GEMM_BM == 0 && N % GEMM_BN == 0 && K % GEMM_BK == 0 &&
      ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) &
       15) == 0;
  if (full_aligned) {
#if GEMM_AUTO_TILE
    // A800 has 108 SMs. The 128x128 tile works well for a small single wave
    // or a sufficiently populated medium grid; it lost at 1536 cubed and
    // at 6144/8192 cubed. Keep the older choices outside these measured bands.
    const size_t scheduled_tiles = (size_t(M) / 128) * (size_t(N) / 128);
    if (UNIT_ALPHA_ZERO_BETA && GEMM_SCHEDULE > 0 &&
        M % 128 == 0 && N % 128 == 0 && K % 32 == 0 &&
        M >= 1024 && N >= 1024 && M <= 4096 && N <= 4096 &&
        K >= 128 && K <= 8192 &&
        ((scheduled_tiles >= 64 && scheduled_tiles <= 108) ||
         (scheduled_tiles >= 256 && scheduled_tiles <= 1024))) {
      // Four stages help long K on grids near 256..324 blocks. If opt-in
      // storage is unavailable, the three-stage static kernel still applies.
      if (scheduled_tiles >= 256 && scheduled_tiles <= 324 && K >= 1024 &&
          launch_schedule<UNIT_ALPHA_ZERO_BETA, 128, 128, 32, 64, 64, 4>
              (A, B, C, M, N, K, alpha, beta))
        return;
      if (launch_schedule<UNIT_ALPHA_ZERO_BETA, 128, 128, 32, 64, 64, 3>
              (A, B, C, M, N, K, alpha, beta))
        return;
    }
    // Larger reuse tiles need enough blocks and K work to amortize their
    // register/output costs. Thresholds were measured on A800, not autotuned.
    if (M % 128 == 0 && N % 128 == 0 && K % 32 == 0 && K >= 2048 &&
        (size_t(M) / 128) * (size_t(N) / 128) >= 1024) {
      launch_schedule<UNIT_ALPHA_ZERO_BETA, 128, 128, 32, 64, 64, 3, 0>
          (A, B, C, M, N, K, alpha, beta);
      return;
    }
    // The 96x128 tile uses 237 registers/thread (232 for alpha=1/beta=0)
    // and fits two blocks per SM on A800. Small grids cannot amortize its
    // larger per-warp work. M=192
    // multiples satisfy both the base 64-row contract and complete 96 rows.
    // Retain the 128x128 large-matrix path above; its reuse is greater still.
    if (M % 192 == 0 && N % 128 == 0 && K % 32 == 0 &&
        (size_t(M) / 96) * (size_t(N) / 128) >= 192) {
      launch_schedule<UNIT_ALPHA_ZERO_BETA, 96, 128, 32, 48, 64, 3, 0>
          (A, B, C, M, N, K, alpha, beta);
      return;
    }
    if (M % 64 == 0 && N % 128 == 0 && K % 32 == 0 && K >= 512 &&
        (size_t(M) / 64) * (size_t(N) / 128) >= 256) {
      launch_schedule<UNIT_ALPHA_ZERO_BETA, 64, 128, 32, 32, 64, 3, 0>
          (A, B, C, M, N, K, alpha, beta);
      return;
    }
    // A800: the 48 KiB BK=64 tile permits three resident blocks per SM.
    // Limit the grid to 3 * 108 blocks to avoid an extra wave; BK=32 uses
    // 24 KiB and handles larger grids better. Require a steady-state chunk
    // after the two-stage prologue. These are measured, device-specific
    // thresholds, not a portable autotuner.
    if (M % 64 == 0 && N % 64 == 0 && K % 64 == 0 && K >= 192 &&
        (size_t(M) / 64) * (size_t(N) / 64) <= 324) {
      launch_schedule<UNIT_ALPHA_ZERO_BETA, 64, 64, 64, 32, 32, 3, 0>
          (A, B, C, M, N, K, alpha, beta);
      return;
    }
#endif
    if (launch_schedule<UNIT_ALPHA_ZERO_BETA, GEMM_BM, GEMM_BN, GEMM_BK,
                     GEMM_WM, GEMM_WN, GEMM_STAGES,
                     GEMM_AUTO_TILE ? 0 : GEMM_SCHEDULE>(A, B, C, M, N, K, alpha, beta))
      return;
  }
#endif
  // Large aligned tile overrides must not inflate generic shared storage.
  const size_t tiles = ((size_t(M) + 63) / 64) * ((size_t(N) + 63) / 64);
  const unsigned blocks = unsigned(tiles < 2147483647u ? tiles : 2147483647u);
  gemm_wmma_tiled_pipeline_aligned<UNIT_ALPHA_ZERO_BETA, false, 64, 64, 32,
                                  32, 32, 16>
      <<<blocks, 128>>>(A, B, C, M, N, K, alpha, beta);
}

#ifndef GEMM_SPECIALIZE_ALPHA_BETA
#define GEMM_SPECIALIZE_ALPHA_BETA 1
#endif

// Dispatch exact scalar values on the host. Other alpha/beta values keep the
// general epilogue. +0 and -0 beta both mean that the previous C is ignored.
extern "C" void solve(const half *A, const half *B, half *C, int M, int N,
                      int K, float alpha, float beta) {
  if (M <= 0 || N <= 0)
    return;
#if GEMM_SPECIALIZE_ALPHA_BETA
  if (alpha == 1.0f && beta == 0.0f) {
    solve_schedule<true>(A, B, C, M, N, K, alpha, beta);
    return;
  }
#endif
  solve_schedule<false>(A, B, C, M, N, K, alpha, beta);
}
