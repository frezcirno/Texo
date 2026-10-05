// Final scheduled WMMA GEMM implementation with distributed copies and
// interleaved operands.
#ifndef GEMM_AUTO_TILE
#if defined(GEMM_BM) || defined(GEMM_BN) || defined(GEMM_BK) ||                \
    defined(GEMM_WM) || defined(GEMM_WN) || defined(GEMM_STAGES)
#define GEMM_AUTO_TILE 0
#else
#define GEMM_AUTO_TILE 1
#endif
#endif

#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

// Each thread owns disjoint groups of eight half elements. Shared rows and
// group starts are 16-byte aligned; global row strides/pointers need not be.
template <size_t ROWS, size_t COLS, size_t STRIDE, size_t BLOCK_SIZE>
__device__ __forceinline__ void
stage_input_tile(half (&tile)[ROWS][STRIDE], const half *input, size_t rows,
                 size_t cols, size_t row0, size_t col0) {
  constexpr size_t PACK = 8;
  for (int pack = threadIdx.x; pack < ROWS * (COLS / PACK);
       pack += BLOCK_SIZE) {
    const size_t row = pack / (COLS / PACK);
    const size_t col = (pack % (COLS / PACK)) * PACK;
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

template <bool NO_ALPHA_BETA, size_t BM, size_t BN, size_t BK, size_t WM,
          size_t WN, size_t SKEW>
__global__ void gemm_generic(const half *__restrict__ A,
                             const half *__restrict__ B, half *__restrict__ C,
                             size_t M, size_t N, size_t K, float alpha,
                             float beta) {
  static_assert(BM > 0 && BN > 0 && BK > 0 && WM > 0 && WN > 0,
                "Tile dimensions must be positive");
  static_assert(BM % WM == 0 && BN % WN == 0, "Whole warp tiles per block");
  static_assert(WM % 16 == 0 && WN % 16 == 0 && BK % 16 == 0,
                "WMMA dimensions must be multiples of 16");
  // Fragment origins are multiples of 16 rows/columns and remain 32-byte
  // aligned; WMMA's half stride and each async copy need 16-byte alignment.
  static_assert(SKEW % 8 == 0, "Preserve 16-byte row alignment");

  constexpr size_t WARP_NUM = (BM / WM) * (BN / WN);
  static_assert(WARP_NUM > 0 && WARP_NUM <= 32, "Valid CUDA block size");

  constexpr size_t BLOCK_SIZE = WARP_NUM * 32;
  constexpr size_t WARP_FRAG_NUM_M = WM / 16, WARP_FRAG_NUM_N = WN / 16;

  const size_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const size_t warp_in_block_row = (warp / (BN / WN)) * WM;
  const size_t warp_in_block_col = (warp % (BN / WN)) * WN;

  __shared__ __align__(32) half A_tile[2][BM][BK + SKEW];
  __shared__ __align__(32) half B_tile[2][BK][BN + SKEW];
  __shared__ __align__(32) float output[WARP_NUM][16][16];

  const size_t tiles_m = (M + BM - 1) / BM;
  const size_t tiles_n = (N + BN - 1) / BN;
  const size_t tiles_k = (K + BK - 1) / BK;
  // Flatten output tiles so tall matrices do not exceed the grid.y limit.
  for (size_t tile = blockIdx.x; tile < tiles_m * tiles_n; tile += gridDim.x) {
    const size_t row0 = (tile / tiles_n) * BM;
    const size_t col0 = (tile % tiles_n) * BN;
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
      const size_t cur = t & 1, next = cur ^ 1;
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
          const size_t row = e / 16, col = e % 16;
          const size_t m = row0 + warp_in_block_row + i * 16 + row;
          const size_t n = col0 + warp_in_block_col + j * 16 + col;
          if (m < M && n < N) {
            const size_t index = m * N + n;
            if (NO_ALPHA_BETA) {
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
template <size_t COLS>
__device__ __forceinline__ int swizzled_offset(int row, int col) {
  static_assert(COLS >= 16 && (COLS & (COLS - 1)) == 0,
                "Swizzled input width must be a power of two >= 16");
  return (row * COLS + col) ^ ((row & 7) * 8);
}

template <size_t ROWS, size_t COLS, size_t BLOCK_SIZE>
struct SwizzledInputTile {
  static constexpr size_t WORKS = ROWS * COLS / 8;
  static constexpr size_t TH_WORKS = (WORKS + BLOCK_SIZE - 1) / BLOCK_SIZE;
  // COMPACT 模式下只保存源指针和行距
  static constexpr bool COMPACT = BLOCK_SIZE % (COLS / 8) == 0;

  const half *source[TH_WORKS];
  size_t row_step;
  int destination[TH_WORKS];

  __device__ __forceinline__ SwizzledInputTile(const half *input, size_t stride,
                                               size_t row0, size_t col0) {
    if (COMPACT) {
      row_step = stride * (BLOCK_SIZE / (COLS / 8));
    }
#pragma unroll
    for (int i = 0; i < TH_WORKS; ++i) {
      const size_t pack = threadIdx.x + i * BLOCK_SIZE;
      if (WORKS % BLOCK_SIZE == 0 || pack < WORKS) {
        const size_t row = pack / (COLS / 8), col = (pack % (COLS / 8)) * 8;
        if (!COMPACT || i == 0) {
          source[i] = input + (row0 + row) * stride + col0 + col;
        }
        destination[i] = swizzled_offset<COLS>(row, col);
      }
    }
  }

  __device__ __forceinline__ void Copy(half *outile, size_t advance) {
#pragma unroll
    for (int i = 0; i < TH_WORKS; ++i) {
      const size_t work = threadIdx.x + i * BLOCK_SIZE;
      if (WORKS % BLOCK_SIZE == 0 || work < WORKS) {
        if (!COMPACT || i == 0) {
          source[i] += advance;
        }

        // copy 8 elems
        const half *src = COMPACT ? source[0] + i * row_step : source[i];
#if __CUDA_ARCH__ >= 800
        __pipeline_memcpy_async(outile + destination[i], src, 16);
#else
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          outile[destination[i] + e] = src[e];
        }
#endif
      }
    }
  }

  // Each copy belongs to exactly one group. The unrolled caller supplies a
  // constant group, so inactive copies and their pointer updates disappear.
  template <size_t GROUPS>
  __device__ __forceinline__ void CopyGroup(half *tile, size_t advance,
                                            int group) {
#pragma unroll
    for (int i = 0; i < TH_WORKS; ++i) {
      const size_t pack = threadIdx.x + i * BLOCK_SIZE;
      if (i * GROUPS / TH_WORKS == group &&
          (WORKS % BLOCK_SIZE == 0 || pack < WORKS)) {
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
__device__ __forceinline__ void ldmatrix(uint32_t (&r)[4], uint32_t address) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(address)
               : "memory");
}

__device__ __forceinline__ void ldmatrix_trans(uint32_t (&r)[4],
                                               uint32_t address) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(address)
      : "memory");
}

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4],
                                    const uint32_t *b) {
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
template <size_t BM, size_t BN, size_t BK> struct ScheduledOperands {
  uint32_t base_a, base_b, offset_a, offset_b;

  __device__ __forceinline__ void Init(half (*a)[BM * BK], half (*b)[BK * BN],
                                       size_t m, size_t n, size_t l) {
    base_a = static_cast<uint32_t>(__cvta_generic_to_shared(a));
    base_b = static_cast<uint32_t>(__cvta_generic_to_shared(b));
    offset_a = 2 * swizzled_offset<BK>(m + l % 16, (l / 16) * 8);
    offset_b = 2 * swizzled_offset<BN>(l % 16, n + (l / 16) * 8);
  }

  __device__ __forceinline__ void LoadA(uint32_t (&r)[4], int stage, size_t k,
                                        size_t i) const {
    ldmatrix(r, base_a + stage * (2 * BM * BK) + (offset_a ^ uint32_t(k * 32)) +
                    i * (32 * BK));
  }

  __device__ __forceinline__ void LoadB(uint32_t (&r)[4], int stage, size_t k,
                                        size_t j) const {
    ldmatrix_trans(r, base_b + stage * (2 * BK * BN) +
                          (offset_b ^ uint32_t(j * 32)) + k * (32 * BN));
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
template <bool NO_ALPHA_BETA, size_t BM, size_t BN, size_t BK, size_t WM,
          size_t WN, int STAGES>
__global__ __launch_bounds__(
    (BM / WM) * (BN / WN) * 32,
    GEMM_MULTISTAGE_MIN_BLOCKS > 0
        ? GEMM_MULTISTAGE_MIN_BLOCKS
        : ((BM / WM) * (BN / WN) == 4
               ? (BM == 64 && BN == 64
                      ? (BK == 64 ? 4 : 5)
                      : ((BM == 96 || BM == 128) && BN == 128 ? 2 : 3))
               : 2)) void gemm_schedule(const half *__restrict__ A,
                                        const half *__restrict__ B,
                                        half *__restrict__ C, size_t M,
                                        size_t N, size_t K, float alpha,
                                        float beta) {
  static_assert(BM > 0 && BN > 0 && BK > 0 && WM > 0 && WN > 0,
                "Tile dimensions must be positive");
  static_assert(BM % WM == 0 && BN % WN == 0 && WM % 16 == 0 && WN % 16 == 0,
                "Whole 16x16 warp fragments and whole warp tiles required");

  constexpr size_t WARPS = (BM / WM) * (BN / WN), BLOCK_SIZE = WARPS * 32;
  static_assert(WARPS > 0 && WARPS <= 32, "Valid CUDA block size");

  const size_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const size_t warp_m = (warp / (BN / WN)) * WM;
  const size_t warp_n = (warp % (BN / WN)) * WN;

  static_assert(STAGES >= 2 && STAGES <= 4, "Two to four input stages");

  // kernel 为 A/B 输入流水线需要多少共享内存
  // 不超过 48 KiB：用静态共享内存数组 at_static 和 bt_static
  // 超过 48 KiB：用动态共享内存 input_storage；其中先放 A 的所有 stages，再放 B
  // 的所有 stages
  constexpr size_t SHM_SIZE = STAGES * (BM + BN) * BK * sizeof(half);
  constexpr bool DYNAMIC = SHM_SIZE > 48 * 1024;
  static_assert(SHM_SIZE <= 163 * 1024,
                "Input stages must fit the sm_80 opt-in shared memory limit");

  // Preserve the original static layout for small tiles. Unused static arrays
  // are removed in dynamic specializations; A and B each remain 32B aligned.
  __shared__ __align__(32)
      half at_static[DYNAMIC ? 1 : STAGES][DYNAMIC ? 1 : BM * BK];
  __shared__ __align__(32)
      half bt_static[DYNAMIC ? 1 : STAGES][DYNAMIC ? 1 : BK * BN];

  extern __shared__ __align__(32) half input_storage[];
  half(*at)[BM * BK] = reinterpret_cast<half(*)[BM * BK]>(
      DYNAMIC ? input_storage : &at_static[0][0]);
  half(*bt)[BK * BN] = reinterpret_cast<half(*)[BK * BN]>(
      DYNAMIC ? input_storage + STAGES * BM * BK : &bt_static[0][0]);

  const size_t tiles_n = N / BN, tiles = (M / BM) * tiles_n;
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    static_assert(GEMM_GROUP_M > 0, "Positive output row group");
    const size_t group = tile / (GEMM_GROUP_M * tiles_n);
    const size_t first_m = group * GEMM_GROUP_M;
    const size_t remaining_m = M / BM - first_m;
    const size_t group_m =
        remaining_m < GEMM_GROUP_M ? remaining_m : GEMM_GROUP_M;
    const size_t local = tile % (GEMM_GROUP_M * tiles_n);
    const size_t row0 = (first_m + local % group_m) * BM;
    const size_t col0 = (local / group_m) * BN;

    float acc[WM / 16][WN / 8][4];
#pragma unroll
    for (int i = 0; i < WM / 16; ++i) {
#pragma unroll
      for (int j = 0; j < WN / 8; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          acc[i][j][e] = 0;
        }
      }
    }

    SwizzledInputTile<BM, BK, BLOCK_SIZE> a_copy(A, K, row0, 0);
    SwizzledInputTile<BK, BN, BLOCK_SIZE> b_copy(B, N, 0, col0);

    const int chunks = K / BK;

    // Commit even empty drain groups. Keeping a fixed group distance lets
    // wait_prior(STAGES - 2) make the current tile ready for short K as well.
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (s < chunks) {
        a_copy.Copy(at[s], s == 0 ? 0 : BK);
        b_copy.Copy(bt[s], s == 0 ? 0 : BK * N);
      }
      __syncwarp();
      __pipeline_commit();
    }

    __pipeline_wait_prior(STAGES - 2);
    __syncthreads();

    size_t cur = 0, next = STAGES - 1;
    constexpr bool CROSS_STAGE = BK >= 32;

    ScheduledOperands<BM, BN, BK> operands;
    operands.Init(at, bt, warp_m, warp_n, lane);

    uint32_t a[2][WM / 16][4], b[2][WN / 16][4];
    if (CROSS_STAGE) {
#pragma unroll
      for (int i = 0; i < WM / 16; ++i) {
        operands.LoadA(a[0][i], 0, 0, i);
      }
#pragma unroll
      for (int j = 0; j < WN / 16; ++j) {
        operands.LoadB(b[0][j], 0, 0, j);
      }
    }

    for (int t = 0; t < chunks; ++t) {
      // This stage is free: all warps finished reading it before the
      // preceding block barrier. Input pointers only advance for an in-bounds
      // K chunk.
      if (BK < 32 && t + STAGES - 1 < chunks) {
        a_copy.Copy(at[next], BK);
        b_copy.Copy(bt[next], BK * N);
      }
      if (BK < 32) {
        __syncwarp();
        __pipeline_commit();
      }
      if (!CROSS_STAGE) {
#pragma unroll
        for (int i = 0; i < WM / 16; ++i)
          operands.LoadA(a[0][i], cur, 0, i);
#pragma unroll
        for (int j = 0; j < WN / 16; ++j)
          operands.LoadB(b[0][j], cur, 0, j);
      }
#pragma unroll
      for (int k = 0; k < BK / 16; ++k) {
        const int slot = k & 1;
        const int load_k = k + 1, load_slot = slot ^ 1;
        const bool prefetch =
            load_k < BK / 16 || (CROSS_STAGE && t + 1 < chunks);
        const int operand_k = load_k < BK / 16 ? load_k : 0;
        if (prefetch && !CROSS_STAGE) {
#pragma unroll
          for (int i = 0; i < WM / 16; ++i)
            operands.LoadA(a[load_slot][i], cur, operand_k, i);
#pragma unroll
          for (int j = 0; j < WN / 16; ++j)
            operands.LoadB(b[load_slot][j], cur, operand_k, j);
        }
#pragma unroll
        for (int i = 0; i < WM / 16; ++i) {
#pragma unroll
          for (int j = 0; j < WN / 8; ++j) {
            mma(acc[i][j], a[slot][i], &b[slot][j / 2][(j % 2) * 2]);
            // Keep each B fragment's reuse across all M fragments. Only its
            // other register slot is filled, once per K group.
            if (CROSS_STAGE && prefetch && i == 0 && j % 2 == 1)
              operands.LoadB(b[load_slot][j / 2], cur, operand_k, j / 2);
          }
          if (CROSS_STAGE && prefetch)
            operands.LoadA(a[load_slot][i], cur, operand_k, i);
          // Finish issuing this stage before the final K group. Its commit
          // still represents a whole stage, including an empty drain group.
          if (BK >= 32 && k < BK / 16 - 1 && t + STAGES - 1 < chunks) {
            constexpr int GROUPS = (BK / 16 - 1) * (WM / 16);
            const int group = k * (WM / 16) + i;
            a_copy.template CopyGroup<GROUPS>(at[next], BK, group);
            b_copy.template CopyGroup<GROUPS>(bt[next], BK * N, group);
          }
        }
        if (BK >= 32 && k == BK / 16 - 2) {
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

    __pipeline_wait_prior(0);

    // Pack each lane's adjacent output values, then exchange the two N=8
    // fragments so a warp writes four complete 16-element rows. Direct stores
    // in the MMA register order would scatter each instruction over eight rows.
    const bool aligned_c = (reinterpret_cast<uintptr_t>(C) & 3) == 0;
    // Keep the general epilogue's original loop structure. The specialized
    // large-warp path visits all N fragments within an eight-row stripe;
    // the small-warp path converts each pair with the half2 intrinsic.
    if (NO_ALPHA_BETA && WM >= 48) {
#pragma unroll
      for (int group = 0; group < (WM / 16) * (WN / 16) * 2; ++group) {
        const size_t i = group / (2 * (WN / 16));
        const size_t j = group % (WN / 16);
        const size_t half_rows = (group / (WN / 16)) % 2;
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
          const uint32_t left = __shfl_sync(0xffffffff, packed[0], source_lane);
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
              const size_t col =
                  col0 + warp_n + j * 16 + h * 8 + (lane % 4) * 2;
              const size_t index = row * N + col;
              // The true specialization contains no scaling or old-C loads.
              // C++14 folds this template-constant branch before code
              // generation.
              const float x =
                  NO_ALPHA_BETA
                      ? acc[i][j * 2 + h][half_rows * 2]
                      : alpha * acc[i][j * 2 + h][half_rows * 2] +
                            (beta == 0.0f ? 0.0f
                                          : beta * __half2float(C[index]));
              const float y =
                  NO_ALPHA_BETA
                      ? acc[i][j * 2 + h][half_rows * 2 + 1]
                      : alpha * acc[i][j * 2 + h][half_rows * 2 + 1] +
                            (beta == 0.0f ? 0.0f
                                          : beta * __half2float(C[index + 1]));
              if (NO_ALPHA_BETA) {
                const __half2_raw pair = __floats2half2_rn(x, y);
                packed[h] = uint32_t(pair.x) | (uint32_t(pair.y) << 16);
              } else {
                packed[h] =
                    uint32_t(__half_as_ushort(__float2half_rn(x))) |
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
              const size_t row = row0 + warp_m + i * 16 + half_rows * 8 +
                                 stripe * 4 + lane / 8;
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

static void log_schedule_fallback(const char *reason, size_t M, size_t N,
                                  size_t K, size_t BM, size_t BN, size_t BK,
                                  int stages, cudaError_t error = cudaSuccess,
                                  size_t requested = 0, size_t available = 0) {
  fprintf(stderr,
          "gemm_schedule fallback: %s; shape=%zux%zux%zu tile=%zux%zux%zu "
          "stages=%d",
          reason, M, N, K, BM, BN, BK, stages);
  if (error != cudaSuccess)
    fprintf(stderr, "; CUDA error: %s", cudaGetErrorString(error));
  if (requested)
    fprintf(stderr, "; dynamic shared memory=%zu bytes, device limit=%zu bytes",
            requested, available);
  fputc('\n', stderr);
}

template <bool NO_ALPHA_BETA, size_t BM, size_t BN, size_t BK, size_t WM,
          size_t WN, int STAGES>
static bool launch_schedule(const half *__restrict__ A,
                            const half *__restrict__ B, half *__restrict__ C,
                            size_t M, size_t N, size_t K, float alpha,
                            float beta) {
  constexpr int BLOCK_SIZE = (BM / WM) * (BN / WN) * 32;
  constexpr size_t SHM_SIZE = STAGES * (BM + BN) * BK * sizeof(half);
  constexpr size_t DYNAMIC_SHM_SIZE = SHM_SIZE > 48 * 1024 ? SHM_SIZE : 0;
  if (DYNAMIC_SHM_SIZE) {
    int device = 0, limit = 0;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) {
      log_schedule_fallback("could not get current device", M, N, K, BM, BN, BK,
                            STAGES, error);
      return false;
    }
    error = cudaDeviceGetAttribute(
        &limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (error != cudaSuccess) {
      log_schedule_fallback("could not query dynamic shared-memory limit", M, N,
                            K, BM, BN, BK, STAGES, error);
      return false;
    }
    // Unsupported devices retain the fixed generic fallback. Configure on
    // each launch so device/context changes require no host-side cache.
    if (int(DYNAMIC_SHM_SIZE) > limit) {
      log_schedule_fallback(
          "dynamic shared-memory requirement exceeds device limit", M, N, K, BM,
          BN, BK, STAGES, cudaSuccess, DYNAMIC_SHM_SIZE, size_t(limit));
      return false;
    }
    error = cudaFuncSetAttribute(
        gemm_schedule<NO_ALPHA_BETA, BM, BN, BK, WM, WN, STAGES>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, int(DYNAMIC_SHM_SIZE));
    if (error != cudaSuccess) {
      log_schedule_fallback("could not set kernel dynamic shared-memory size",
                            M, N, K, BM, BN, BK, STAGES, error);
      return false;
    }
  }
  const size_t tiles = (M / BM) * (N / BN);
  const unsigned blocks = unsigned(tiles < 2147483647u ? tiles : 2147483647u);
  gemm_schedule<NO_ALPHA_BETA, BM, BN, BK, WM, WN, STAGES>
      <<<blocks, BLOCK_SIZE, DYNAMIC_SHM_SIZE>>>(A, B, C, M, N, K, alpha, beta);
  return true;
}

// Row-major FP16 A[M,K], B[K,N], C[M,N]; FP32 accumulation and alpha/beta.
// Nonnegative dimensions; M/N=0 is a no-op, K=0 scales C by beta.
// Fast tiles require complete dimensions and 16-byte-aligned A/B. C needs
// only half alignment; a misaligned half2 output uses scalar stores instead.
// Default tile selection uses 64x64, 64x128, 96x128, or 128x128 blocks and
// three/four input stages. Small output grids use BK=64 when K is a multiple of
// 64 and has at least three chunks; other configurations retain BK=32. Explicit
// GEMM_BM/BN/BK/WM/WN/STAGES disables automatic selection. Unsupported
// layouts/tails use the fixed 64x64 generic WMMA pipeline. sm_75 uses
// synchronous copies and K=8 MMA; sm_80+ uses async copies and K=16. Larger
// explicit tiles use dynamic shared storage above 48 KiB, with opt-in on the
// active device. An unsupported size falls back to the generic kernel. The
// scheduled 128x128 path is selected only in measured A800 grid bands; other
// shapes and general alpha/beta retain the preceding dispatcher.
template <bool NO_ALPHA_BETA>
static void gemm(const half *__restrict__ A, const half *__restrict__ B,
                 half *__restrict__ C, size_t M, size_t N, size_t K,
                 float alpha, float beta) {
  if (M <= 0 || N <= 0)
    return;

  // XOR swizzle 计算共享内存地址，而这套布局实现要求 tile 宽度是 2 的幂
#if (GEMM_BK & (GEMM_BK - 1)) == 0 && (GEMM_BN & (GEMM_BN - 1)) == 0

  // 1. K 非零
  // 2. M/N/K 分别能被配置的 BM/BN/BK 整除
  // 3. A、B 的起始地址按 16 字节对齐
  const bool full_aligned =
      K > 0 && M % GEMM_BM == 0 && N % GEMM_BN == 0 && K % GEMM_BK == 0 &&
      ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) &
       15) == 0;
  if (full_aligned) {
#if GEMM_AUTO_TILE
    // A800 has 108 SMs. The 128x128 tile works well for a small single wave
    // or a sufficiently populated medium grid; it lost at 1536 cubed and
    // at 6144/8192 cubed. Keep the older choices outside these measured bands.
    const size_t scheduled_tiles = (M / 128) * (N / 128);
    if (NO_ALPHA_BETA && M % 128 == 0 && N % 128 == 0 && K % 32 == 0 &&
        M >= 1024 && N >= 1024 && M <= 4096 && N <= 4096 && K >= 128 &&
        K <= 8192 &&
        ((scheduled_tiles >= 64 && scheduled_tiles <= 108) ||
         (scheduled_tiles >= 256 && scheduled_tiles <= 1024))) {
      // Four stages help long K on grids near 256..324 blocks. If opt-in
      // storage is unavailable, the three-stage static kernel still applies.
      if (scheduled_tiles >= 256 && scheduled_tiles <= 324 && K >= 1024 &&
          launch_schedule<NO_ALPHA_BETA, 128, 128, 32, 64, 64, 4>(
              A, B, C, M, N, K, alpha, beta)) {
        return;
      }
      if (launch_schedule<NO_ALPHA_BETA, 128, 128, 32, 64, 64, 3>(
              A, B, C, M, N, K, alpha, beta)) {
        return;
      }
    }
    // Larger reuse tiles need enough blocks and K work to amortize their
    // register/output costs. Thresholds were measured on A800, not autotuned.
    if (M % 128 == 0 && N % 128 == 0 && K % 32 == 0 && K >= 2048 &&
        (M / 128) * (N / 128) >= 1024) {
      if (launch_schedule<NO_ALPHA_BETA, 128, 128, 32, 64, 64, 3>(
              A, B, C, M, N, K, alpha, beta)) {
        return;
      }
    }
    // The 96x128 tile uses 237 registers/thread (232 for alpha=1/beta=0)
    // and fits two blocks per SM on A800. Small grids cannot amortize its
    // larger per-warp work. M=192
    // multiples satisfy both the base 64-row contract and complete 96 rows.
    // Retain the 128x128 large-matrix path above; its reuse is greater still.
    if (M % 192 == 0 && N % 128 == 0 && K % 32 == 0 &&
        (M / 96) * (N / 128) >= 192) {
      if (launch_schedule<NO_ALPHA_BETA, 96, 128, 32, 48, 64, 3>(
              A, B, C, M, N, K, alpha, beta)) {
        return;
      }
    }
    if (M % 64 == 0 && N % 128 == 0 && K % 32 == 0 && K >= 512 &&
        (M / 64) * (N / 128) >= 256) {
      if (launch_schedule<NO_ALPHA_BETA, 64, 128, 32, 32, 64, 3>(
              A, B, C, M, N, K, alpha, beta)) {
        return;
      }
    }
    // A800: the 48 KiB BK=64 tile permits three resident blocks per SM.
    // Limit the grid to 3 * 108 blocks to avoid an extra wave; BK=32 uses
    // 24 KiB and handles larger grids better. Require a steady-state chunk
    // after the two-stage prologue. These are measured, device-specific
    // thresholds, not a portable autotuner.
    if (M % 64 == 0 && N % 64 == 0 && K % 64 == 0 && K >= 192 &&
        (M / 64) * (N / 64) <= 324) {
      if (launch_schedule<NO_ALPHA_BETA, 64, 64, 64, 32, 32, 3>(
              A, B, C, M, N, K, alpha, beta)) {
        return;
      }
    }
#endif // GEMM_AUTO_TILE

    if (launch_schedule<NO_ALPHA_BETA, GEMM_BM, GEMM_BN, GEMM_BK, GEMM_WM,
                        GEMM_WN, GEMM_STAGES>(A, B, C, M, N, K, alpha, beta)) {
      return;
    }
  }
#endif // (GEMM_BK & (GEMM_BK - 1)) == 0 && (GEMM_BN & (GEMM_BN - 1)) == 0

  // Large aligned tile overrides must not inflate generic shared storage.
  const size_t tiles = ((M + 63) / 64) * ((N + 63) / 64);
  const unsigned blocks = unsigned(tiles < 2147483647u ? tiles : 2147483647u);
  gemm_generic<NO_ALPHA_BETA, 64, 64, 32, 32, 32, 16>
      <<<blocks, 128>>>(A, B, C, M, N, K, alpha, beta);
}

// Dispatch exact scalar values on the host. Other alpha/beta values keep the
// general epilogue. +0 and -0 beta both mean that the previous C is ignored.
extern "C" void solve(const half *A, const half *B, half *C, int M, int N,
                      int K, float alpha, float beta) {
  if (M <= 0 || N <= 0)
    return;
  if (alpha == 1.0f && beta == 0.0f) {
    gemm<true>(A, B, C, M, N, K, alpha, beta);
    return;
  }
  gemm<false>(A, B, C, M, N, K, alpha, beta);
}
