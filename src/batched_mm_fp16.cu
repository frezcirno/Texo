// Batched FP16 GEMM derived from gemm_wmma_tiled_pipeline_schedule.cu:
// C[b] = A[b] @ B[b] with FP32 accumulation and FP16 output.
// The scheduled path keeps swizzled input stages, ldmatrix + mma.m16n8k16,
// cross-stage operand prefetch, copies interleaved with the MMAs, and the
// shuffled half2 epilogue. The batch index is folded into the flattened tile
// loop, and input copies zero-fill outside M/N/K, so tails need no whole tiles.
// Other layouts use the generic two-stage WMMA kernel.
#include <cstdint>
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
    if (global_row < rows && global_col + PACK <= cols) {
      const half *src = input + global_row * cols + global_col;
      if ((reinterpret_cast<uintptr_t>(src) & 15) == 0) {
        __pipeline_memcpy_async(&tile[row][col], src, 16);
        continue;
      }
    }
#endif
    // Partial/unaligned groups, and pre-Ampere builds, use ordinary loads.
#pragma unroll
    for (int e = 0; e < PACK; ++e) {
      tile[row][col + e] = (global_row < rows && global_col + e < cols)
                               ? input[global_row * cols + global_col + e]
                               : half(0.0f);
    }
  }
}

template <size_t BM, size_t BN, size_t BK, size_t WM, size_t WN, size_t SKEW>
__global__ void batched_generic(const half *__restrict__ A,
                                const half *__restrict__ B,
                                half *__restrict__ C, size_t batches, size_t M,
                                size_t N, size_t K) {
  static_assert(BM % WM == 0 && BN % WN == 0, "Whole warp tiles per block");
  static_assert(WM % 16 == 0 && WN % 16 == 0 && BK % 16 == 0,
                "WMMA dimensions must be multiples of 16");
  static_assert(SKEW % 8 == 0, "Preserve 16-byte row alignment");

  constexpr size_t WARP_NUM = (BM / WM) * (BN / WN);
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
  const size_t tiles = tiles_m * tiles_n;
  for (size_t work = blockIdx.x; work < tiles * batches; work += gridDim.x) {
    const size_t tile = work % tiles, batch = work / tiles;
    const half *a_batch = A + batch * M * K;
    const half *b_batch = B + batch * K * N;
    half *c_batch = C + batch * M * N;
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
      stage_input_tile<BM, BK, BK + SKEW, BLOCK_SIZE>(A_tile[0], a_batch, M, K,
                                                      row0, 0);
      stage_input_tile<BK, BN, BN + SKEW, BLOCK_SIZE>(B_tile[0], b_batch, K, N,
                                                      0, col0);
      __syncwarp();
      __pipeline_commit();
      __pipeline_wait_prior(0);
      __syncthreads();
    }

    for (size_t t = 0; t < tiles_k; ++t) {
      const size_t cur = t & 1, next = cur ^ 1;
      if (t + 1 < tiles_k) {
        const size_t tb = (t + 1) * BK;
        stage_input_tile<BM, BK, BK + SKEW, BLOCK_SIZE>(A_tile[next], a_batch,
                                                        M, K, row0, tb);
        stage_input_tile<BK, BN, BN + SKEW, BLOCK_SIZE>(B_tile[next], b_batch,
                                                        K, N, tb, col0);
        __syncwarp();
        __pipeline_commit();
      }

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
          if (m < M && n < N)
            c_batch[m * N + n] = __float2half_rn(output[warp][row][col]);
        }
        __syncwarp();
      }
    }
  }
}

// cp.async with a source size: an invalid copy reads nothing and zero-fills.
__device__ __forceinline__ void copy16(half *dst, const half *src, bool valid) {
#if __CUDA_ARCH__ >= 800
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address),
               "l"(src), "r"(valid ? 16 : 0)
               : "memory");
#else
  *reinterpret_cast<uint4 *>(dst) =
      valid ? *reinterpret_cast<const uint4 *>(src) : make_uint4(0, 0, 0, 0);
#endif
}

__device__ __forceinline__ void copy_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

template <int N> __device__ __forceinline__ void copy_wait() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
#endif
}

// Permute whole eight-half packs; values inside a 16-byte copy stay contiguous.
// For power-of-two COLS >= 16 this XOR is a bijection. Aligned 8x8 matrices
// span all eight 16-byte bank groups; copies and ldmatrix use the same map.
template <size_t COLS>
__device__ __forceinline__ int swizzled_offset(int row, int col) {
  static_assert(COLS >= 16 && (COLS & (COLS - 1)) == 0,
                "Swizzled input width must be a power of two >= 16");
  return (row * COLS + col) ^ ((row & 7) * 8);
}

// Per-thread 16-byte copies of one swizzled input tile, set up once per output
// tile. Between K chunks each source advances by a constant step. The M/N
// coordinate is checked once: an outside row (A) or column pack (B) is clamped
// to index 0 and zero-filled. With K % 8 == 0, a pack of the final partial K
// chunk is entirely inside or outside K. K_ROWS: K runs along tile rows (B).
template <bool K_ROWS, size_t ROWS, size_t COLS, size_t BLOCK_SIZE>
struct SwizzledInputTile {
  static constexpr size_t WORKS = ROWS * COLS / 8;
  static constexpr size_t TH_WORKS = (WORKS + BLOCK_SIZE - 1) / BLOCK_SIZE;

  const half *source[TH_WORKS];
  int destination[TH_WORKS];
  bool valid[TH_WORKS];
  size_t step;

  __device__ __forceinline__ SwizzledInputTile(const half *input, size_t rows,
                                               size_t cols, size_t row0,
                                               size_t col0, size_t advance)
      : step(advance) {
#pragma unroll
    for (int i = 0; i < TH_WORKS; ++i) {
      const size_t pack = threadIdx.x + i * BLOCK_SIZE;
      const size_t row = pack / (COLS / 8), col = (pack % (COLS / 8)) * 8;
      const size_t global_row = row0 + row, global_col = col0 + col;
      valid[i] = K_ROWS ? global_col < cols : global_row < rows;
      source[i] =
          input + (K_ROWS ? global_row * cols + (valid[i] ? global_col : 0)
                          : (valid[i] ? global_row : 0) * cols + global_col);
      destination[i] = swizzled_offset<COLS>(row, col);
    }
  }

  // k_left: K values from this chunk's first column (A) or row (B) to K.
  // Each copy belongs to exactly one of GROUPS groups; group < 0 copies all.
  // The unrolled caller supplies a constant group, so inactive copies and
  // their pointer updates disappear.
  template <size_t GROUPS = 1>
  __device__ __forceinline__ void Copy(half *tile, int k_left, int group = -1) {
#pragma unroll
    for (int i = 0; i < TH_WORKS; ++i) {
      const size_t pack = threadIdx.x + i * BLOCK_SIZE;
      if ((group < 0 || i * GROUPS / TH_WORKS == group) &&
          (WORKS % BLOCK_SIZE == 0 || pack < WORKS)) {
        const int k = K_ROWS ? pack / (COLS / 8) : (pack % (COLS / 8)) * 8;
        copy16(tile + destination[i], source[i], valid[i] && k < k_left);
        source[i] += step;
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

// Requires K % 8 == 0, N % 8 == 0 and 16-byte-aligned A/B, so every batch
// base and every in-bounds pack is 16-byte aligned. M is unrestricted.
template <size_t BM, size_t BN, size_t BK, size_t WM, size_t WN, int STAGES>
__global__ __launch_bounds__(
    (BM / WM) * (BN / WN) * 32,
    (BM / WM) * (BN / WN) == 4
        ? (BM == 64 && BN == 64 ? 5 : (BM == 128 && BN == 128 ? 2 : 3))
        : 2) void batched_schedule(const half *__restrict__ A,
                                   const half *__restrict__ B,
                                   half *__restrict__ C, size_t batches,
                                   size_t M, size_t N, size_t K) {
  static_assert(BM % WM == 0 && BN % WN == 0 && WM % 16 == 0 && WN % 16 == 0,
                "Whole 16x16 warp fragments and whole warp tiles required");
  static_assert(BK >= 32 && BK % 16 == 0,
                "Cross-stage operand prefetch needs two K groups");
  static_assert(STAGES >= 2 && STAGES <= 4, "Two to four input stages");
  static_assert(STAGES * (BM + BN) * BK * sizeof(half) <= 48 * 1024,
                "Input stages use static shared memory");

  constexpr size_t WARPS = (BM / WM) * (BN / WN), BLOCK_SIZE = WARPS * 32;
  static_assert(WARPS > 0 && WARPS <= 32, "Valid CUDA block size");

  const size_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const size_t warp_m = (warp / (BN / WN)) * WM;
  const size_t warp_n = (warp % (BN / WN)) * WN;

  __shared__ __align__(32) half at[STAGES][BM * BK];
  __shared__ __align__(32) half bt[STAGES][BK * BN];

  const size_t tiles_n = (N + BN - 1) / BN, tiles_m = (M + BM - 1) / BM;
  const size_t tiles = tiles_m * tiles_n;
  const int chunks = int((K + BK - 1) / BK);
  const bool aligned_c = (reinterpret_cast<uintptr_t>(C) & 3) == 0;

  for (size_t work = blockIdx.x; work < tiles * batches; work += gridDim.x) {
    // Consecutive blocks take row-major tiles of the same batch.
    const size_t tile = work % tiles, batch = work / tiles;
    const half *a_batch = A + batch * M * K;
    const half *b_batch = B + batch * K * N;
    half *c_batch = C + batch * M * N;
    const size_t row0 = (tile / tiles_n) * BM;
    const size_t col0 = (tile % tiles_n) * BN;

    float acc[WM / 16][WN / 8][4];
#pragma unroll
    for (int i = 0; i < WM / 16; ++i)
#pragma unroll
      for (int j = 0; j < WN / 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          acc[i][j][e] = 0;

    SwizzledInputTile<false, BM, BK, BLOCK_SIZE> a_copy(a_batch, M, K, row0, 0,
                                                        BK);
    SwizzledInputTile<true, BK, BN, BLOCK_SIZE> b_copy(b_batch, K, N, 0, col0,
                                                       BK * N);
    // Uniform across the block; at least BK for every complete chunk.
    auto k_left = [&](int chunk) {
      const size_t left = K - size_t(chunk) * BK;
      return left < BK ? int(left) : int(BK);
    };

    // Commit even empty drain groups. Keeping a fixed group distance lets
    // wait(STAGES - 2) make the current chunk ready for short K as well.
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (s < chunks) {
        a_copy.Copy(at[s], k_left(s));
        b_copy.Copy(bt[s], k_left(s));
      }
      copy_commit();
    }

    copy_wait<STAGES - 2>();
    __syncthreads();

    int cur = 0, next = STAGES - 1;
    ScheduledOperands<BM, BN, BK> operands;
    operands.Init(at, bt, warp_m, warp_n, lane);

    uint32_t a[2][WM / 16][4], b[2][WN / 16][4];
#pragma unroll
    for (int i = 0; i < WM / 16; ++i)
      operands.LoadA(a[0][i], 0, 0, i);
#pragma unroll
    for (int j = 0; j < WN / 16; ++j)
      operands.LoadB(b[0][j], 0, 0, j);

    for (int t = 0; t < chunks; ++t) {
      const bool refill = t + STAGES - 1 < chunks;
      const int refill_left = refill ? k_left(t + STAGES - 1) : 0;
#pragma unroll
      for (int k = 0; k < BK / 16; ++k) {
        const int slot = k & 1;
        const int load_k = k + 1, load_slot = slot ^ 1;
        const bool prefetch = load_k < BK / 16 || t + 1 < chunks;
        const int operand_k = load_k < BK / 16 ? load_k : 0;
#pragma unroll
        for (int i = 0; i < WM / 16; ++i) {
#pragma unroll
          for (int j = 0; j < WN / 8; ++j) {
            mma(acc[i][j], a[slot][i], &b[slot][j / 2][(j % 2) * 2]);
            // Keep each B fragment's reuse across all M fragments. Only its
            // other register slot is filled, once per K group.
            if (prefetch && i == 0 && j % 2 == 1)
              operands.LoadB(b[load_slot][j / 2], cur, operand_k, j / 2);
          }
          if (prefetch)
            operands.LoadA(a[load_slot][i], cur, operand_k, i);
          // Finish issuing the next stage before the final K group.
          if (k < BK / 16 - 1 && refill) {
            constexpr int GROUPS = (BK / 16 - 1) * (WM / 16);
            const int group = k * (WM / 16) + i;
            a_copy.template Copy<GROUPS>(at[next], refill_left, group);
            b_copy.template Copy<GROUPS>(bt[next], refill_left, group);
          }
        }
        if (k == BK / 16 - 2) {
          // The last K group's operands are now in registers. All warps can
          // release cur before its MMA, then prefetch the next stage into the
          // alternate register slot during that final group.
          copy_commit();
          copy_wait<STAGES - 2>();
          __syncthreads();
          cur = cur + 1 == STAGES ? 0 : cur + 1;
          next = next + 1 == STAGES ? 0 : next + 1;
        }
      }
    }

    copy_wait<0>();
    // The next tile's prologue refills stages 0..STAGES-2; every warp must
    // have finished its final ldmatrix reads first.
    __syncthreads();

    // Pack each lane's adjacent output values, then exchange the two N=8
    // fragments so a warp writes four complete 16-element rows. All lanes
    // shuffle; only in-bounds rows/columns store. With N % 8 == 0, a pair is
    // entirely inside or outside N.
#pragma unroll
    for (int i = 0; i < WM / 16; ++i)
#pragma unroll
      for (int j = 0; j < WN / 16; ++j)
#pragma unroll
        for (int half_rows = 0; half_rows < 2; ++half_rows) {
          uint32_t packed[2];
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const __half2_raw pair =
                __floats2half2_rn(acc[i][j * 2 + h][half_rows * 2],
                                  acc[i][j * 2 + h][half_rows * 2 + 1]);
            packed[h] = uint32_t(pair.x) | (uint32_t(pair.y) << 16);
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
            if (row >= M || col >= N)
              continue;
            const size_t index = row * N + col;
            __half2_raw pair;
            pair.x = static_cast<unsigned short>(value);
            pair.y = static_cast<unsigned short>(value >> 16);
            if (aligned_c) {
              *reinterpret_cast<half2 *>(c_batch + index) = half2(pair);
            } else {
              c_batch[index] = __ushort_as_half(pair.x);
              c_batch[index + 1] = __ushort_as_half(pair.y);
            }
          }
        }
  }
}

template <size_t BM, size_t BN, size_t BK, size_t WM, size_t WN, int STAGES>
static void launch_schedule(const half *A, const half *B, half *C,
                            size_t batches, size_t M, size_t N, size_t K) {
  constexpr int BLOCK_SIZE = (BM / WM) * (BN / WN) * 32;
  const size_t work = batches * ((M + BM - 1) / BM) * ((N + BN - 1) / BN);
  const unsigned blocks = unsigned(work < 2147483647u ? work : 2147483647u);
  batched_schedule<BM, BN, BK, WM, WN, STAGES>
      <<<blocks, BLOCK_SIZE>>>(A, B, C, batches, M, N, K);
}

// A, B, and C are device pointers; C[b] = A[b] @ B[b], all row-major, FP32
// accumulation. C needs only half alignment; a misaligned half2 output uses
// scalar stores instead. The scheduled path needs K % 8 == 0, N % 8 == 0 and
// 16-byte-aligned A/B; other inputs use the generic WMMA kernel.
extern "C" void solve(const half *A, // (BATCH, M, K)
                      const half *B, // (BATCH, K, N)
                      half *C,       // (BATCH, M, N)
                      int BATCH, int M, int N, int K) {
  if (M <= 0 || N <= 0 || BATCH <= 0)
    return;
  const size_t batches = BATCH, m = M, n = N, k = K < 0 ? 0 : K;
  const bool scheduled =
      k > 0 && k % 8 == 0 && n % 8 == 0 &&
      ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) &
       15) == 0;
  if (scheduled) {
    // A800 (108 SMs): 128x128 tiles won once both matrix sides reach 512 and
    // they fill a wave (3x1000^3: 142 vs 84 TFLOP/s); 64x64 won on smaller
    // matrices or grids (128x256^3: 92 vs 87). Measured, not autotuned.
    const size_t big_tiles = batches * ((m + 127) / 128) * ((n + 127) / 128);
    if (m >= 512 && n >= 512 && big_tiles >= 108)
      launch_schedule<128, 128, 32, 64, 64, 3>(A, B, C, batches, m, n, k);
    else
      launch_schedule<64, 64, 32, 32, 32, 3>(A, B, C, batches, m, n, k);
    return;
  }
  const size_t work = batches * ((m + 63) / 64) * ((n + 63) / 64);
  const unsigned blocks = unsigned(work < 2147483647u ? work : 2147483647u);
  batched_generic<64, 64, 32, 32, 32, 16>
      <<<blocks, 128>>>(A, B, C, batches, m, n, k);
}
