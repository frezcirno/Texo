// Batched FP32 GEMM derived from gemm_fp32.cu: C[b] += A[b] @ B[b].
// The batch index is folded into the flattened work loop, so each work item is
// a (K slice, batch, output tile) triple. The rest (multi-stage cp.async input
// pipeline, 8x8 per-thread register tiles, grouped rasterization within a
// batch) is unchanged. Since C is accumulated, split-K slices atomically add
// their partials directly without any prologue.
#include <cstdint>
#include <cuda_runtime.h>

#ifndef GEMM_STAGES
#define GEMM_STAGES 4
#endif
#ifndef GEMM_GROUP_M
#define GEMM_GROUP_M 8
#endif

// cp.async with a source size: bytes past src_size are zero-filled, and a
// zero source size reads nothing. Pre-Ampere builds use ordinary loads.
template <int BYTES>
__device__ __forceinline__ void copy_async(float *dst, const float *src,
                                           bool valid) {
#if __CUDA_ARCH__ >= 800
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  const int src_size = valid ? BYTES : 0;
  if (BYTES == 16)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address),
                 "l"(src), "r"(src_size)
                 : "memory");
  else
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" ::"r"(address),
                 "l"(src), "n"(BYTES), "r"(src_size)
                 : "memory");
#else
  if (BYTES == 16)
    *reinterpret_cast<float4 *>(dst) =
        valid ? *reinterpret_cast<const float4 *>(src)
              : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  else
    *dst = valid ? *src : 0.0f;
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

// Stage a ROWS x COLS tile of a row-major matrix one element at a time; used
// when bases or the contiguous dimension rule out 16-byte copies.
template <int ROWS, int COLS, int THREADS>
__device__ __forceinline__ void stage_tile(float *tile, const float *input,
                                           size_t rows, size_t cols,
                                           size_t row0, size_t col0) {
  constexpr int GROUPS = ROWS * COLS / 4;
#pragma unroll
  for (int i = 0; i < (GROUPS + THREADS - 1) / THREADS; ++i) {
    const int g = threadIdx.x + i * THREADS;
    if (GROUPS % THREADS != 0 && g >= GROUPS)
      break;
    const int row = g / (COLS / 4), col = (g % (COLS / 4)) * 4;
    const size_t global_row = row0 + row, global_col = col0 + col;
    float *dst = tile + row * COLS + col;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const bool valid = global_row < rows && global_col + e < cols;
      copy_async<4>(dst + e,
                    valid ? input + global_row * cols + global_col + e : input,
                    valid);
    }
  }
}

// Per-thread 16-byte copies of one input tile, set up once per output tile;
// see gemm_fp32.cu. K_ROWS: K runs along tile rows (B), else columns (A).
template <bool K_ROWS, int ROWS, int COLS, int THREADS> struct TileCopy {
  static constexpr int GROUPS = ROWS * COLS / 4;
  static constexpr int PER_THREAD = (GROUPS + THREADS - 1) / THREADS;
  const float *src[PER_THREAD];
  bool valid[PER_THREAD];

  __device__ __forceinline__ void Init(const float *input, size_t rows,
                                       size_t cols, size_t row0, size_t col0) {
#pragma unroll
    for (int i = 0; i < PER_THREAD; ++i) {
      const int g = threadIdx.x + i * THREADS;
      const size_t row = row0 + g / (COLS / 4);
      const size_t col = col0 + (g % (COLS / 4)) * 4;
      valid[i] = K_ROWS ? col < cols : row < rows;
      src[i] = input + (K_ROWS ? row * cols + (valid[i] ? col : 0)
                               : (valid[i] ? row : 0) * cols + col);
    }
  }

  // k_left: K values from this tile's first column (A) or row (B) to K.
  __device__ __forceinline__ void Copy(float *tile, size_t step, int k_left) {
#pragma unroll
    for (int i = 0; i < PER_THREAD; ++i) {
      const int g = threadIdx.x + i * THREADS;
      if (GROUPS % THREADS == 0 || g < GROUPS) {
        const int k = K_ROWS ? g / (COLS / 4) : (g % (COLS / 4)) * 4;
        copy_async<16>(tile + 4 * g, src[i], valid[i] && k < k_left);
        src[i] += step;
      }
    }
  }
};

template <bool VEC, int BM, int BN, int BK, int STAGES>
__global__ __launch_bounds__(
    (BM / 8) * (BN / 8),
    (BM / 8) * (BN / 8) >= 128
        ? 2
        : 4) void batched_mm(const float *__restrict__ A,
                             const float *__restrict__ B, float *__restrict__ C,
                             size_t batches, size_t M, size_t N, size_t K,
                             bool vec_c, size_t splits, size_t split_tiles) {
  constexpr int TX = BN / 8, TY = BM / 8, THREADS = TX * TY;
  static_assert(BM % 8 == 0 && BN % 8 == 0 && BK % 4 == 0,
                "Whole thread tiles and float4 groups");
  static_assert(TX >= 8 && TX <= 32 && 32 % TX == 0,
                "A quarter-warp shares one thread row");
  static_assert(STAGES >= 2, "At least double buffering");
  static_assert(STAGES * (BM + BN) * BK * sizeof(float) <= 48 * 1024,
                "Input stages use static shared memory");

  __shared__ __align__(16) float at[STAGES][BM * BK];
  __shared__ __align__(16) float bt[STAGES][BK * BN];

  const int tx = threadIdx.x % TX, ty = threadIdx.x / TX;
  const size_t tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN;
  const size_t tiles_k = (K + BK - 1) / BK, tiles = tiles_m * tiles_n;
  const size_t work_count = tiles * batches * splits;

  for (size_t work = blockIdx.x; work < work_count; work += gridDim.x) {
    // Consecutive blocks take neighbouring tiles of one batch and K slice.
    const size_t tile = work % tiles, rest = work / tiles;
    const size_t batch = rest % batches, split = rest / batches;
    const float *a_batch = A + batch * M * K;
    const float *b_batch = B + batch * K * N;
    float *c_batch = C + batch * M * N;
    const size_t k_begin = split * split_tiles;
    const size_t k_count =
        tiles_k - k_begin < split_tiles ? tiles_k - k_begin : split_tiles;
    // Neighbouring blocks share B columns within a group of GROUP_M rows.
    const size_t group = tile / (GEMM_GROUP_M * tiles_n);
    const size_t first_m = group * GEMM_GROUP_M;
    const size_t remaining_m = tiles_m - first_m;
    const size_t group_m =
        remaining_m < GEMM_GROUP_M ? remaining_m : GEMM_GROUP_M;
    const size_t local = tile % (GEMM_GROUP_M * tiles_n);
    const size_t row0 = (first_m + local % group_m) * BM;
    const size_t col0 = (local / group_m) * BN;

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j)
        acc[i][j] = 0.0f;

    TileCopy<false, BM, BK, THREADS> a_copy;
    TileCopy<true, BK, BN, THREADS> b_copy;
    const int k_tiles = int(k_count);
    if (VEC) {
      a_copy.Init(a_batch, M, K, row0, k_begin * BK);
      b_copy.Init(b_batch, K, N, k_begin * BK, col0);
    }
    const size_t k_left0 = K - k_begin * BK;
    auto load_tile = [&](int stage, int t) {
      if (VEC) {
        const size_t left = k_left0 - size_t(t) * BK;
        const int k_left = left < BK ? int(left) : BK;
        a_copy.Copy(at[stage], BK, k_left);
        b_copy.Copy(bt[stage], BK * N, k_left);
      } else {
        const size_t k0 = (k_begin + t) * BK;
        stage_tile<BM, BK, THREADS>(at[stage], a_batch, M, K, row0, k0);
        stage_tile<BK, BN, THREADS>(bt[stage], b_batch, K, N, k0, col0);
      }
    };

    // Commit even empty groups, so wait<STAGES - 2> always means the
    // current K tile has arrived.
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (s < k_tiles)
        load_tile(s, s);
      copy_commit();
    }

    int read_stage = 0, write_stage = STAGES - 1;
    for (int t = 0; t < k_tiles; ++t) {
      copy_wait<STAGES - 2>();
      __syncthreads();
      if (t + STAGES - 1 < k_tiles)
        load_tile(write_stage, t + STAGES - 1);
      copy_commit();
      write_stage = write_stage + 1 == STAGES ? 0 : write_stage + 1;

      const float *a_tile = at[read_stage];
      const float *b_tile = bt[read_stage];
      read_stage = read_stage + 1 == STAGES ? 0 : read_stage + 1;
#pragma unroll
      for (int k = 0; k < BK; k += 4) {
        float a[8][4];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int row = (i / 4) * (BM / 2) + ty * 4 + i % 4;
          const float4 v =
              *reinterpret_cast<const float4 *>(a_tile + row * BK + k);
          a[i][0] = v.x, a[i][1] = v.y, a[i][2] = v.z, a[i][3] = v.w;
        }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          float b[8];
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const float4 v = *reinterpret_cast<const float4 *>(
                b_tile + (k + kk) * BN + h * (BN / 2) + tx * 4);
            b[h * 4] = v.x, b[h * 4 + 1] = v.y, b[h * 4 + 2] = v.z,
                  b[h * 4 + 3] = v.w;
          }
#pragma unroll
          for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j)
              acc[i][j] = fmaf(a[i][kk], b[j], acc[i][j]);
        }
      }
    }
    copy_wait<0>();
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const size_t row = row0 + (i / 4) * (BM / 2) + ty * 4 + i % 4;
      if (row >= M)
        continue;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const size_t col = col0 + h * (BN / 2) + tx * 4;
        float *out = c_batch + row * N + col;
        if (splits > 1) {
#pragma unroll
          for (int e = 0; e < 4; ++e)
            if (col + e < N)
              atomicAdd(out + e, acc[i][h * 4 + e]);
        } else if (vec_c && col + 4 <= N) {
          float4 v = *reinterpret_cast<const float4 *>(out);
          v.x += acc[i][h * 4], v.y += acc[i][h * 4 + 1],
              v.z += acc[i][h * 4 + 2], v.w += acc[i][h * 4 + 3];
          *reinterpret_cast<float4 *>(out) = v;
        } else {
#pragma unroll
          for (int e = 0; e < 4; ++e)
            if (col + e < N)
              out[e] += acc[i][h * 4 + e];
        }
      }
    }
  }
}

template <int BM, int BN>
static void launch(const float *A, const float *B, float *C, size_t batches,
                   size_t M, size_t N, size_t K) {
  constexpr int BK = 8, THREADS = (BM / 8) * (BN / 8);
  const size_t tiles = batches * ((M + BM - 1) / BM) * ((N + BN - 1) / BN);
  const size_t tiles_k = (K + BK - 1) / BK;

  // Split K only for grids under three waves; aim for four waves while
  // keeping at least 16 K tiles per slice (same rule as gemm_fp32.cu).
  int device = 0, sms = 1, per_sm = 1;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_sm, batched_mm<true, BM, BN, BK, GEMM_STAGES>, THREADS, 0);
  const size_t slots = size_t(sms > 0 ? sms : 1) * (per_sm > 0 ? per_sm : 1);
  size_t splits = 1;
  if (tiles < 3 * slots) {
    splits = (4 * slots + tiles - 1) / tiles;
    if (splits > tiles_k / 16)
      splits = tiles_k / 16;
    splits = splits < 1 ? 1 : splits > 128 ? 128 : splits;
  }
  const size_t split_tiles = (tiles_k + splits - 1) / splits;
  splits = (tiles_k + split_tiles - 1) / split_tiles;

  const size_t work = tiles * splits;
  const unsigned blocks = unsigned(work < 2147483647u ? work : 2147483647u);
  const bool vec =
      K % 4 == 0 && N % 4 == 0 &&
      ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) &
       15) == 0;
  const bool vec_c = N % 4 == 0 && (reinterpret_cast<uintptr_t>(C) & 15) == 0;
  if (vec)
    batched_mm<true, BM, BN, BK, GEMM_STAGES><<<blocks, THREADS>>>(
        A, B, C, batches, M, N, K, vec_c, splits, split_tiles);
  else
    batched_mm<false, BM, BN, BK, GEMM_STAGES><<<blocks, THREADS>>>(
        A, B, C, batches, M, N, K, vec_c, splits, split_tiles);
}

// A, B, and C are device pointers; C[b] += A[b] @ B[b], all row-major.
extern "C" void solve(const float *A, // (BATCH, M, K)
                      const float *B, // (BATCH, K, N)
                      float *C,       // (BATCH, M, N)
                      int BATCH, int M, int N, int K) {
  if (BATCH <= 0 || M <= 0 || N <= 0 || K <= 0)
    return;
  // Small matrices waste most of a 128-row tile.
  if (M <= 64)
    launch<64, 64>(A, B, C, BATCH, M, N, K);
  else
    launch<128, 64>(A, B, C, BATCH, M, N, K);
}
