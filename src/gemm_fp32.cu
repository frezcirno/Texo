// FP32 SIMT GEMM derived from gemm_wmma_tiled_pipeline_schedule.cu.
// Tensor-core operand loads and MMA are replaced by CUDA-core FFMA with
// per-thread register tiles; the multi-stage async input pipeline, flattened
// grid-stride tile loop, and grouped tile rasterization are retained. Small
// output grids split K across blocks and accumulate with atomics.
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

// Stage a ROWS x COLS tile of a row-major matrix in groups of four floats.
// VEC: the base is 16-byte aligned and cols % 4 == 0, so each group is either
// complete or entirely outside. Otherwise copy elements one at a time. An
// out-of-bounds source is replaced by the base pointer and never read.
template <bool VEC, int ROWS, int COLS, int THREADS>
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
    if (VEC) {
      const bool valid = global_row < rows && global_col < cols;
      copy_async<16>(
          dst, valid ? input + global_row * cols + global_col : input, valid);
    } else {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const bool valid = global_row < rows && global_col + e < cols;
        copy_async<4>(
            dst + e, valid ? input + global_row * cols + global_col + e : input,
            valid);
      }
    }
  }
}

// Per-thread 16-byte copies of one input tile for the VEC case, set up once
// per output tile. Between K tiles only the K coordinate changes, so each
// copy advances its source by a constant step. The other coordinate is
// checked once: an outside row (A) or column group (B) is clamped to index 0
// and copied as zeros. With K % 4 == 0, a group of the final partial K tile
// is entirely inside or outside K, so one offset comparison covers the tail.
// Group g lands at float offset 4 * g of the row-major tile.
// K_ROWS: K runs along tile rows (B); otherwise along tile columns (A).
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

// Each thread computes an 8x8 output tile, split into 4x4 quarters at BM/2 and
// BN/2. Consecutive lanes take consecutive 4-column groups, so each quarter-
// warp reads 128 contiguous B bytes; its lanes share one A row (broadcast).
// Work items are (K slice, output tile) pairs. With splits > 1, each slice
// atomically adds alpha * partial to C, which the host has set to beta * C.
template <bool VEC, int BM, int BN, int BK, int STAGES>
__global__
__launch_bounds__((BM / 8) * (BN / 8),
                  (BM / 8) * (BN / 8) >= 256 ? 2
                  : (BM / 8) * (BN / 8) >= 128
                      ? 4
                      : 1) void gemm_fp32(const float *__restrict__ A,
                                          const float *__restrict__ B,
                                          float *__restrict__ C, size_t M,
                                          size_t N, size_t K, float alpha,
                                          float beta, bool vec_c, size_t splits,
                                          size_t split_tiles) {
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

  for (size_t work = blockIdx.x; work < tiles * splits; work += gridDim.x) {
    // Consecutive blocks take neighbouring tiles of the same K slice.
    const size_t tile = work % tiles, split = work / tiles;
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

    // K tiles are loaded in order. The aligned specialization uses the
    // incremental copies; the unaligned one checks every element.
    TileCopy<false, BM, BK, THREADS> a_copy;
    TileCopy<true, BK, BN, THREADS> b_copy;
    const int k_tiles = int(k_count);
    if (VEC) {
      a_copy.Init(A, M, K, row0, k_begin * BK);
      b_copy.Init(B, K, N, k_begin * BK, col0);
    }
    // Uniform across the block; at least BK for every complete tile.
    const size_t k_left0 = K - k_begin * BK;
    auto load_tile = [&](int stage, int t) {
      if (VEC) {
        const size_t left = k_left0 - size_t(t) * BK;
        const int k_left = left < BK ? int(left) : BK;
        a_copy.Copy(at[stage], BK, k_left);
        b_copy.Copy(bt[stage], BK * N, k_left);
      } else {
        const size_t k0 = (k_begin + t) * BK;
        stage_tile<VEC, BM, BK, THREADS>(at[stage], A, M, K, row0, k0);
        stage_tile<VEC, BK, BN, THREADS>(bt[stage], B, K, N, k0, col0);
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
      // Tile t is visible to all threads, and every warp finished tile t - 1,
      // whose stage is refilled below.
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
        // Four K values for each of the eight rows.
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
    // Only empty groups can remain. All warps must finish reading before the
    // next tile's prologue overwrites the stages.
    copy_wait<0>();
    __syncthreads();

    // beta == 0 never reads C, so uninitialized or NaN output is overwritten.
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const size_t row = row0 + (i / 4) * (BM / 2) + ty * 4 + i % 4;
      if (row >= M)
        continue;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const size_t col = col0 + h * (BN / 2) + tx * 4;
        float *out = C + row * N + col;
        if (splits > 1) {
#pragma unroll
          for (int e = 0; e < 4; ++e)
            if (col + e < N)
              atomicAdd(out + e, alpha * acc[i][h * 4 + e]);
        } else if (vec_c && col + 4 <= N) {
          float4 v =
              make_float4(alpha * acc[i][h * 4], alpha * acc[i][h * 4 + 1],
                          alpha * acc[i][h * 4 + 2], alpha * acc[i][h * 4 + 3]);
          if (beta != 0.0f) {
            const float4 old = *reinterpret_cast<const float4 *>(out);
            v.x += beta * old.x, v.y += beta * old.y, v.z += beta * old.z,
                v.w += beta * old.w;
          }
          *reinterpret_cast<float4 *>(out) = v;
        } else {
#pragma unroll
          for (int e = 0; e < 4; ++e)
            if (col + e < N)
              out[e] = alpha * acc[i][h * 4 + e] +
                       (beta == 0.0f ? 0.0f : beta * out[e]);
        }
      }
    }
  }
}

// Split-K prologue for beta != 0: C = beta * C.
__global__ void scale_fp32(float *C, size_t count, float beta) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x)
    C[i] *= beta;
}

template <int BM, int BN>
static void launch(const float *A, const float *B, float *C, size_t M, size_t N,
                   size_t K, float alpha, float beta, size_t splits) {
  constexpr int BK = 8, THREADS = (BM / 8) * (BN / 8);
  const size_t tiles_k = (K + BK - 1) / BK;
  // Even K slices; drop any slice the rounding leaves empty.
  const size_t split_tiles =
      splits > 1 && tiles_k > 0 ? (tiles_k + splits - 1) / splits : tiles_k;
  splits = split_tiles > 0 ? (tiles_k + split_tiles - 1) / split_tiles : 1;
  if (splits > 1) {
    if (beta == 0.0f) {
      cudaMemsetAsync(C, 0, M * N * sizeof(float));
    } else if (beta != 1.0f) {
      const size_t blocks = (M * N + 255) / 256;
      scale_fp32<<<unsigned(blocks < 65535 ? blocks : 65535), 256>>>(C, M * N,
                                                                     beta);
    }
  }
  const size_t work = ((M + BM - 1) / BM) * ((N + BN - 1) / BN) * splits;
  const unsigned blocks = unsigned(work < 2147483647u ? work : 2147483647u);
  const bool vec =
      K % 4 == 0 && N % 4 == 0 &&
      ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) &
       15) == 0;
  const bool vec_c = N % 4 == 0 && (reinterpret_cast<uintptr_t>(C) & 15) == 0;
  if (vec)
    gemm_fp32<true, BM, BN, BK, GEMM_STAGES><<<blocks, THREADS>>>(
        A, B, C, M, N, K, alpha, beta, vec_c, splits, split_tiles);
  else
    gemm_fp32<false, BM, BN, BK, GEMM_STAGES><<<blocks, THREADS>>>(
        A, B, C, M, N, K, alpha, beta, vec_c, splits, split_tiles);
}

enum Tile { TILE_AUTO = 0, TILE_128x128 = 1, TILE_128x64 = 2, TILE_64x64 = 3 };

// Resident blocks for one tile configuration on the current device. Cached
// per device; a failed query assumes one block on one SM.
template <int BM, int BN> static size_t resident_blocks() {
  static int cached_device = -1;
  static size_t cached = 1;
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess)
    return 1;
  if (device != cached_device) {
    int sms = 0, per_sm = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) !=
            cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, gemm_fp32<true, BM, BN, 8, GEMM_STAGES>,
            (BM / 8) * (BN / 8), 0) != cudaSuccess)
      sms = per_sm = 1;
    cached = size_t(sms) * size_t(per_sm > 0 ? per_sm : 1);
    cached_device = device;
  }
  return cached;
}

// Split K only for grids under three waves; then aim for four waves while
// keeping at least 16 K tiles (128 values) per slice, and at most 128
// slices. Large grids stay unsplit, so their per-element summation order is
// the plain K loop.
// Thresholds were fit on A800 shapes from 64x64x16384 to 8192x4096x6144.
template <int BM, int BN> static int auto_split(size_t M, size_t N, size_t K) {
  const size_t tiles = ((M + BM - 1) / BM) * ((N + BN - 1) / BN);
  const size_t slots = resident_blocks<BM, BN>(), k_tiles = (K + 7) / 8;
  if (tiles >= 3 * slots)
    return 1;
  size_t split = (4 * slots + tiles - 1) / tiles;
  if (split > k_tiles / 16)
    split = k_tiles / 16;
  return int(split < 1 ? 1 : split > 128 ? 128 : split);
}

// Explicit configuration, for benchmarks and forced-path tests. tile=0 and
// split=0 select automatically: 128x64 tiles were within a few percent of the
// best measured tile on every A800 shape except single-tile outputs.
extern "C" void solve_with_config(const float *A, const float *B, float *C,
                                  int M, int N, int K, float alpha, float beta,
                                  int tile, int split) {
  if (M <= 0 || N <= 0)
    return;
  const size_t m = M, n = N, k = K < 0 ? 0 : K;
  if (tile == TILE_AUTO)
    tile = TILE_128x64;
  if (split <= 0)
    split = tile == TILE_128x128  ? auto_split<128, 128>(m, n, k)
            : tile == TILE_128x64 ? auto_split<128, 64>(m, n, k)
                                  : auto_split<64, 64>(m, n, k);
  if (tile == TILE_128x128)
    launch<128, 128>(A, B, C, m, n, k, alpha, beta, split);
  else if (tile == TILE_128x64)
    launch<128, 64>(A, B, C, m, n, k, alpha, beta, split);
  else
    launch<64, 64>(A, B, C, m, n, k, alpha, beta, split);
}

#ifndef GEMM_FP32_TILE
#define GEMM_FP32_TILE 0
#endif
#ifndef GEMM_FP32_SPLIT
#define GEMM_FP32_SPLIT 0
#endif

// Row-major FP32 A[M,K], B[K,N], C[M,N]: C = alpha * A @ B + beta * C.
// M/N=0 is a no-op; K=0 scales C by beta. Any sizes and float-aligned
// pointers are accepted; 16-byte copies/stores need aligned bases and
// multiples of four in the contiguous dimension. Split-K results are summed
// with atomics, so their rounding order is not deterministic.
extern "C" void solve(const float *A, const float *B, float *C, int M, int N,
                      int K, float alpha, float beta) {
  solve_with_config(A, B, C, M, N, K, alpha, beta, GEMM_FP32_TILE,
                    GEMM_FP32_SPLIT);
}
