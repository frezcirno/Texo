#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// y[m,n] = sum_k x[m,k] * (q[n,k] - 8) * scales[n, k / group_size], where
// w_q[n,i] holds q[n,2i] in the high nibble and q[n,2i+1] in the low one.

// Fallback for group_size < 16 or K % 32 != 0, in the reference's FP32 order.
__global__ void mm_int4_naive(const __half *__restrict__ x,
                              const uint8_t *__restrict__ w_q,
                              const __half *__restrict__ scales,
                              __half *__restrict__ y, int M, int N, int K,
                              int group_size) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N)
    return;
  const __half *xr = x + size_t(m) * K;
  const uint8_t *wr = w_q + size_t(n) * (K / 2);
  const __half *sr = scales + size_t(n) * (K / group_size);
  float dot = 0;
  for (int k = 0; k < K; k += 2) {
    const float s = __half2float(sr[k / group_size]);
    const int byte = wr[k / 2];
    dot += __half2float(xr[k]) * (float((byte >> 4) - 8) * s);
    dot += __half2float(xr[k + 1]) * (float((byte & 15) - 8) * s);
  }
  y[size_t(m) * N + n] = __float2half_rn(dot);
}

// Tensor core path. The MMA multiplies x by the exact integers q-8 in FP16
// with FP32 accumulation, and the scales are applied in FP32 between groups:
// at the end of group g the accumulator is multiplied by s_g / s_{g+1} (by
// s_g for the last group), so it always holds sum_{i<=g} P_i s_i / s_g and
// finally sum_i P_i s_i, within a few FP32 roundings of the reference.
// (Folding the scale into FP16 weights first rounds q*s and exceeds atol on
// a few near-zero outputs of the 4096^3 case; a separate per-group partial
// accumulator would cost another 64 registers per thread.)
#ifndef INT4_BK
#define INT4_BK 64
#endif
#ifndef INT4_STAGES
#define INT4_STAGES 3
#endif
#ifndef INT4_GROUP_M
#define INT4_GROUP_M 8
#endif
#ifndef INT4_BM
#define INT4_BM 128
#endif
#ifndef INT4_BN
#define INT4_BN 128
#endif
#ifndef INT4_WM
#define INT4_WM 128
#endif
#ifndef INT4_WN
#define INT4_WN 32
#endif
constexpr int BM = INT4_BM, BN = INT4_BN, BK = INT4_BK;
constexpr int WM = INT4_WM, WN = INT4_WN;
constexpr int STAGES = INT4_STAGES;
constexpr int WARPS = (BM / WM) * (BN / WN);
constexpr int THREADS = WARPS * 32;
constexpr int A_CHUNKS = BK / 8, B_CHUNKS = BK / 32; // 16-byte chunks per row
constexpr int A_BYTES = BM * BK * 2;
constexpr int B_BYTES = BN * BK / 2;
constexpr int STAGE_BYTES = A_BYTES + B_BYTES;
constexpr int SMEM_BYTES = STAGES * STAGE_BYTES;
static_assert(BK == 64, "swizzles and GS dispatch assume 64-wide K tiles");

// x tile: XOR the 16-byte chunk with row % 8, so the eight row addresses of
// one ldmatrix phase cover all 32 banks.
__device__ __forceinline__ int a_offset(int row, int chunk) {
  return row * (BK * 2) + ((chunk ^ (row & 7)) << 4);
}

// w tile: a warp reads 8 bytes from the same offset of rows r..r+7; each
// half-warp phase covers four 32-byte rows, so no swizzle is needed.
__device__ __forceinline__ int b_offset(int row, int chunk) {
  return row * (BK / 2) + (chunk << 4);
}

// 16-byte global->shared copy, zero filled when !valid.
__device__ __forceinline__ void copy16(void *dst, const void *src, bool valid) {
#if __CUDA_ARCH__ >= 800
  const unsigned addr = unsigned(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(addr),
               "l"(src), "r"(valid ? 16 : 0));
#else
  int4 v = make_int4(0, 0, 0, 0);
  if (valid)
    v = *reinterpret_cast<const int4 *>(src);
  *reinterpret_cast<int4 *>(dst) = v;
#endif
}

__device__ __forceinline__ void copy_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;");
#endif
}

template <int N> __device__ __forceinline__ void copy_wait() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
#endif
}

__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

// D += A*B for one m16n8k16 tile, FP16 inputs and FP32 accumulation.
__device__ __forceinline__ void mma(float (&c)[4], const unsigned (&a)[4],
                                    unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
  // Two k8 halves: K slots 0-7 are a0/a1/b0, slots 8-15 are a2/a3/b1.
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(b0));
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[2]), "r"(a[3]), "r"(b1));
#endif
}

// Byte t of word (select = t, 4, t, 4: 0x64 from the constant) ->
// half2(hi - 8, lo - 8): weights k, k+1 of one B fragment register. Byte 0
// keeps the high nibble in place (0x64x0 is 1024 + 16*hi) and byte 2 the low
// one (0x640x is 1024 + lo); one HFMA2 then removes the offsets exactly.
__device__ __forceinline__ unsigned dequant(unsigned word, unsigned select) {
  const unsigned bits = (__byte_perm(word, 0x64646464u, select) & 0xff0ffff0u);
  const __half2 v =
      __hfma2(*reinterpret_cast<const __half2 *>(&bits),
              __halves2half2(__float2half(1.f / 16), __float2half(1.f)),
              __halves2half2(__float2half(-72.f), __float2half(-1032.f)));
  return *reinterpret_cast<const unsigned *>(&v);
}

// Zero scales become 2^-64, so the ratios stay finite and the zero-scale
// group's contribution is negligible rather than NaN.
__device__ __forceinline__ float safe_scale(__half s) {
  const float v = __half2float(s);
  return v == 0.f ? 0x1p-64f : v;
}

// factor[group][n]: s_g / s_{g+1}, or s_g for the last group, in FP32; zero
// for padding columns n >= N and one for padding rows group >= groups, so
// loads need no predicates.
__global__ void scale_factors(const __half *__restrict__ scales,
                              float *__restrict__ factor, int N, int Np,
                              int groups) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (groups + BK / 16) * Np)
    return;
  const int group = i / Np, n = i % Np;
  float f = group < groups ? 0.f : 1.f;
  if (n < N && group < groups) {
    const __half *s = scales + size_t(n) * groups + group;
    f = group + 1 < groups ? safe_scale(s[0]) / safe_scale(s[1])
                           : safe_scale(s[0]);
  }
  factor[i] = f;
}

// GS: k16 steps per group, capped at a tile (BK / 16); larger groups span
// tile_mask + 1 tiles.
template <int GS>
__global__ void __launch_bounds__(THREADS)
    mm_int4_mma(const __half *__restrict__ x, const uint8_t *__restrict__ w_q,
                const float *__restrict__ factor, __half *__restrict__ y, int M,
                int N, int K, int Np, int tile_mask) {
  constexpr int MI = WM / 16, NI = WN / 8;
  extern __shared__ __align__(128) uint8_t smem[];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int g = lane / 4, t = lane % 4;
  const int warp_row = (warp / (BN / WN)) * WM;
  const int warp_col = (warp % (BN / WN)) * WN;
  // Block swizzle: walk a group of INT4_GROUP_M tile rows column by column,
  // so concurrently running blocks share x rows and w columns in L2.
  const int tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN;
  const int group_tiles = INT4_GROUP_M * tiles_n;
  const int first_m = blockIdx.x / group_tiles * INT4_GROUP_M;
  const int group_m = min(tiles_m - first_m, INT4_GROUP_M);
  const int in_group = blockIdx.x % group_tiles;
  const int row0 = (first_m + in_group % group_m) * BM;
  const int col0 = (in_group / group_m) * BN;
  const int tiles_k = (K + BK - 1) / BK;

  // Each thread copies the same chunk column of A_COPIES x rows and
  // B_COPIES w rows per stage, A_STEP (B_STEP) rows apart. K % 32 == 0, so
  // every chunk is entirely inside or outside K. Rows past M or N and the K
  // tail are zero filled; cp.async reads nothing from their addresses.
  constexpr int A_STEP = THREADS / A_CHUNKS, A_COPIES = BM / A_STEP;
  constexpr int B_STEP = THREADS / B_CHUNKS, B_COPIES = BN / B_STEP;
  static_assert(A_STEP % 8 == 0, "copies keep their row's swizzle");
  const int a_row = threadIdx.x / A_CHUNKS, a_chunk = threadIdx.x % A_CHUNKS;
  const int b_row = threadIdx.x / B_CHUNKS, b_chunk = threadIdx.x % B_CHUNKS;
  const __half *a_src = x + size_t(row0 + a_row) * K + a_chunk * 8;
  const uint8_t *b_src = w_q + size_t(col0 + b_row) * (K / 2) + b_chunk * 16;
  const int a_stride = A_STEP * K, b_stride = B_STEP * (K / 2);
  const int a_rows =
      min(A_COPIES, max(0, (M - row0 - a_row + A_STEP - 1) / A_STEP));
  const int b_rows =
      min(B_COPIES, max(0, (N - col0 - b_row + B_STEP - 1) / B_STEP));
  const int a_dst = a_offset(a_row, a_chunk);
  const int b_dst = A_BYTES + b_offset(b_row, b_chunk);
  int k_left = K - a_chunk * 8, kb_left = K - b_chunk * 32;

  // Copies the next K tile into stage `stage` and advances the sources.
  auto load_stage = [&](int stage) {
    uint8_t *tile = smem + stage * STAGE_BYTES;
#pragma unroll
    for (int j = 0; j < A_COPIES; ++j)
      copy16(tile + a_dst + j * A_STEP * BK * 2, a_src + j * a_stride,
             j < a_rows && k_left > 0);
#pragma unroll
    for (int j = 0; j < B_COPIES; ++j)
      copy16(tile + b_dst + j * B_STEP * (BK / 2), b_src + j * b_stride,
             j < b_rows && kb_left > 0);
    a_src += BK;
    b_src += BK / 2;
    k_left -= BK;
    kb_left -= BK;
  };

  float acc[MI][NI][4] = {};
  float2 scale[NI];
  // Factors of columns n_base + ni*8 + {0,1}; advanced by one group row.
  const int n_base = col0 + warp_col + t * 2;
  const float *scale_src = factor + n_base;
  const unsigned select = 0x4040u + 0x0101u * t; // bytes t, 0x64, t, 0x64
  const unsigned smem_base = unsigned(__cvta_generic_to_shared(smem));
  // ldmatrix lane -> (row, chunk offset): matrices are (rows 0-7, k 0-7),
  // (rows 8-15, k 0-7), (rows 0-7, k 8-15), (rows 8-15, k 8-15), which is
  // the m16n8k16 A fragment order a0..a3. Rows mi*16 apart share row % 8.
  const int lm_row = warp_row + (lane % 8) + ((lane >> 3) & 1) * 8;
  const unsigned a_frag = smem_base + lm_row * (BK * 2);
  const int a_flip = (lane >> 4) ^ (lm_row & 7);
  const uint8_t *b_frag = smem + A_BYTES + (warp_col + g) * (BK / 2);

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < tiles_k)
      load_stage(s);
    copy_commit();
  }

  int read_stage = 0, write_stage = STAGES - 1;
  for (int kt = 0; kt < tiles_k; ++kt) {
    copy_wait<STAGES - 2>();
    __syncthreads();
    // The stage being refilled was consumed in iteration kt-1; the barrier
    // above guarantees every warp has finished with it.
    if (kt + STAGES - 1 < tiles_k)
      load_stage(write_stage);
    copy_commit();
    write_stage = write_stage == STAGES - 1 ? 0 : write_stage + 1;

    const unsigned a_tile = a_frag + read_stage * STAGE_BYTES;
    const uint8_t *b_tile = b_frag + read_stage * STAGE_BYTES;
    read_stage = read_stage == STAGES - 1 ? 0 : read_stage + 1;
    // Groups of at least a tile start and end on tile boundaries.
    const bool tile_start = GS < BK / 16 || (kt & tile_mask) == 0;
    const bool tile_end = GS < BK / 16 || ((kt + 1) & tile_mask) == 0;
    // Straight-line steps (no break for the K tail: its zero-filled steps
    // add nothing, and factor rows past the last group are 1).
#pragma unroll
    for (int ks = 0; ks < BK / 16; ++ks) {
      if (ks % GS == 0 && tile_start) {
        // Loaded at the start of a group, used at its end.
#pragma unroll
        for (int ni = 0; ni < NI; ++ni)
          scale[ni] = *reinterpret_cast<const float2 *>(scale_src + ni * 8);
        scale_src += Np;
      }

      // B fragment of n = warp_col + ni*8 + g: b0 holds k = 2t, 2t+1 (byte t
      // of the step's 8 bytes) and b1 holds k = 2t+8, 2t+9 (byte t+4). The
      // four lanes of a quad read the same 8 bytes.
      unsigned b[NI][2];
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) {
        const uint2 w = *reinterpret_cast<const uint2 *>(
            b_tile + ni * 8 * (BK / 2) + ks * 8);
        b[ni][0] = dequant(w.x, select);
        b[ni][1] = dequant(w.y, select);
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        unsigned a[4];
        ldmatrix_x4(a,
                    a_tile + mi * 16 * (BK * 2) + (((ks * 2) ^ a_flip) << 4));
#pragma unroll
        for (int ni = 0; ni < NI; ++ni)
          mma(acc[mi][ni], a, b[ni][0], b[ni][1]);
      }

      if ((ks + 1) % GS == 0 && tile_end) {
#pragma unroll
        for (int mi = 0; mi < MI; ++mi)
#pragma unroll
          for (int ni = 0; ni < NI; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e)
              acc[mi][ni][e] *= e % 2 ? scale[ni].y : scale[ni].x;
      }
    }
  }
  copy_wait<0>();

  // Accumulator fragment: c[h*2+j] is row g+8h, column 2t+j of the n8 tile.
  const bool pairs = N % 2 == 0 && (reinterpret_cast<uintptr_t>(y) & 3) == 0;
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int m = row0 + warp_row + mi * 16 + g + h * 8;
      if (m >= M)
        continue;
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) {
        const int n = n_base + ni * 8;
        __half *dst = y + size_t(m) * N + n;
        if (pairs && n + 1 < N) {
          *reinterpret_cast<__half2 *>(dst) =
              __floats2half2_rn(acc[mi][ni][h * 2], acc[mi][ni][h * 2 + 1]);
        } else {
          if (n < N)
            dst[0] = __float2half_rn(acc[mi][ni][h * 2]);
          if (n + 1 < N)
            dst[1] = __float2half_rn(acc[mi][ni][h * 2 + 1]);
        }
      }
    }
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

static int compute_major() {
  static int major = -1;
  if (major < 0) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    int minor = 0;
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    if (major * 10 + minor < 75)
      major = 0;
    if (major > 0) {
      cudaFuncSetAttribute(mm_int4_mma<1>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           SMEM_BYTES);
      cudaFuncSetAttribute(mm_int4_mma<2>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           SMEM_BYTES);
      cudaFuncSetAttribute(mm_int4_mma<BK / 16>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           SMEM_BYTES);
    }
  }
  return major;
}

// x, w_q, scales, y are device pointers
extern "C" void solve(const __half *x,      // (M, K)
                      const uint8_t *w_q,   // (N, K/2)
                      const __half *scales, // (N, K/G)
                      __half *y,            // (M, N)
                      int M, int N, int K, int group_size) {
  if (M <= 0 || N <= 0)
    return;
  if (compute_major() == 0 || group_size < 16 || K % 32 != 0) {
    dim3 blockDim(32, 8);
    dim3 gridDim((N + 31) / 32, (M + 7) / 8);
    mm_int4_naive<<<gridDim, blockDim>>>(x, w_q, scales, y, M, N, K,
                                         group_size);
    return;
  }
  const int Np = (N + BN - 1) / BN * BN, groups = K / group_size;
  const int rows = groups + BK / 16;
  float *factor =
      static_cast<float *>(scratch(size_t(rows) * Np * sizeof(float)));
  scale_factors<<<(rows * Np + 255) / 256, 256>>>(scales, factor, N, Np,
                                                  groups);
  const unsigned tiles = unsigned((M + BM - 1) / BM) * unsigned(Np / BN);
  const int steps = group_size / 16;
  auto kernel = steps == 1   ? mm_int4_mma<1>
                : steps == 2 ? mm_int4_mma<2>
                             : mm_int4_mma<BK / 16>;
  const int tile_mask = group_size >= BK ? group_size / BK - 1 : 0;
  kernel<<<tiles, THREADS, SMEM_BYTES>>>(x, w_q, factor, y, M, N, K, Np,
                                         tile_mask);
}
