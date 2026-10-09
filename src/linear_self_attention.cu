// Linear self-attention (Katharopoulos et al.) with phi(x) = ELU(x) + 1:
// O = phi(Q) (phi(K)^T V) / (phi(Q) z), z = sum_j phi(K_j).
// The d x d state S = phi(K)^T V and z do not depend on the query, so the
// work is two FP32 GEMMs with a small output:
// 1. kv_state_kernel splits the M keys across blocks; each block writes a
//    partial S and z, padded to D x D and D, and reduce_kernel sums them.
// 2. output_kernel computes phi(Q) S and phi(Q) z for BM query rows a block
//    and divides.
// phi is applied once per element in shared memory, after the tile arrives.
// Padding (rows past M, columns past d) is zero after phi, so it adds nothing
// to S, z or the products.
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

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

__device__ __forceinline__ void copy_wait_all() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

// Stage a ROWS x COLS tile of a row-major rows x cols matrix into shared
// memory with row stride LD, in groups of four floats. Outside elements are
// zero. VEC: the base is 16-byte aligned and cols % 4 == 0, so each group is
// either complete or entirely outside. Otherwise copy elements one at a time.
template <bool VEC, int ROWS, int COLS, int LD, int THREADS>
__device__ __forceinline__ void stage_tile(float *tile, const float *input,
                                           int rows, int cols, int row0,
                                           int col0) {
  constexpr int GROUPS = ROWS * COLS / 4;
  static_assert(GROUPS % THREADS == 0, "Whole rounds of groups");
#pragma unroll
  for (int i = 0; i < GROUPS / THREADS; ++i) {
    const int g = threadIdx.x + i * THREADS;
    const int row = g / (COLS / 4), col = (g % (COLS / 4)) * 4;
    const int global_row = row0 + row, global_col = col0 + col;
    const float *src = input + size_t(global_row) * cols + global_col;
    float *dst = tile + row * LD + col;
    if (VEC) {
      const bool valid = global_row < rows && global_col < cols;
      copy_async<16>(dst, valid ? src : input, valid);
    } else {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const bool valid = global_row < rows && global_col + e < cols;
        copy_async<4>(dst + e, valid ? src + e : input, valid);
      }
    }
  }
}

__device__ __forceinline__ float phi(float x) {
  return x > 0.0f ? x + 1.0f : expf(x);
}

constexpr int D = 128;          // padded head dimension
constexpr int THREADS = 256;    // 16 x 16 threads
constexpr int STATE = D * D + D; // S, then z
constexpr int MAX_SPLITS = 256;

__device__ float g_partial[MAX_SPLITS * STATE];
__device__ float g_state[STATE];

// ---- 1. Partial S = phi(K)^T V and z over keys [m_begin, m_end) ----------
constexpr int RK = 32; // keys per chunk
constexpr int KV_STAGE = 2 * RK * D;
constexpr size_t KV_SMEM = 2 * KV_STAGE * sizeof(float);

// Thread (ty, tx) owns S rows ty * 4 + 64 * a + i and columns
// tx * 4 + 64 * b + j, so each key is four float4 reads for 64 FMAs.
template <bool VEC>
__global__ __launch_bounds__(THREADS) void kv_state_kernel(
    const float *__restrict__ K, const float *__restrict__ V, int M, int d,
    int rows_per_split) {
  extern __shared__ __align__(16) float smem[];
  const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
  const int m_begin = blockIdx.x * rows_per_split;
  const int m_end = min(M, m_begin + rows_per_split);
  const int chunks = (m_end - m_begin + RK - 1) / RK;

  auto load = [&](int c) {
    float *ks = smem + (c & 1) * KV_STAGE, *vs = ks + RK * D;
    const int row0 = m_begin + c * RK;
    stage_tile<VEC, RK, D, D, THREADS>(ks, K, m_end, d, row0, 0);
    stage_tile<VEC, RK, D, D, THREADS>(vs, V, m_end, d, row0, 0);
  };

  float acc[8][8] = {};
  float z = 0.0f; // column threadIdx.x % D, every other key
  load(0);
  copy_commit();
  for (int c = 0; c < chunks; ++c) {
    // Chunk c has arrived and every warp finished chunk c - 1, so its stage
    // can be refilled.
    copy_wait_all();
    __syncthreads();
    if (c + 1 < chunks)
      load(c + 1);
    copy_commit();

    float *ks = smem + (c & 1) * KV_STAGE;
    const float *vs = ks + RK * D;
    const int row0 = m_begin + c * RK, col = threadIdx.x % D;
#pragma unroll
    for (int r = threadIdx.x / D; r < RK; r += THREADS / D) {
      const bool valid = row0 + r < m_end && col < d;
      const float f = valid ? phi(ks[r * D + col]) : 0.0f;
      ks[r * D + col] = f;
      z += f;
    }
    __syncthreads();

#pragma unroll 4
    for (int k = 0; k < RK; ++k) {
      float a[8], b[8];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float4 u =
            *reinterpret_cast<const float4 *>(ks + k * D + h * 64 + ty * 4);
        const float4 v =
            *reinterpret_cast<const float4 *>(vs + k * D + h * 64 + tx * 4);
        a[h * 4] = u.x, a[h * 4 + 1] = u.y, a[h * 4 + 2] = u.z,
              a[h * 4 + 3] = u.w;
        b[h * 4] = v.x, b[h * 4 + 1] = v.y, b[h * 4 + 2] = v.z,
              b[h * 4 + 3] = v.w;
      }
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
          acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
  }

  float *partial = g_partial + size_t(blockIdx.x) * STATE;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int row = (i / 4) * 64 + ty * 4 + i % 4;
#pragma unroll
    for (int h = 0; h < 2; ++h)
      *reinterpret_cast<float4 *>(partial + row * D + h * 64 + tx * 4) =
          make_float4(acc[i][h * 4], acc[i][h * 4 + 1], acc[i][h * 4 + 2],
                      acc[i][h * 4 + 3]);
  }
  // Two threads hold each column of z.
  __syncthreads();
  smem[threadIdx.x] = z;
  __syncthreads();
  if (threadIdx.x < D)
    partial[D * D + threadIdx.x] = smem[threadIdx.x] + smem[threadIdx.x + D];
}

// ---- Sum the partial states ----------------------------------------------
__global__ void reduce_kernel(int splits) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= STATE)
    return;
  float sum = 0.0f;
  for (int p = 0; p < splits; ++p)
    sum += g_partial[size_t(p) * STATE + e];
  g_state[e] = sum;
}

// ---- 2. O = phi(Q) S / phi(Q) z for BM query rows -------------------------
constexpr int BM = 32;      // query rows per block
constexpr int OUT_THREADS = 128; // 16 x 8 threads
constexpr int KC = 32;      // head-dimension chunk
constexpr int LDQ = KC + 4; // padded Q rows: conflict-free row reads
constexpr int OUT_STAGE = BM * LDQ + KC * D + KC;
constexpr size_t OUT_SMEM = 2 * OUT_STAGE * sizeof(float);

// Thread (ty, tx) owns rows ty * 4 + i and columns tx * 4 + 64 * h + j. Its
// share of the denominator is k = tx and tx + 16 of each chunk; the 16 lanes
// of a row sit in one half-warp, so the sum is four xor shuffles.
template <bool VEC>
__global__ __launch_bounds__(OUT_THREADS) void output_kernel(
    const float *__restrict__ Q, float *__restrict__ output, int M, int d) {
  extern __shared__ __align__(16) float smem[];
  const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
  const int row0 = blockIdx.x * BM;
  const int chunks = (d + KC - 1) / KC; // phi(Q) is zero past d

  auto load = [&](int c) {
    float *qs = smem + (c & 1) * OUT_STAGE, *ss = qs + BM * LDQ;
    stage_tile<VEC, BM, KC, LDQ, OUT_THREADS>(qs, Q, M, d, row0, c * KC);
    stage_tile<true, KC, D, D, OUT_THREADS>(ss, g_state, D, D, c * KC, 0);
    if (threadIdx.x < KC / 4)
      copy_async<16>(ss + KC * D + threadIdx.x * 4,
                     g_state + D * D + c * KC + threadIdx.x * 4, true);
  };

  float acc[4][8] = {}, den[4] = {};
  load(0);
  copy_commit();
  for (int c = 0; c < chunks; ++c) {
    copy_wait_all();
    __syncthreads();
    if (c + 1 < chunks)
      load(c + 1);
    copy_commit();

    float *qs = smem + (c & 1) * OUT_STAGE;
    const float *ss = qs + BM * LDQ, *zs = ss + KC * D;
    const int col = threadIdx.x % KC;
#pragma unroll
    for (int r = threadIdx.x / KC; r < BM; r += OUT_THREADS / KC) {
      const bool valid = row0 + r < M && c * KC + col < d;
      qs[r * LDQ + col] = valid ? phi(qs[r * LDQ + col]) : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float *q = qs + (ty * 4 + i) * LDQ;
      den[i] = fmaf(q[tx], zs[tx], den[i]);
      den[i] = fmaf(q[tx + 16], zs[tx + 16], den[i]);
    }
#pragma unroll
    for (int k = 0; k < KC; k += 4) {
      float4 q[4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        q[i] = *reinterpret_cast<const float4 *>(qs + (ty * 4 + i) * LDQ + k);
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        float b[8];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float4 v = *reinterpret_cast<const float4 *>(
              ss + (k + kk) * D + h * 64 + tx * 4);
          b[h * 4] = v.x, b[h * 4 + 1] = v.y, b[h * 4 + 2] = v.z,
                b[h * 4 + 3] = v.w;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float a = kk == 0 ? q[i].x
                          : kk == 1 ? q[i].y
                          : kk == 2 ? q[i].z
                                    : q[i].w;
#pragma unroll
          for (int j = 0; j < 8; ++j)
            acc[i][j] = fmaf(a, b[j], acc[i][j]);
        }
      }
    }
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1)
      den[i] += __shfl_xor_sync(0xffffffff, den[i], offset);
    const int row = row0 + ty * 4 + i;
    if (row >= M)
      continue;
    float *out = output + size_t(row) * d;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int col = h * 64 + tx * 4;
      if (VEC && col < d) {
        *reinterpret_cast<float4 *>(out + col) = make_float4(
            acc[i][h * 4] / den[i], acc[i][h * 4 + 1] / den[i],
            acc[i][h * 4 + 2] / den[i], acc[i][h * 4 + 3] / den[i]);
      } else if (!VEC) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
          if (col + j < d)
            out[col + j] = acc[i][h * 4 + j] / den[i];
      }
    }
  }
}

template <bool VEC> static void run(const float *Q, const float *K,
                                    const float *V, float *output, int M,
                                    int d) {
  static bool configured = false;
  static int sms = 1;
  if (!configured) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    cudaFuncSetAttribute(kv_state_kernel<VEC>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         int(KV_SMEM));
    cudaFuncSetAttribute(output_kernel<VEC>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         int(OUT_SMEM));
    configured = true;
  }
  // About one split per SM, each a whole number of chunks.
  const int max_splits = std::min({sms, MAX_SPLITS, (M + RK - 1) / RK});
  const int rows = ((M + max_splits - 1) / max_splits + RK - 1) / RK * RK;
  const int splits = (M + rows - 1) / rows;
  kv_state_kernel<VEC><<<splits, THREADS, KV_SMEM>>>(K, V, M, d, rows);
  reduce_kernel<<<(STATE + THREADS - 1) / THREADS, THREADS>>>(splits);
  output_kernel<VEC><<<(M + BM - 1) / BM, OUT_THREADS, OUT_SMEM>>>(Q, output, M,
                                                              d);
}

// Q, K, V, output are (M, d) row-major device pointers, d <= 128.
extern "C" void solve(const float *Q, const float *K, const float *V,
                      float *output, int M, int d) {
  if (M <= 0 || d <= 0)
    return;
  const bool vec = d % 4 == 0 && ((reinterpret_cast<uintptr_t>(Q) |
                                   reinterpret_cast<uintptr_t>(K) |
                                   reinterpret_cast<uintptr_t>(V) |
                                   reinterpret_cast<uintptr_t>(output)) &
                                  15) == 0;
  if (vec)
    run<true>(Q, K, V, output, M, d);
  else
    run<false>(Q, K, V, output, M, d);
}
