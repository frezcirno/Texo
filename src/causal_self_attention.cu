// Fused FP32 causal self-attention: O = softmax(Q K^T / sqrt(d) + mask) V,
// where query i attends only to keys j <= i.
// One block owns BR query rows and a DV-wide slice of the output columns. It
// walks the keys in BC-wide tiles up to the diagonal: S = Q K_j^T stays in
// registers, the online softmax rescales the running output, and P V_j
// accumulates into it, so the M x M score matrix is never stored. As in
// softmax_online.cu, small grids split the keys across blocks; each split
// writes a partial (max, sum, O), and the last block of an output tile merges
// them.
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

template <int N> __device__ __forceinline__ void copy_wait() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
#endif
}

// Stage a ROWS x COLS tile of a row-major rows x cols matrix into shared
// memory with row stride LD, in groups of four floats. Outside elements are
// zero. VEC: the base is 16-byte aligned and cols % 4 == 0, so each group is
// either complete or entirely outside. Otherwise copy elements one at a time.
template <bool VEC, int ROWS, int COLS, int LD, int THREADS>
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
    float *dst = tile + row * LD + col;
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

constexpr int BR = 64;       // query rows per block
constexpr int BC = 64;       // keys per tile
constexpr int KC = 32;       // head-dimension chunk of the S = Q K^T reduction
constexpr int LDK = KC + 4;  // padded Q/K chunk rows: conflict-free K reads
constexpr int LDP = BC + 4;  // padded P rows
constexpr int THREADS = 256; // 16 x 16 threads, 4 rows each
constexpr int STAGE = (BR + BC) * LDK;
static_assert(BR * LDP <= STAGE, "P reuses a Q/K stage");
// The diagonal tile starts at row0, so every row sees at least one key in
// every tile it visits.
static_assert(BR == BC, "Query and key tiles share the diagonal");

constexpr int MAX_PARTIALS = 512; // (output tile, key split) pairs
constexpr int MAX_DV = 128;

__device__ float g_partial_o[MAX_PARTIALS * BR * MAX_DV];
__device__ float2 g_partial_ml[MAX_PARTIALS * BR]; // (max, exp sum) per row
__device__ unsigned g_done[MAX_PARTIALS];

// Thread (ty, tx) owns rows ty * 4 + i of the block. For S it holds columns
// tx + 16 * c, so lanes read consecutive padded K rows; for O it holds the
// four-column groups tx * 4 + 64 * h. Both layouts keep a row on the 16 lanes
// of one half-warp, so row max and sum are four xor shuffles.
// Scores are rounded like the FP32 reference, s * (1 / sqrt(d)) and then
// expf: with large inputs the logits reach thousands, and folding log2(e)
// into the scale rounds them differently enough to move near-tied weights.
template <bool VEC, int DV>
__global__ __launch_bounds__(THREADS) void attention_kernel(
    const float *__restrict__ Q, const float *__restrict__ K,
    const float *__restrict__ V, float *__restrict__ output, int M, int N,
    int d, float scale, int split_tiles) {
  static_assert(DV % 64 == 0 && DV <= MAX_DV, "Whole float4 column groups");
  constexpr int OC = DV / 16; // output columns per thread
  extern __shared__ __align__(16) float smem[];
  float *vs = smem + 2 * STAGE;

  const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
  // The last query tiles have the most keys, so schedule them first.
  const int tile_m = gridDim.x - 1 - blockIdx.x;
  const int row0 = tile_m * BR, dv0 = blockIdx.y * DV;
  const int split = blockIdx.z, splits = gridDim.z;
  const int tiles_n = min((N + BC - 1) / BC, tile_m + 1); // up to the diagonal
  const int active = (tiles_n + split_tiles - 1) / split_tiles;
  if (split >= active)
    return; // entirely above the diagonal
  const int j_begin = split * split_tiles;
  const int j_end = min(tiles_n, j_begin + split_tiles);
  const int nc = (d + KC - 1) / KC;
  const int chunks = (j_end - j_begin) * nc;

  // Chunks are numbered across key tiles, so the next tile's first chunk is
  // prefetched while the current tile finishes.
  auto load_qk = [&](int t) {
    float *stage = smem + (t & 1) * STAGE;
    const size_t key0 = size_t(j_begin + t / nc) * BC, k0 = (t % nc) * KC;
    stage_tile<VEC, BR, KC, LDK, THREADS>(stage, Q, M, d, row0, k0);
    stage_tile<VEC, BC, KC, LDK, THREADS>(stage + BR * LDK, K, N, d, key0, k0);
  };

  float m[4], l[4], o[4][OC];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    m[i] = -INFINITY;
    l[i] = 0.0f; // this thread's columns only; reduced at the end
#pragma unroll
    for (int c = 0; c < OC; ++c)
      o[i][c] = 0.0f;
  }

  load_qk(0);
  copy_commit();
  int t = 0;
  for (int j = j_begin; j < j_end; ++j) {
    float s[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int c = 0; c < 4; ++c)
        s[i][c] = 0.0f;

    for (int chunk = 0; chunk < nc; ++chunk, ++t) {
      // Chunk t (and V of the previous tile) has arrived, and every warp
      // finished the previous chunk and the previous P V, so the other stage
      // (which held P) and vs can be refilled.
      copy_wait<0>();
      __syncthreads();
      if (chunk == 0)
        stage_tile<VEC, BC, DV, DV, THREADS>(vs, V, N, d, size_t(j) * BC, dv0);
      if (t + 1 < chunks)
        load_qk(t + 1);
      copy_commit();

      const float *qs = smem + (t & 1) * STAGE, *ks = qs + BR * LDK;
#pragma unroll
      for (int k = 0; k < KC; k += 4) {
        float4 q[4], key[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
          q[i] = *reinterpret_cast<const float4 *>(qs + (ty * 4 + i) * LDK + k);
#pragma unroll
        for (int c = 0; c < 4; ++c)
          key[c] =
              *reinterpret_cast<const float4 *>(ks + (tx + 16 * c) * LDK + k);
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            s[i][c] = fmaf(q[i].x, key[c].x, s[i][c]);
            s[i][c] = fmaf(q[i].y, key[c].y, s[i][c]);
            s[i][c] = fmaf(q[i].z, key[c].z, s[i][c]);
            s[i][c] = fmaf(q[i].w, key[c].w, s[i][c]);
          }
      }
    }

    // Online softmax: rescale the running sum and output to the new max.
    // Every row has a valid column in every key tile (key j * BC <= row0), so
    // the new max is finite and the first tile's alpha is exp(-inf) = 0.
    float *ps = smem + ((t - 1) & 1) * STAGE; // the last chunk's stage
    __syncthreads();                          // every warp finished reading it
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      // Rows past M only need some finite key, so they keep the < N bound.
      const int last_key = min(row0 + ty * 4 + i, N - 1);
      float tile_max = -INFINITY;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        s[i][c] =
            j * BC + tx + 16 * c <= last_key ? s[i][c] * scale : -INFINITY;
        tile_max = fmaxf(tile_max, s[i][c]);
      }
#pragma unroll
      for (int offset = 8; offset > 0; offset >>= 1)
        tile_max =
            fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, offset));
      const float next_max = fmaxf(m[i], tile_max);
      const float alpha = expf(m[i] - next_max);
      float tile_sum = 0.0f;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float p = expf(s[i][c] - next_max);
        tile_sum += p;
        ps[(ty * 4 + i) * LDP + tx + 16 * c] = p;
      }
      l[i] = l[i] * alpha + tile_sum;
      m[i] = next_max;
#pragma unroll
      for (int c = 0; c < OC; ++c)
        o[i][c] *= alpha;
    }
    // With one chunk per tile, V shares its group with the newest prefetch.
    if (nc == 1)
      copy_wait<0>();
    __syncthreads();

    // O += P V_j
#pragma unroll 4
    for (int k = 0; k < BC; k += 4) {
      float p[4][4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float4 v =
            *reinterpret_cast<const float4 *>(ps + (ty * 4 + i) * LDP + k);
        p[i][0] = v.x, p[i][1] = v.y, p[i][2] = v.z, p[i][3] = v.w;
      }
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        float b[OC];
#pragma unroll
        for (int h = 0; h < DV / 64; ++h) {
          const float4 v = *reinterpret_cast<const float4 *>(
              vs + (k + kk) * DV + h * 64 + tx * 4);
          b[h * 4] = v.x, b[h * 4 + 1] = v.y, b[h * 4 + 2] = v.z,
                b[h * 4 + 3] = v.w;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int c = 0; c < OC; ++c)
            o[i][c] = fmaf(p[i][kk], b[c], o[i][c]);
      }
    }
  }

  // alpha is uniform across a row, so the per-thread sums reduce once here.
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1)
      l[i] += __shfl_xor_sync(0xffffffff, l[i], offset);

  if (active == 1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = row0 + ty * 4 + i;
      if (row >= M)
        continue;
      const float inverse_sum = 1.0f / l[i];
#pragma unroll
      for (int c = 0; c < OC; ++c) {
        const int col = dv0 + (c / 4) * 64 + tx * 4 + c % 4;
        if (col < d)
          output[size_t(row) * d + col] = o[i][c] * inverse_sum;
      }
    }
    return;
  }

  // Partial for this (output tile, split): unnormalized O and (max, sum).
  const int tile = blockIdx.y * gridDim.x + blockIdx.x;
  const int item = tile * splits + split;
  float *partial_o = g_partial_o + size_t(item) * BR * DV;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int h = 0; h < DV / 64; ++h)
      *reinterpret_cast<float4 *>(partial_o + (ty * 4 + i) * DV + h * 64 +
                                  tx * 4) =
          make_float4(o[i][h * 4], o[i][h * 4 + 1], o[i][h * 4 + 2],
                      o[i][h * 4 + 3]);
    if (tx == 0)
      g_partial_ml[item * BR + ty * 4 + i] = make_float2(m[i], l[i]);
  }
  __threadfence(); // 保证其他 block 能看到 partial

  __shared__ bool is_last;
  __syncthreads();
  if (threadIdx.x == 0) {
    is_last = atomicAdd(&g_done[tile], 1) == unsigned(active) - 1;
    if (is_last)
      g_done[tile] = 0; // 为下一次调用重置计数器
  }
  __syncthreads();
  if (!is_last)
    return;

  // Merge the splits like finalize_partials, with O rescaled alongside the
  // sum. __ldcg skips L1, which may not see other blocks' writes.
  const float *tile_o = g_partial_o + size_t(tile) * splits * BR * DV;
  const float2 *tile_ml = g_partial_ml + tile * splits * BR;
  for (int e = threadIdx.x; e < BR * DV; e += THREADS) {
    const int r = e / DV, c = e % DV;
    const int row = row0 + r, col = dv0 + c;
    if (row >= M || col >= d)
      continue;
    float running_max = -INFINITY, running_sum = 0.0f, value = 0.0f;
    for (int sp = 0; sp < active; ++sp) {
      const float2 ml = __ldcg(tile_ml + sp * BR + r);
      const float partial = __ldcg(tile_o + (size_t(sp) * BR + r) * DV + c);
      const float next_max = fmaxf(running_max, ml.x);
      const float a = expf(running_max - next_max), b = expf(ml.x - next_max);
      running_sum = running_sum * a + ml.y * b;
      value = value * a + partial * b;
      running_max = next_max;
    }
    output[size_t(row) * d + col] = value / running_sum;
  }
}

template <int DV> constexpr size_t smem_bytes() {
  return (2 * STAGE + BC * DV) * sizeof(float);
}

// Opt in to the dynamic shared memory once per kernel and return how many
// blocks fit on one SM, or 0 if the kernel cannot launch on this device.
template <bool VEC, int DV> static int setup(int device) {
  static int cached_device = -1, cached = 0;
  if (device != cached_device) {
    int optin = 0, per_sm = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           device);
    if (size_t(optin) >= smem_bytes<DV>() &&
        cudaFuncSetAttribute(attention_kernel<VEC, DV>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             int(smem_bytes<DV>())) == cudaSuccess &&
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, attention_kernel<VEC, DV>, THREADS, smem_bytes<DV>()) ==
            cudaSuccess)
      cached = per_sm;
    else
      cached = 0;
    cached_device = device;
  }
  return cached;
}

// Split the keys only when the output tiles fill less than one wave, keeping
// at least four key tiles per split. The split size is set by the last query
// tile; earlier tiles stop at the diagonal and use fewer splits.
template <bool VEC, int DV>
static void launch(const float *Q, const float *K, const float *V,
                   float *output, int M, int N, int d, int per_sm, int sms) {
  const int tiles_m = (M + BR - 1) / BR, tiles_d = (d + DV - 1) / DV;
  const int tiles_n = (N + BC - 1) / BC, tiles = tiles_m * tiles_d;
  const int slots = per_sm * sms;
  int splits = 1;
  if (tiles < slots) {
    splits = (slots + tiles - 1) / tiles;
    splits = std::min(splits, tiles_n / 4);
    splits = std::min(splits, MAX_PARTIALS / tiles);
    splits = std::max(splits, 1);
  }
  const int split_tiles = (tiles_n + splits - 1) / splits;
  splits = (tiles_n + split_tiles - 1) / split_tiles; // no empty split
  const float scale = 1.0f / sqrtf(float(d));
  attention_kernel<VEC, DV>
      <<<dim3(tiles_m, tiles_d, splits), THREADS, smem_bytes<DV>()>>>(
          Q, K, V, output, M, N, d, scale, split_tiles);
}

template <bool VEC>
static void dispatch(const float *Q, const float *K, const float *V,
                     float *output, int M, int d) {
  int device = 0, sms = 1;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  // Wider output slices recompute S fewer times, but need more shared memory
  // than pre-Ampere devices allow.
  const int wide = d > 64 ? setup<VEC, 128>(device) : 0;
  if (wide > 0)
    launch<VEC, 128>(Q, K, V, output, M, M, d, wide, sms);
  else
    launch<VEC, 64>(Q, K, V, output, M, M, d, setup<VEC, 64>(device), sms);
}

// Q, K, V and output (M, d), all row-major device pointers.
extern "C" void solve(const float *Q, const float *K, const float *V,
                      float *output, int M, int d) {
  if (M <= 0 || d <= 0)
    return;
  const bool vec = d % 4 == 0 && ((reinterpret_cast<uintptr_t>(Q) |
                                   reinterpret_cast<uintptr_t>(K) |
                                   reinterpret_cast<uintptr_t>(V)) &
                                  15) == 0;
  if (vec)
    dispatch<true>(Q, K, V, output, M, d);
  else
    dispatch<false>(Q, K, V, output, M, d);
}
