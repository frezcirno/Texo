#include <cmath>
#include <cuda_runtime.h>

// FlashAttention-style tiled GQA (fp32 SIMT).
//
// Q/O are [num_q_heads, S, d] and the `group` query heads sharing a KV head are
// adjacent, so for one KV head their rows form a contiguous M = group * S row
// matrix. Each block takes BR of those rows (possibly spanning several query
// heads) and streams K/V in BC-key tiles, so every K/V tile loaded into shared
// memory is reused by BR queries.
//
// Thread layout: NTY x NTX threads, each owns 4 consecutive query rows.
// The NTX threads of one row group are adjacent lanes, so row max/sum
// reductions are warp shuffles and the softmax state stays in registers.
//   S tile (BR x BC): thread holds 4 rows x SC = BC / NTX adjacent keys.
//   O tile (BR x D):  thread holds 4 rows x OG float4 column groups.
// head_dim is zero-padded to D in shared memory.

template <int D, int BR, int BC, int THREADS> struct GqaConfig {
  static constexpr int NTY = BR / 4;
  static constexpr int NTX = THREADS / NTY;
  static constexpr int SC = BC / NTX;      // keys per thread in the S tile
  static constexpr int OG = D / (NTX * 4); // float4 output groups per thread
  static constexpr int SMEM_FLOATS = D * BR + D * BC + BC * BR;
  static_assert(NTY * NTX == THREADS, "bad thread layout");
  static_assert(NTX <= 32 && 32 % NTX == 0, "row group must fit in a warp");
  static_assert(SC * NTX == BC && OG * NTX * 4 == D, "bad tile shape");
};

template <int D, int BR, int BC, int THREADS>
__global__ void __launch_bounds__(THREADS)
    gqa_flash_kernel(const float *__restrict__ Q, const float *__restrict__ K,
                     const float *__restrict__ V, float *__restrict__ O,
                     int group, int S, int d, float q_scale) {
  using Cfg = GqaConfig<D, BR, BC, THREADS>;
  constexpr int NTX = Cfg::NTX, SC = Cfg::SC, OG = Cfg::OG;

  extern __shared__ __align__(16) float smem[];
  float *q_t = smem;        // [D][BR], Q transposed and pre-scaled
  float *kv = q_t + D * BR; // K tile as [D][BC], then V tile as [BC][D]
  float *p_t = kv + D * BC; // [BC][BR], P transposed

  const int tid = threadIdx.x;
  const int tx = tid % NTX;
  const int ty = tid / NTX;
  const int kv_head = blockIdx.y;
  const int M = group * S;
  const int row0 = blockIdx.x * BR;
  const int d4 = d / 4;

  const float *q_base = Q + size_t(kv_head) * M * d;
  const float *k_base = K + size_t(kv_head) * S * d;
  const float *v_base = V + size_t(kv_head) * S * d;
  float *o_base = O + size_t(kv_head) * M * d;

  // Load Q tile transposed; consecutive threads take consecutive rows so the
  // transposed shared-memory stores are conflict-free.
  for (int idx = tid; idx < BR * D / 4; idx += THREADS) {
    const int r = idx % BR, k4 = idx / BR;
    float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
    if (row0 + r < M && k4 < d4)
      val = reinterpret_cast<const float4 *>(q_base + size_t(row0 + r) * d)[k4];
    q_t[(k4 * 4 + 0) * BR + r] = val.x * q_scale;
    q_t[(k4 * 4 + 1) * BR + r] = val.y * q_scale;
    q_t[(k4 * 4 + 2) * BR + r] = val.z * q_scale;
    q_t[(k4 * 4 + 3) * BR + r] = val.w * q_scale;
  }

  float acc[4][OG][4];
  float row_max[4], row_sum[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    row_max[i] = -INFINITY;
    row_sum[i] = 0.f; // per-thread partial, reduced across the row at the end
#pragma unroll
    for (int g = 0; g < OG; ++g)
#pragma unroll
      for (int e = 0; e < 4; ++e)
        acc[i][g][e] = 0.f;
  }

  for (int key0 = 0; key0 < S; key0 += BC) {
    // ---- K tile, transposed to [D][BC] ----
    for (int idx = tid; idx < BC * D / 4; idx += THREADS) {
      const int c = idx % BC, k4 = idx / BC;
      float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
      if (key0 + c < S && k4 < d4)
        val =
            reinterpret_cast<const float4 *>(k_base + size_t(key0 + c) * d)[k4];
      kv[(k4 * 4 + 0) * BC + c] = val.x;
      kv[(k4 * 4 + 1) * BC + c] = val.y;
      kv[(k4 * 4 + 2) * BC + c] = val.z;
      kv[(k4 * 4 + 3) * BC + c] = val.w;
    }
    __syncthreads();

    // ---- S = Q K^T (already in log2 units) ----
    float s[4][SC];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < SC; ++j)
        s[i][j] = 0.f;

#pragma unroll 8
    for (int k = 0; k < D; ++k) {
      const float4 q = *reinterpret_cast<const float4 *>(&q_t[k * BR + ty * 4]);
      float kk[SC];
#pragma unroll
      for (int j = 0; j < SC; ++j)
        kk[j] = kv[k * BC + tx * SC + j];
#pragma unroll
      for (int j = 0; j < SC; ++j) {
        s[0][j] = fmaf(q.x, kk[j], s[0][j]);
        s[1][j] = fmaf(q.y, kk[j], s[1][j]);
        s[2][j] = fmaf(q.z, kk[j], s[2][j]);
        s[3][j] = fmaf(q.w, kk[j], s[3][j]);
      }
    }

    // ---- online softmax ----
    float alpha[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float tile_max = -INFINITY;
#pragma unroll
      for (int j = 0; j < SC; ++j) {
        if (key0 + tx * SC + j >= S)
          s[i][j] = -INFINITY;
        tile_max = fmaxf(tile_max, s[i][j]);
      }
#pragma unroll
      for (int off = NTX / 2; off > 0; off >>= 1)
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, off));
      // Every tile holds at least one valid key, so new_max is finite.
      const float new_max = fmaxf(row_max[i], tile_max);
      alpha[i] = exp2f(row_max[i] - new_max);
      row_max[i] = new_max;
      float sum = 0.f;
#pragma unroll
      for (int j = 0; j < SC; ++j) {
        s[i][j] = exp2f(s[i][j] - new_max);
        sum += s[i][j];
      }
      row_sum[i] = row_sum[i] * alpha[i] + sum;
    }
#pragma unroll
    for (int j = 0; j < SC; ++j)
      *reinterpret_cast<float4 *>(&p_t[(tx * SC + j) * BR + ty * 4]) =
          make_float4(s[0][j], s[1][j], s[2][j], s[3][j]);
    __syncthreads(); // K tile no longer needed; P visible to all

    // ---- V tile, row-major [BC][D]; rows past S are zero ----
    for (int idx = tid; idx < BC * D / 4; idx += THREADS) {
      const int k4 = idx % (D / 4), c = idx / (D / 4);
      float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
      if (key0 + c < S && k4 < d4)
        val =
            reinterpret_cast<const float4 *>(v_base + size_t(key0 + c) * d)[k4];
      reinterpret_cast<float4 *>(kv)[idx] = val;
    }
    __syncthreads();

    // ---- O = O * alpha + P V ----
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int g = 0; g < OG; ++g)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          acc[i][g][e] *= alpha[i];

#pragma unroll 4
    for (int c = 0; c < BC; ++c) {
      const float4 p = *reinterpret_cast<const float4 *>(&p_t[c * BR + ty * 4]);
#pragma unroll
      for (int g = 0; g < OG; ++g) {
        const float4 v = *reinterpret_cast<const float4 *>(
            &kv[c * D + g * NTX * 4 + tx * 4]);
        const float pv[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          acc[i][g][0] = fmaf(pv[i], v.x, acc[i][g][0]);
          acc[i][g][1] = fmaf(pv[i], v.y, acc[i][g][1]);
          acc[i][g][2] = fmaf(pv[i], v.z, acc[i][g][2]);
          acc[i][g][3] = fmaf(pv[i], v.w, acc[i][g][3]);
        }
      }
    }
    __syncthreads(); // before the next K tile overwrites kv / p_t
  }

  // ---- finalize ----
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float sum = row_sum[i];
#pragma unroll
    for (int off = NTX / 2; off > 0; off >>= 1)
      sum += __shfl_xor_sync(0xffffffff, sum, off);
    const float inv = 1.f / sum;
    const int r = row0 + ty * 4 + i;
    if (r >= M)
      continue;
    float4 *o_row = reinterpret_cast<float4 *>(o_base + size_t(r) * d);
#pragma unroll
    for (int g = 0; g < OG; ++g) {
      const int k4 = g * NTX + tx;
      if (k4 < d4)
        o_row[k4] = make_float4(acc[i][g][0] * inv, acc[i][g][1] * inv,
                                acc[i][g][2] * inv, acc[i][g][3] * inv);
    }
  }
}

template <int D, int BR, int BC, int THREADS>
static void launch_gqa(const float *Q, const float *K, const float *V, float *O,
                       int num_q_heads, int num_kv_heads, int S, int d) {
  using Cfg = GqaConfig<D, BR, BC, THREADS>;
  auto kernel = gqa_flash_kernel<D, BR, BC, THREADS>;
  const int smem = Cfg::SMEM_FLOATS * int(sizeof(float));
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem);
  const int group = num_q_heads / num_kv_heads;
  const int M = group * S;
  const float q_scale =
      1.4426950408889634f / sqrtf(float(d)); // log2(e)/sqrt(d)
  dim3 grid((M + BR - 1) / BR, num_kv_heads);
  kernel<<<grid, THREADS, smem>>>(Q, K, V, O, group, S, d, q_scale);
}

// Q, K, V, output are device pointers; head_dim is a multiple of 4.
extern "C" void solve(const float *Q, const float *K, const float *V,
                      float *output, int num_q_heads, int num_kv_heads,
                      int seq_len, int head_dim) {
  if (head_dim <= 64)
    launch_gqa<64, 64, 32, 128>(Q, K, V, output, num_q_heads, num_kv_heads,
                                seq_len, head_dim);
  else if (head_dim <= 128)
    launch_gqa<128, 64, 32, 128>(Q, K, V, output, num_q_heads, num_kv_heads,
                                 seq_len, head_dim);
  else
    launch_gqa<256, 32, 16, 128>(Q, K, V, output, num_q_heads, num_kv_heads,
                                 seq_len, head_dim);
}
