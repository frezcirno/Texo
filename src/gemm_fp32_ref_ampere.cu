#include <cstdint>
#include <cuda_runtime.h>

namespace {

constexpr int WARP_SIZE = 32;
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 8;
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 16;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int WARP_TILES_M = 4;
constexpr int WARP_TILES_N = 4;
constexpr int STAGES = 2;
constexpr int B_STRIDE = BLOCK_N;
constexpr int A_STAGE_FLOATS = BLOCK_M * BLOCK_K;
constexpr int B_STAGE_FLOATS = BLOCK_K * B_STRIDE;
constexpr int STAGE_FLOATS = A_STAGE_FLOATS + B_STAGE_FLOATS;
constexpr int SMEM_BYTES =
    STAGES * STAGE_FLOATS * static_cast<int>(sizeof(float));

__device__ __forceinline__ void cp_async_16(void *smem_ptr,
                                            const void *gmem_ptr) {
  const unsigned smem_addr =
      static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr),
               "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_group_0() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ uint32_t float_to_tf32(float x) {
  uint32_t y;
  asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ void
mma_m16n8k8_tf32(const uint32_t a[4], const uint32_t b[2], float d[4]) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                 "r"(b[1]));
}

__device__ __forceinline__ int a_smem_index(int row, int col) {
  const int group = col >> 2;
  const int within = col & 3;
  const int swizzled_group = group ^ ((row >> 1) & ((BLOCK_K / 4) - 1));
  return row * BLOCK_K + swizzled_group * 4 + within;
}

__device__ __forceinline__ int b_smem_index(int row, int col) {
  const int group = col >> 2;
  const int within = col & 3;
  const int pair = group >> 1;
  const int swizzled_pair = (pair + (row & 3)) & 3;
  const int swizzled_group = (group & ~7) + swizzled_pair * 2 + (group & 1);
  return row * B_STRIDE + swizzled_group * 4 + within;
}

__device__ __forceinline__ void
load_a_fastfp32_fragment(const float *smem, int row_offset, int k_offset,
                         int lane_id, uint32_t a0[4], uint32_t a1[4]) {
  const int lane_quad = lane_id & 3;
  const int lane_group = lane_id >> 2;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int packed =
        lane_quad * 16 + lane_group + (r & 1) * 8 + (r >> 1) * 64;
    const int local_m = packed & 15;
    const int local_k = packed >> 4;
    const float value =
        smem[a_smem_index(row_offset + local_m, k_offset + local_k)];
    a0[r] = float_to_tf32(value);
    a1[r] = float_to_tf32(value - __uint_as_float(a0[r]));
  }
}

__device__ __forceinline__ void
load_b_fastfp32_fragment(const float *smem, int k_offset, int col_offset,
                         int lane_id, uint32_t b0[2], uint32_t b1[2]) {
  const int lane_quad = lane_id & 3;
  const int lane_group = lane_id >> 2;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int packed = lane_quad * 8 + lane_group + r * 32;
    const int local_k = packed >> 3;
    const int local_n = packed & 7;
    const float value =
        smem[b_smem_index(k_offset + local_k, col_offset + local_n)];
    b0[r] = float_to_tf32(value);
    b1[r] = float_to_tf32(value - __uint_as_float(b0[r]));
  }
}

__device__ __forceinline__ void
cp_async_stage_full(const float *__restrict__ A, const float *__restrict__ B,
                    float *smem_stage, int K_gemm, int N_gemm, int block_m,
                    int block_n, int k0) {
  constexpr int ATasks = (BLOCK_M * BLOCK_K) / 4;
  constexpr int BTasks = BLOCK_K * (BLOCK_N / 4);
  constexpr int MaxTasks = (ATasks > BTasks) ? ATasks : BTasks;
  constexpr int Iters = (MaxTasks + (WARPS_PER_BLOCK * WARP_SIZE) - 1) /
                        (WARPS_PER_BLOCK * WARP_SIZE);

  float *As = smem_stage;
  float *Bs = smem_stage + A_STAGE_FLOATS;

#pragma unroll
  for (int i = 0; i < Iters; ++i) {
    const int task = threadIdx.x + i * blockDim.x;
    if (task < ATasks) {
      const int a_row = task / (BLOCK_K / 4);
      const int a_group = task - a_row * (BLOCK_K / 4);
      const int a_store_group = a_group ^ ((a_row >> 1) & ((BLOCK_K / 4) - 1));
      const float *src = A + (block_m + a_row) * K_gemm + k0 + a_group * 4;
      float *dst = As + a_row * BLOCK_K + a_store_group * 4;
      cp_async_16(dst, src);
    }
    if (task < BTasks) {
      const int b_row = task / (BLOCK_N / 4);
      const int b_group = task - b_row * (BLOCK_N / 4);
      const float *src = B + (k0 + b_row) * N_gemm + block_n + b_group * 4;
      float *dst = Bs + b_smem_index(b_row, b_group * 4);
      cp_async_16(dst, src);
    }
  }
}

__device__ __forceinline__ void
load_stage_padded(const float *__restrict__ A, const float *__restrict__ B,
                  float *smem_stage, int M_gemm, int N_gemm, int K_gemm,
                  int block_m, int block_n, int k0) {
  constexpr int ATasks = (BLOCK_M * BLOCK_K) / 4;
  constexpr int BTasks = BLOCK_K * (BLOCK_N / 4);
  constexpr int MaxTasks = (ATasks > BTasks) ? ATasks : BTasks;
  constexpr int Iters = (MaxTasks + (WARPS_PER_BLOCK * WARP_SIZE) - 1) /
                        (WARPS_PER_BLOCK * WARP_SIZE);

  float *As = smem_stage;
  float *Bs = smem_stage + A_STAGE_FLOATS;

#pragma unroll
  for (int i = 0; i < Iters; ++i) {
    const int task = threadIdx.x + i * blockDim.x;
    if (task < ATasks) {
      const int a_row = task / (BLOCK_K / 4);
      const int a_group = task - a_row * (BLOCK_K / 4);
      const int a_store_group = a_group ^ ((a_row >> 1) & ((BLOCK_K / 4) - 1));
      const int global_m = block_m + a_row;
      const int global_k = k0 + a_group * 4;
      float *dst = As + a_row * BLOCK_K + a_store_group * 4;
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const int kk = global_k + v;
        dst[v] = (global_m < M_gemm && kk < K_gemm) ? A[global_m * K_gemm + kk]
                                                    : 0.0f;
      }
    }
    if (task < BTasks) {
      const int b_row = task / (BLOCK_N / 4);
      const int b_group = task - b_row * (BLOCK_N / 4);
      const int global_k = k0 + b_row;
      const int global_n = block_n + b_group * 4;
      float *dst = Bs + b_smem_index(b_row, b_group * 4);
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const int nn = global_n + v;
        dst[v] = (global_k < K_gemm && nn < N_gemm) ? B[global_k * N_gemm + nn]
                                                    : 0.0f;
      }
    }
  }
}

// Fast path: M, N and K are exact multiples of BLOCK_M, BLOCK_N and BLOCK_K.
// There are no boundary checks. The next K tile is prefetched while the
// current K tile is being computed.
__global__ __launch_bounds__(
    WARPS_PER_BLOCK *WARP_SIZE,
    1) void matmul_mma_fastfp32_128x128_4x4warp_kernel(const float
                                                           *__restrict__ A,
                                                       const float
                                                           *__restrict__ B,
                                                       float *__restrict__ C,
                                                       int M_gemm, int N_gemm,
                                                       int K_gemm) {
  extern __shared__ __align__(16) float smem[];

  constexpr int WarpTileM = WARP_TILES_M * MMA_M;
  constexpr int WarpTileN = WARP_TILES_N * MMA_N;
  constexpr int WarpsN = BLOCK_N / WarpTileN; // 4

  const int block_m = blockIdx.y * BLOCK_M;
  const int block_n = blockIdx.x * BLOCK_N;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x & (WARP_SIZE - 1);

  const int warp_m = warp_id / WarpsN;
  const int warp_n = warp_id - warp_m * WarpsN;
  const int warp_row = warp_m * WarpTileM;
  const int warp_col = warp_n * WarpTileN;

  float acc[WARP_TILES_M][WARP_TILES_N][4] = {};

  const int k_tiles = (K_gemm + BLOCK_K - 1) / BLOCK_K;

  // Prime the two-stage pipeline with K tile 0.
  cp_async_stage_full(A, B, smem, K_gemm, N_gemm, block_m, block_n, 0);
  cp_async_commit();

  for (int tile = 0; tile < k_tiles; ++tile) {
    const int stage = tile & 1;
    float *smem_stage = smem + stage * STAGE_FLOATS;

    // The current stage must be ready before any thread reads it.
    cp_async_wait_group_0();
    __syncthreads();

    // Start loading the next stage. Its copy overlaps the MMA work below.
    const int future = tile + 1;
    if (future < k_tiles) {
      cp_async_stage_full(A, B, smem + ((future & 1) * STAGE_FLOATS), K_gemm,
                          N_gemm, block_m, block_n, future * BLOCK_K);
      cp_async_commit();
    }

    const float *As = smem_stage;
    const float *Bs = smem_stage + A_STAGE_FLOATS;

#pragma unroll
    for (int kk = 0; kk < BLOCK_K; kk += MMA_K) {
      uint32_t a0_frag[WARP_TILES_M][4];
      uint32_t a1_frag[WARP_TILES_M][4];
      uint32_t b0_frag[WARP_TILES_N][2];
      uint32_t b1_frag[WARP_TILES_N][2];

#pragma unroll
      for (int mi = 0; mi < WARP_TILES_M; ++mi) {
        load_a_fastfp32_fragment(As, warp_row + mi * MMA_M, kk, lane_id,
                                 a0_frag[mi], a1_frag[mi]);
      }

#pragma unroll
      for (int ni = 0; ni < WARP_TILES_N; ++ni) {
        load_b_fastfp32_fragment(Bs, kk, warp_col + ni * MMA_N, lane_id,
                                 b0_frag[ni], b1_frag[ni]);
      }

#pragma unroll
      for (int mi = 0; mi < WARP_TILES_M; ++mi) {
#pragma unroll
        for (int ni = 0; ni < WARP_TILES_N; ++ni) {
          mma_m16n8k8_tf32(a0_frag[mi], b0_frag[ni], acc[mi][ni]);
          mma_m16n8k8_tf32(a0_frag[mi], b1_frag[ni], acc[mi][ni]);
          mma_m16n8k8_tf32(a1_frag[mi], b0_frag[ni], acc[mi][ni]);
        }
      }
    }
    __syncthreads();
  }

  const int lane_quad = lane_id & 3;
  const int lane_group = lane_id >> 2;
#pragma unroll
  for (int mi = 0; mi < WARP_TILES_M; ++mi) {
#pragma unroll
    for (int ni = 0; ni < WARP_TILES_N; ++ni) {
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const int packed =
            lane_quad * 32 + lane_group + (v & 1) * 16 + (v >> 1) * 8;
        const int local_m = packed & 15;
        const int local_n = packed >> 4;
        const int global_m = block_m + warp_row + mi * MMA_M + local_m;
        const int global_n = block_n + warp_col + ni * MMA_N + local_n;
        C[global_m * N_gemm + global_n] = acc[mi][ni][v];
      }
    }
  }
}

// General path: edge tiles are allowed. Out-of-range A/B elements are filled
// with zero in shared memory, and out-of-range C elements are not stored.
__global__ __launch_bounds__(
    WARPS_PER_BLOCK *WARP_SIZE,
    1) void matmul_mma_fastfp32_128x128_4x4warp_padded_kernel(const float
                                                                  *__restrict__ A,
                                                              const float
                                                                  *__restrict__ B,
                                                              float
                                                                  *__restrict__ C,
                                                              int M_gemm,
                                                              int N_gemm,
                                                              int K_gemm) {
  extern __shared__ __align__(16) float smem[];

  constexpr int WarpTileM = WARP_TILES_M * MMA_M;
  constexpr int WarpTileN = WARP_TILES_N * MMA_N;
  constexpr int WarpsN = BLOCK_N / WarpTileN;

  const int block_m = blockIdx.y * BLOCK_M;
  const int block_n = blockIdx.x * BLOCK_N;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x & (WARP_SIZE - 1);

  const int warp_m = warp_id / WarpsN;
  const int warp_n = warp_id - warp_m * WarpsN;
  const int warp_row = warp_m * WarpTileM;
  const int warp_col = warp_n * WarpTileN;

  float acc[WARP_TILES_M][WARP_TILES_N][4] = {};

  const int k_tiles = (K_gemm + BLOCK_K - 1) / BLOCK_K;

  for (int tile = 0; tile < k_tiles; ++tile) {
    const int stage = tile & 1;
    float *smem_stage = smem + stage * STAGE_FLOATS;
    const int k0 = tile * BLOCK_K;

    // Interior tiles can still use a 16-byte cp.async load. Edge tiles use
    // guarded scalar loads and write zero for every out-of-range element.
    const bool stage_full =
        (block_m + BLOCK_M <= M_gemm) && (block_n + BLOCK_N <= N_gemm) &&
        (k0 + BLOCK_K <= K_gemm) && ((K_gemm & 3) == 0) && ((N_gemm & 3) == 0);
    if (stage_full) {
      cp_async_stage_full(A, B, smem_stage, K_gemm, N_gemm, block_m, block_n,
                          k0);
      cp_async_commit();
      cp_async_wait_group_0();
    } else {
      load_stage_padded(A, B, smem_stage, M_gemm, N_gemm, K_gemm, block_m,
                        block_n, k0);
    }
    __syncthreads();

    const float *As = smem_stage;
    const float *Bs = smem_stage + A_STAGE_FLOATS;

#pragma unroll
    for (int kk = 0; kk < BLOCK_K; kk += MMA_K) {
      uint32_t a0_frag[WARP_TILES_M][4];
      uint32_t a1_frag[WARP_TILES_M][4];
      uint32_t b0_frag[WARP_TILES_N][2];
      uint32_t b1_frag[WARP_TILES_N][2];

#pragma unroll
      for (int mi = 0; mi < WARP_TILES_M; ++mi) {
        load_a_fastfp32_fragment(As, warp_row + mi * MMA_M, kk, lane_id,
                                 a0_frag[mi], a1_frag[mi]);
      }

#pragma unroll
      for (int ni = 0; ni < WARP_TILES_N; ++ni) {
        load_b_fastfp32_fragment(Bs, kk, warp_col + ni * MMA_N, lane_id,
                                 b0_frag[ni], b1_frag[ni]);
      }

#pragma unroll
      for (int mi = 0; mi < WARP_TILES_M; ++mi) {
#pragma unroll
        for (int ni = 0; ni < WARP_TILES_N; ++ni) {
          mma_m16n8k8_tf32(a0_frag[mi], b0_frag[ni], acc[mi][ni]);
          mma_m16n8k8_tf32(a0_frag[mi], b1_frag[ni], acc[mi][ni]);
          mma_m16n8k8_tf32(a1_frag[mi], b0_frag[ni], acc[mi][ni]);
        }
      }
    }
    __syncthreads();
  }

  const int lane_quad = lane_id & 3;
  const int lane_group = lane_id >> 2;
#pragma unroll
  for (int mi = 0; mi < WARP_TILES_M; ++mi) {
#pragma unroll
    for (int ni = 0; ni < WARP_TILES_N; ++ni) {
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const int packed =
            lane_quad * 32 + lane_group + (v & 1) * 16 + (v >> 1) * 8;
        const int local_m = packed & 15;
        const int local_n = packed >> 4;
        const int global_m = block_m + warp_row + mi * MMA_M + local_m;
        const int global_n = block_n + warp_col + ni * MMA_N + local_n;
        if (global_m < M_gemm && global_n < N_gemm) {
          C[global_m * N_gemm + global_n] = acc[mi][ni][v];
        }
      }
    }
  }
}

} // namespace

// Platform signature: A is M x N, B is N x K, C is M x K.
extern "C" void solve(const float *A, const float *B, float *C, int M, int N,
                      int K) {
  const int M_gemm = M;
  const int K_gemm = N;
  const int N_gemm = K;

  if ((M_gemm % BLOCK_M == 0) && (N_gemm % BLOCK_N == 0) &&
      (K_gemm % BLOCK_K == 0)) {
    // Full path: M/N/K exactly cover complete 128 x 128 x 16 tiles.
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE); // 8 warps = 256 threads
    dim3 grid(N_gemm / BLOCK_N, M_gemm / BLOCK_M);

    cudaFuncSetAttribute(matmul_mma_fastfp32_128x128_4x4warp_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SMEM_BYTES);
    cudaFuncSetCacheConfig(matmul_mma_fastfp32_128x128_4x4warp_kernel,
                           cudaFuncCachePreferShared);

    matmul_mma_fastfp32_128x128_4x4warp_kernel<<<grid, block, SMEM_BYTES>>>(
        A, B, C, M_gemm, N_gemm, K_gemm);
  } else {
    // Padded path: round the grid up because the last M/N tiles may be partial.
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE); // 8 warps = 256 threads
    dim3 grid((N_gemm + BLOCK_N - 1) / BLOCK_N,
              (M_gemm + BLOCK_M - 1) / BLOCK_M);

    cudaFuncSetAttribute(matmul_mma_fastfp32_128x128_4x4warp_padded_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SMEM_BYTES);
    cudaFuncSetCacheConfig(matmul_mma_fastfp32_128x128_4x4warp_padded_kernel,
                           cudaFuncCachePreferShared);

    matmul_mma_fastfp32_128x128_4x4warp_padded_kernel<<<grid, block,
                                                        SMEM_BYTES>>>(
        A, B, C, M_gemm, N_gemm, K_gemm);
  }
}
