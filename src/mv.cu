#include <cuda_runtime.h>

template <typename T> __device__ inline T warp_sum(T val) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, off);
  }
  return val;
}

template <size_t BLOCK_SIZE, typename T> __device__ inline T block_sum(T val) {
  constexpr size_t NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ T warp_sums[NUM_WARPS];
  size_t lane = threadIdx.x % 32;
  size_t warp_idx = threadIdx.x / 32;
  val = warp_sum(val);
  if (lane == 0)
    warp_sums[warp_idx] = val;

  __syncthreads();

  if (warp_idx == 0) {
    val = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
    val = warp_sum(val);
  }
  return val;
}

template <int BLOCK_SIZE, typename T1, typename T2, typename T3>
__global__ void mv_one_block_per_row(const T1 *__restrict__ A, // (M, N)
                                     const T2 *__restrict__ B, // (N,)
                                     T3 *__restrict__ C,       // (M,)
                                     size_t M, size_t N) {
  const size_t row = blockIdx.x;
  T3 sum = 0.0f;

  // 同一个 block 的线程分担这一行的列
  for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
    sum += A[row * N + i] * B[i];
  }

  sum = block_sum<BLOCK_SIZE>(sum);

  if (threadIdx.x == 0) {
    C[row] = sum;
  }
}

template <typename T1, typename T2, typename T3>
__global__ void mv_one_warp_per_row(const T1 *__restrict__ A, // (M, N)
                                    const T2 *__restrict__ B, // (N,)
                                    T3 *__restrict__ C,       // (M,)
                                    size_t M, size_t N) {
  const size_t lane = threadIdx.x & 31;
  const size_t warp = threadIdx.x >> 5;
  const size_t warps_per_block = blockDim.x / 32;
  const size_t row = blockIdx.x * warps_per_block + warp;

  // 同一个 warp 的 row 相同，因此整个 warp 一起退出
  if (row >= M) {
    return;
  }

  T3 sum = 0.0f;
  for (int i = lane; i < N; i += 32) {
    sum += A[row * N + i] * B[i];
  }

  sum = warp_sum(sum);

  if (lane == 0) {
    C[row] = sum;
  }
}

void mv(const float *A, // (M, N)
        const float *x, // (N,)
        float *y,       // (M,)
        int M, int N, int nnz) {
  if (M <= 0) {
    return;
  }
  // A800 measurements: short rows favor warp-only reduction. With fewer
  // rows, a whole block per row pays off at a smaller column count.
  const int block_min_columns = M <= 64 ? 1024 : (M <= 1024 ? 2048 : 4096);
  if (N >= block_min_columns) {
    mv_one_block_per_row<256><<<M, 256>>>(A, x, y, M, N);
  } else {
    mv_one_warp_per_row<<<(M + 7) / 8, 256>>>(A, x, y, M, N);
  }
}

// A, x, y are device pointers
extern "C" void solve(const float *A, // (M, N)
                      const float *x, // (N,)
                      float *y,       // (M,)
                      int M, int N, int nnz) {
  mv(A, x, y, M, N, nnz);
}
