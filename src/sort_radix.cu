#include <cuda_runtime.h>

constexpr int THREADS = 256;
constexpr int RADIX = 16; // 每轮处理 4 bit
constexpr int MAX_N = 50000000;
constexpr int MAX_BLOCKS = (MAX_N + THREADS - 1) / THREADS;

__device__ unsigned int g_tmp[MAX_N];
__device__ int g_block_counts[RADIX * MAX_BLOCKS];
__device__ int g_block_offsets[RADIX * MAX_BLOCKS];
__device__ int g_bucket_start[RADIX];

__global__ void count_digits(const unsigned int *in, int N, int shift) {
  __shared__ int counts[RADIX];

  int tid = threadIdx.x;
  if (tid < RADIX)
    counts[tid] = 0;
  __syncthreads();

  int i = blockIdx.x * blockDim.x + tid;
  if (i < N) {
    int digit = (in[i] >> shift) & 15u;
    atomicAdd(&counts[digit], 1);
  }
  __syncthreads();

  if (tid < RADIX) {
    // 布局：[digit][block]
    g_block_counts[tid * gridDim.x + blockIdx.x] = counts[tid];
  }
}

__global__ void make_offsets(int blocks) {
  __shared__ int totals[RADIX];
  int digit = threadIdx.x;

  if (digit < RADIX) {
    int sum = 0;
    for (int b = 0; b < blocks; ++b) {
      g_block_offsets[digit * blocks + b] = sum;
      sum += g_block_counts[digit * blocks + b];
    }
    totals[digit] = sum;
  }
  __syncthreads();

  if (digit == 0) {
    int base = 0;
    for (int d = 0; d < RADIX; ++d) {
      g_bucket_start[d] = base;
      base += totals[d];
    }
  }
}

__global__ void stable_scatter(const unsigned int *in, unsigned int *out, int N,
                               int shift, int blocks) {
  __shared__ int warp_counts[8][RADIX];

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int i = blockIdx.x * blockDim.x + tid;
  bool valid = i < N;

  unsigned int value = valid ? in[i] : 0;
  int digit = (value >> shift) & 15u;
  int rank_in_warp = 0;

  // 对每个 digit 建立该 warp 的计数，并求每个元素在 warp 内的稳定 rank。
  for (int d = 0; d < RADIX; ++d) {
    unsigned mask = __ballot_sync(0xffffffff, valid && digit == d);

    if (lane == 0) {
      warp_counts[warp][d] = __popc(mask);
    }
    if (valid && digit == d) {
      rank_in_warp = __popc(mask & ((1u << lane) - 1));
    }
  }
  __syncthreads();

  if (valid) {
    int rank = rank_in_warp;

    // 加上此前 warp 中相同 digit 的数量。
    for (int w = 0; w < warp; ++w) {
      rank += warp_counts[w][digit];
    }

    int pos = g_bucket_start[digit] +
              g_block_offsets[digit * blocks + blockIdx.x] + rank;
    out[pos] = value;
  }
}

// input、output 均为 device pointer
extern "C" void solve(const unsigned int *input, unsigned int *output, int N) {
  int blocks = (N + THREADS - 1) / THREADS;

  // tmp 要在 host 端参与 ping-pong，需取其设备地址（不分配内存）
  unsigned int *tmp;
  cudaGetSymbolAddress((void **)&tmp, g_tmp);

  const unsigned int *src = input;
  unsigned int *dst = tmp;

  for (int shift = 0; shift < 32; shift += 4) {
    count_digits<<<blocks, THREADS>>>(src, N, shift);
    make_offsets<<<1, RADIX>>>(blocks);
    stable_scatter<<<blocks, THREADS>>>(src, dst, N, shift, blocks);

    src = dst;
    dst = (dst == tmp) ? output : tmp;
  }
}
