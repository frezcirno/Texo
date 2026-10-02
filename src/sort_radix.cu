#include <cuda_runtime.h>

constexpr int THREADS = 256;
constexpr int RADIX = 16; // 每轮处理 4 bit

__global__ void count_digits(const unsigned int *in, int *block_counts, int N,
                             int shift) {
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
    block_counts[tid * gridDim.x + blockIdx.x] = counts[tid];
  }
}

__global__ void make_offsets(const int *block_counts, int *block_offsets,
                             int *bucket_start, int blocks) {
  __shared__ int totals[RADIX];
  int digit = threadIdx.x;

  if (digit < RADIX) {
    int sum = 0;
    for (int b = 0; b < blocks; ++b) {
      block_offsets[digit * blocks + b] = sum;
      sum += block_counts[digit * blocks + b];
    }
    totals[digit] = sum;
  }
  __syncthreads();

  if (digit == 0) {
    int base = 0;
    for (int d = 0; d < RADIX; ++d) {
      bucket_start[d] = base;
      base += totals[d];
    }
  }
}

__global__ void stable_scatter(const unsigned int *in, unsigned int *out,
                               const int *block_offsets,
                               const int *bucket_start, int N, int shift,
                               int blocks) {
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

    int pos =
        bucket_start[digit] + block_offsets[digit * blocks + blockIdx.x] + rank;
    out[pos] = value;
  }
}

// input、output 均为 device pointer
extern "C" void solve(const unsigned int *input, unsigned int *output, int N) {
  int blocks = (N + THREADS - 1) / THREADS;

  unsigned int *tmp;
  int *block_counts;
  int *block_offsets;
  int *bucket_start;

  cudaMalloc(&tmp, N * sizeof(unsigned int));
  cudaMalloc(&block_counts, RADIX * blocks * sizeof(int));
  cudaMalloc(&block_offsets, RADIX * blocks * sizeof(int));
  cudaMalloc(&bucket_start, RADIX * sizeof(int));

  const unsigned int *src = input;
  unsigned int *dst = tmp;

  for (int shift = 0; shift < 32; shift += 4) {
    count_digits<<<blocks, THREADS>>>(src, block_counts, N, shift);
    make_offsets<<<1, RADIX>>>(block_counts, block_offsets, bucket_start,
                               blocks);
    stable_scatter<<<blocks, THREADS>>>(src, dst, block_offsets, bucket_start,
                                        N, shift, blocks);

    src = dst;
    dst = (dst == tmp) ? output : tmp;
  }

  // 8 轮后结果恰好在 output。
  cudaFree(tmp);
  cudaFree(block_counts);
  cudaFree(block_offsets);
  cudaFree(bucket_start);
}
