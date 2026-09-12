#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 假设一维 block，完整 warp 的 32 个线程一起调用。
// Exclusive 是编译期参数：true 为 exclusive，false 为 inclusive。
template <bool Exclusive = false, typename T> __device__ T warp_scan(T val) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    T other = __shfl_up_sync(0xffffffff, val, off);
    if (lane >= off) {
      val += other;
    }
  }
  if (Exclusive) {
    // 移动 inclusive 结果，避免用减法转换带来的额外浮点误差。
    T previous = __shfl_up_sync(0xffffffff, val, 1);
    return lane == 0 ? T(0) : previous;
  }
  return val;
}

// 假设一维 block，线程数是 32 的倍数，所有线程都调用
// Exclusive 是编译期参数：true 为 exclusive，false 为 inclusive。
template <bool Exclusive, typename T> __device__ T block_scan(T val) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int num_warps = blockDim.x / 32;

  __shared__ T warp_sums[32];

  // 1. warp 内 inclusive scan，用来获取 warp 总和
  T inclusive = warp_scan<false>(val);

  if (lane == 31) {
    warp_sums[warp] = inclusive;
  }

  T local_prefix = inclusive;
  if (Exclusive) {
    // 只有 exclusive 需要移动；整个 block 的分支选择一致。
    T previous = __shfl_up_sync(0xffffffff, inclusive, 1);
    local_prefix = lane == 0 ? T(0) : previous;
  }

  __syncthreads();

  // 2. 第一个 warp 扫描所有 warp 的总和
  if (warp == 0) {
    T sum = lane < num_warps ? warp_sums[lane] : T(0);
    T prefix = warp_scan<false>(sum);

    if (lane < num_warps) {
      warp_sums[lane] = prefix;
    }
  }

  __syncthreads();

  // 3. 加上前面所有 warp 的总和
  T offset = warp > 0 ? warp_sums[warp - 1] : T(0);
  T result = local_prefix + offset;

  // 支持重复调用时安全复用 shared memory
  __syncthreads();

  return result;
}

// 每个 block 独立扫描，并记录这个 block 的总和。
template <bool Exclusive>
__global__ void block_scan(const float *input, // (N,)
                           float *output,      // (N,)
                           float *block_sums,  // ([N/BLOCK_SIZE],)
                           int N) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const float val = tid < N ? input[tid] : 0.0f;
  const float prefix_sum = block_scan<Exclusive>(val);
  if (tid < N)
    output[tid] = prefix_sum;
  if (block_sums != nullptr && threadIdx.x == blockDim.x - 1) {
    // block last thread runs
    block_sums[blockIdx.x] = Exclusive ? prefix_sum + val : prefix_sum;
  }
}

__global__ void add_block_offsets(float *output, const float *block_prefixes,
                                  int N) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N && blockIdx.x > 0) {
    // block_prefixes 是 inclusive：前一个位置就是当前 block 的偏移。
    output[tid] += block_prefixes[blockIdx.x - 1];
  }
}

template <bool Exclusive> void scan(const float *input, float *output, int N) {
  if (N <= 0)
    return;

  constexpr int BLOCK_SIZE = 256;
  const int block_num = 1 + (N - 1) / BLOCK_SIZE;

  // 递归终点：一个 block 就能完成扫描。
  if (block_num == 1) {
    block_scan<Exclusive><<<1, BLOCK_SIZE>>>(input, output, nullptr, N);
    return;
  }

  float *block_sums = nullptr;
  cudaMalloc(&block_sums, block_num * sizeof(float));

  block_scan<Exclusive>
      <<<block_num, BLOCK_SIZE>>>(input, output, block_sums, N);

  // block_sums 本身也可能超过一个 block，因此递归做 inclusive scan。
  scan<false>(block_sums, block_sums, block_num);
  add_block_offsets<<<block_num, BLOCK_SIZE>>>(output, block_sums, N);

  cudaFree(block_sums);
}

// input, output are device pointers. output[i] = input[0] + ... + input[i]
// (inclusive).
extern "C" void solve(const float *input, float *output, int N) {
  scan<false>(input, output, N);
}
