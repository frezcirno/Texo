#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <limits>

// 假设一维 block，完整 warp 的 32 个线程一起调用。
template <typename T> struct Add {
  static constexpr T identity = 0;
  __device__ static T apply(T a, T b) { return a + b; }
};
template <typename T> struct Max {
  static constexpr T identity = std::numeric_limits<T>::has_infinity
                                    ? -std::numeric_limits<T>::infinity()
                                    : std::numeric_limits<T>::lowest();
  __device__ static T apply(T a, T b) { return max(a, b); }
};
template <typename T> struct Min {
  static constexpr T identity = std::numeric_limits<T>::has_infinity
                                    ? std::numeric_limits<T>::infinity()
                                    : std::numeric_limits<T>::max();
  __device__ static T apply(T a, T b) { return min(a, b); }
};

// 假设一维 block，完整 warp 的 32 个线程一起调用。
template <bool Exclusive = false, template <typename> class Op = Add,
          typename T>
__device__ inline T warp_scan(T val) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    T other = __shfl_up_sync(0xffffffff, val, off);
    if (lane >= off) {
      val = Op<T>::apply(val, other);
    }
  }
  if (Exclusive) {
    // 移动 inclusive 结果，避免用减法转换带来的额外浮点误差。
    T previous = __shfl_up_sync(0xffffffff, val, 1);
    return lane == 0 ? Op<T>::identity : previous;
  }
  return val;
}

// 假设一维 block，线程数是 32 的倍数，所有线程都调用
template <bool Exclusive = false, template <typename> class Op = Add,
          typename T>
__device__ T block_scan(T val) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int num_warps = blockDim.x / 32;

  __shared__ T warp_sums[32];

  // 1. warp 内 inclusive scan，用来获取 warp 总和
  T inclusive = warp_scan<false, Op>(val);

  if (lane == 31) {
    warp_sums[warp] = inclusive;
  }

  T local_prefix = inclusive;
  if (Exclusive) {
    // 只有 exclusive 需要移动；整个 block 的分支选择一致。
    T previous = __shfl_up_sync(0xffffffff, inclusive, 1);
    local_prefix = lane == 0 ? Op<T>::identity : previous;
  }

  __syncthreads();

  // 2. 第一个 warp 扫描所有 warp 的总和
  if (warp == 0) {
    T sum = lane < num_warps ? warp_sums[lane] : Op<T>::identity;
    T prefix = warp_scan<false, Op>(sum);

    if (lane < num_warps) {
      warp_sums[lane] = prefix;
    }
  }

  __syncthreads();

  // 3. 加上前面所有 warp 的总和
  T offset = warp > 0 ? warp_sums[warp - 1] : Op<T>::identity;
  T result = Op<T>::apply(local_prefix, offset);

  // 支持重复调用时安全复用 shared memory
  __syncthreads();

  return result;
}

// 每个 block 独立扫描，并记录这个 block 的总和。
template <bool Exclusive = false, template <typename> class Op = Add,
          typename T>
__global__ void block_scan(const T *input, // (N,)
                           T *output,      // (N,)
                           T *block_sums,  // ([N/BLOCK_SIZE],)
                           int N) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const T val = tid < N ? input[tid] : Op<T>::identity;
  const T prefix_sum = block_scan<Exclusive, Op>(val);
  if (tid < N) {
    output[tid] = prefix_sum;
  }
  if (block_sums != nullptr && threadIdx.x == blockDim.x - 1) {
    // block last thread runs
    block_sums[blockIdx.x] =
        Exclusive ? Op<T>::apply(prefix_sum, val) : prefix_sum;
  }
}

template <template <typename> class Op = Add, typename T>
__global__ void add_block_offsets(T *output,               // (N,)
                                  const T *block_prefixes, // (N/32,)
                                  int N) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N && blockIdx.x > 0) {
    // block_prefixes 是 inclusive：前一个位置就是当前 block 的偏移。
    output[tid] = Op<T>::apply(output[tid], block_prefixes[blockIdx.x - 1]);
  }
}

// Op 是运算模板，T 由 input/output 的指针类型推导。
template <bool Exclusive = false, template <typename> class Op = Add,
          typename T>
void scan(const T *input, T *output, int N) {
  if (N <= 0)
    return;

  constexpr size_t BLOCK_SIZE = 256;
  const size_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  T *block_sums = nullptr;

  // 递归终点：一个 block 就能完成扫描。
  if (block_num == 1) {
    block_scan<Exclusive, Op><<<1, BLOCK_SIZE>>>(input, output, block_sums, N);
    return;
  }

  cudaMalloc(&block_sums, block_num * sizeof(T));

  block_scan<Exclusive, Op>
      <<<block_num, BLOCK_SIZE>>>(input, output, block_sums, N);

  // block_sums 本身也可能超过一个 block，因此递归做 inclusive scan。
  scan<false, Op>(block_sums, block_sums, block_num);
  add_block_offsets<Op><<<block_num, BLOCK_SIZE>>>(output, block_sums, N);

  cudaFree(block_sums);
}

// input, output are device pointers. output[i] = input[0] + ... + input[i]
// (inclusive).
extern "C" void solve(const float *input, float *output, int N) {
  scan(input, output, N);
}
