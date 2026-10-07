// Scan-then-Propagate：每个 block 扫描一个 tile 并写出局部结果和 block 总和，
// 递归扫描 block 总和，再给每个 block 的输出加上前一个 block 的前缀。
// 显存流量 4N（读 2N、写 2N），每层递归 2 次 kernel 启动，临时内存用
// cudaMallocAsync。 Op 只需满足结合律。

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <limits>

// 假设一维 block，完整 warp 的 32 个线程一起调用。
template <typename T> struct Add {
  __host__ __device__ static constexpr T identity() { return T(0); }
  __device__ static T apply(T a, T b) { return a + b; }
};
template <typename T> struct Mul {
  __host__ __device__ static constexpr T identity() { return T(1); }
  __device__ static T apply(T a, T b) { return a * b; }
};
template <typename T> struct Max {
  __host__ __device__ static constexpr T identity() {
    return std::numeric_limits<T>::has_infinity
               ? -std::numeric_limits<T>::infinity()
               : std::numeric_limits<T>::lowest();
  }
  __device__ static T apply(T a, T b) { return max(a, b); }
};
template <typename T> struct Min {
  __host__ __device__ static constexpr T identity() {
    return std::numeric_limits<T>::has_infinity
               ? std::numeric_limits<T>::infinity()
               : std::numeric_limits<T>::max();
  }
  __device__ static T apply(T a, T b) { return min(a, b); }
};

// 假设一维 block，完整 warp 的 32 个线程一起调用。
template <bool Exclusive = false, template <typename> class Op = Add,
          typename T>
__device__ inline T warp_scan(T val) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    T other = __shfl_up_sync(0xffffffff, val, off);
    if (lane >= off) {
      val = Op<T>::apply(other, val);
    }
  }
  if (Exclusive) {
    // 移动 inclusive 结果，避免用减法转换带来的额外浮点误差。
    T previous = __shfl_up_sync(0xffffffff, val, 1);
    return lane == 0 ? Op<T>::identity() : previous;
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
    local_prefix = lane == 0 ? Op<T>::identity() : previous;
  }

  __syncthreads();

  // 2. 第一个 warp 扫描所有 warp 的总和
  if (warp == 0) {
    T sum = lane < num_warps ? warp_sums[lane] : Op<T>::identity();
    T prefix = warp_scan<false, Op>(sum);

    if (lane < num_warps) {
      warp_sums[lane] = prefix;
    }
  }

  __syncthreads();

  // 3. 加上前面所有 warp 的总和
  T offset = warp > 0 ? warp_sums[warp - 1] : Op<T>::identity();
  T result = Op<T>::apply(offset, local_prefix);

  // 支持重复调用时安全复用 shared memory
  __syncthreads();

  return result;
}

constexpr int BLOCK_SIZE = 256;
constexpr int ITEMS = 8; // 每个线程负责的连续元素数
constexpr int TILE = BLOCK_SIZE * ITEMS;

// 每 32 个元素插入一个空位，使 tid * ITEMS + k 的按线程读取避开 bank conflict。
__host__ __device__ constexpr int padded(int i) { return i + i / 32; }

// 每个 block 独立扫描 TILE 个元素，并记录这个 block 的总和。
// 全局内存按 striped 方式合并读写，经 shared memory 转置成每线程连续 ITEMS 个。
template <bool Exclusive = false, template <typename> class Op = Add,
          typename T>
__global__ void __launch_bounds__(BLOCK_SIZE)
    block_scan(const T *input, // (B, N,)
               T *output,      // (B, N,)
               T *block_sums,  // (B, block_num,)
               const size_t block_num, const size_t N) {
  __shared__ T tile[padded(TILE)];
  const size_t base = blockIdx.y * N + blockIdx.x * size_t(TILE);
  const int valid = int(min(size_t(TILE), N - blockIdx.x * size_t(TILE)));

#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    const int i = k * BLOCK_SIZE + threadIdx.x;
    tile[padded(i)] = i < valid ? input[base + i] : Op<T>::identity();
  }
  __syncthreads();

  // 线程内对连续 ITEMS 个元素做 inclusive scan。
  T items[ITEMS];
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    items[k] = tile[padded(threadIdx.x * ITEMS + k)];
  }
#pragma unroll
  for (int k = 1; k < ITEMS; ++k) {
    items[k] = Op<T>::apply(items[k - 1], items[k]);
  }

  // 只对每个线程的总和做 block scan；block_scan 结尾的同步保证 tile 可以复用。
  const T thread_prefix = block_scan<true, Op>(items[ITEMS - 1]);

#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    const T local =
        Exclusive ? (k == 0 ? Op<T>::identity() : items[k - 1]) : items[k];
    tile[padded(threadIdx.x * ITEMS + k)] = Op<T>::apply(thread_prefix, local);
  }
  if (block_sums != nullptr && threadIdx.x == BLOCK_SIZE - 1) {
    block_sums[blockIdx.y * block_num + blockIdx.x] =
        Op<T>::apply(thread_prefix, items[ITEMS - 1]);
  }
  __syncthreads();

#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    const int i = k * BLOCK_SIZE + threadIdx.x;
    if (i < valid) {
      output[base + i] = tile[padded(i)];
    }
  }
}

// 第 0 个 block 不需要偏移，所以 grid 从第 1 个 block 开始。
template <template <typename> class Op = Add, typename T>
__global__ void __launch_bounds__(BLOCK_SIZE)
    add_block_offsets(T *output,               // (B, N,)
                      const T *block_prefixes, // (B, block_num,)
                      const size_t N, const size_t block_num) {
  const size_t block = blockIdx.x + 1;
  const size_t base = blockIdx.y * N + block * TILE;
  const int valid = int(min(size_t(TILE), N - block * TILE));
  // block_prefixes 是 inclusive：前一个位置就是当前 block 的偏移。
  const T offset = block_prefixes[blockIdx.y * block_num + block - 1];
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    const int i = k * BLOCK_SIZE + threadIdx.x;
    if (i < valid) {
      output[base + i] = Op<T>::apply(offset, output[base + i]);
    }
  }
}

// Op 是运算模板，T 由 input/output 的指针类型推导。
template <bool Exclusive = false, template <typename> class Op = Add,
          typename T>
void scan(const T *input, T *output, const size_t B, const size_t N) {
  if (N <= 0)
    return;

  const size_t block_num = (N + TILE - 1) / TILE;

  // 递归终点：一个 block 就能完成扫描。
  if (block_num == 1) {
    block_scan<Exclusive, Op, T>
        <<<dim3(1, B), BLOCK_SIZE>>>(input, output, nullptr, block_num, N);
    return;
  }

  T *block_sums = nullptr;
  cudaMallocAsync(&block_sums, B * block_num * sizeof(T), 0);

  block_scan<Exclusive, Op><<<dim3(block_num, B), BLOCK_SIZE>>>(
      input, output, block_sums, block_num, N);

  // block_sums 本身也可能超过一个 block，因此递归做 inclusive scan。
  scan<false, Op>(block_sums, block_sums, B, block_num);
  add_block_offsets<Op><<<dim3(block_num - 1, B), BLOCK_SIZE>>>(
      output, block_sums, N, block_num);

  cudaFreeAsync(block_sums, 0);
}

// input, output are device pointers. output[i] = input[0] + ... + input[i]
// (inclusive).
extern "C" void solve(const float *input, float *output, int N) {
  scan(input, output, 1, N);
}
