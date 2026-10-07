// Reduce-then-Scan：每行切成若干段（段数 ≈ 同时驻留的 block 数），
// 第一遍只求每段总和，第二遍每段从前面各段的总和得到
// carry，再扫描并直接写出最终结果。 显存流量 3N（读 2N、写 N），2 次 kernel
// 启动，分段总和放在静态 __device__ 数组里。 Op
// 需满足结合律和交换律；静态数组全局共享，不能在多个 stream 上并发调用。

#include <algorithm>
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

// 假设一维 block，线程数是 32 的倍数，所有线程都调用。
// total 非空时写入整个 block 的总和（每个线程都拿到同一个值）。
template <bool Exclusive = false, template <typename> class Op = Add,
          typename T>
__device__ T block_scan(T val, T *total = nullptr) {
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
  if (total != nullptr) {
    *total = warp_sums[num_warps - 1];
  }

  // 支持重复调用时安全复用 shared memory
  __syncthreads();

  return result;
}

constexpr int BLOCK_SIZE = 256;
constexpr int ITEMS = 8; // 每个线程负责的连续元素数
constexpr int TILE = BLOCK_SIZE * ITEMS;
// tile 数不超过它时用单个 block 串行扫完整行，比多启动一个 kernel 更快（A800
// 实测）。
constexpr size_t SERIAL_TILES = 4;
// 所有行的分段总和共用这块静态内存，按字节计，避免为每个 T 单独定义。
constexpr size_t PARTIALS_BYTES = 256 * 1024;
__device__ uint4 g_partials[PARTIALS_BYTES / sizeof(uint4)];

// 每 32 个元素插入一个空位，使 tid * ITEMS + k 的按线程读取避开 bank conflict。
__host__ __device__ constexpr int padded(int i) { return i + i / 32; }

// 扫描一个 tile 并把 carry 加到每个结果上，返回 tile 总和（所有线程一致）。
// 全局内存按 striped 方式合并读写，经 shared memory 转置成每线程连续 ITEMS 个。
template <bool Exclusive, template <typename> class Op, typename T>
__device__ T scan_tile(const T *input, T *output, const int valid,
                       const T carry, T *tile) {
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    const int i = k * BLOCK_SIZE + threadIdx.x;
    tile[padded(i)] = i < valid ? input[i] : Op<T>::identity();
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
  T total;
  const T prefix =
      Op<T>::apply(carry, block_scan<true, Op>(items[ITEMS - 1], &total));

#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    const T local =
        Exclusive ? (k == 0 ? Op<T>::identity() : items[k - 1]) : items[k];
    tile[padded(threadIdx.x * ITEMS + k)] = Op<T>::apply(prefix, local);
  }
  __syncthreads();

#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    const int i = k * BLOCK_SIZE + threadIdx.x;
    if (i < valid) {
      output[i] = tile[padded(i)];
    }
  }
  // 下一个 tile 会覆盖 shared memory。
  __syncthreads();
  return total;
}

// 每个 block 求一段连续 tile 的总和。
// 线程按 striped 顺序累加，所以要求 Op 满足交换律（Add/Mul/Max/Min 都满足）。
template <template <typename> class Op, typename T>
__global__ void __launch_bounds__(BLOCK_SIZE)
    reduce_segments(const T *input, // (B, N,)
                    T *partials,    // (B, segments,)
                    const size_t N, const size_t tiles_per_segment) {
  const size_t row = blockIdx.y * N;
  const size_t begin = blockIdx.x * tiles_per_segment * TILE;
  const size_t end = min(N, begin + tiles_per_segment * TILE);

  T acc = Op<T>::identity();
  for (size_t base = begin; base < end; base += TILE) {
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
      const size_t i = base + k * BLOCK_SIZE + threadIdx.x;
      if (i < end) {
        acc = Op<T>::apply(acc, input[row + i]);
      }
    }
  }

  T total;
  block_scan<false, Op>(acc, &total);
  if (threadIdx.x == 0) {
    partials[blockIdx.y * gridDim.x + blockIdx.x] = total;
  }
}

// 每个 block 依次扫描一段连续 tile。
// partials 是各段总和，前面各段的归约就是这一段的初始 carry。
template <bool Exclusive, template <typename> class Op, typename T>
__global__ void __launch_bounds__(BLOCK_SIZE)
    scan_segments(const T *input,    // (B, N,)
                  T *output,         // (B, N,)
                  const T *partials, // (B, segments,)，nullptr 表示只有一段
                  const size_t N, const size_t tiles_per_segment) {
  __shared__ T tile[padded(TILE)];
  const size_t row = blockIdx.y * N;
  const size_t begin = blockIdx.x * tiles_per_segment * TILE;
  const size_t end = min(N, begin + tiles_per_segment * TILE);

  // 段数不超过同时驻留的 block 数，直接从 L2 读，省掉单独扫描 partials 的
  // kernel。
  T carry = Op<T>::identity();
  if (partials != nullptr && blockIdx.x > 0) {
    const T *row_partials = partials + blockIdx.y * gridDim.x;
    T acc = Op<T>::identity();
    for (int i = threadIdx.x; i < int(blockIdx.x); i += BLOCK_SIZE) {
      acc = Op<T>::apply(acc, row_partials[i]);
    }
    block_scan<false, Op>(acc, &carry);
  }
  for (size_t base = begin; base < end; base += TILE) {
    const int valid = int(min(size_t(TILE), end - base));
    carry = Op<T>::apply(carry, scan_tile<Exclusive, Op>(input + row + base,
                                                         output + row + base,
                                                         valid, carry, tile));
  }
}

template <typename F> int max_active_blocks(F kernel) {
  int device = 0, sms = 0, per_sm = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, BLOCK_SIZE, 0);
  return sms * per_sm;
}

// Op 是运算模板，T 由 input/output 的指针类型推导。
// reduce-then-scan：每行切成若干段，先求每段总和，
// 再让每段带着前面各段的总和重新扫描，共读 2N、写 N。支持 input == output。
template <bool Exclusive = false, template <typename> class Op = Add,
          typename T>
void scan(const T *input, T *output, const size_t B, const size_t N) {
  if (N <= 0 || B <= 0)
    return;

  // 让所有 block 同时驻留即可，更多的段只会增加分段总和的开销。
  static const int resident_blocks =
      max_active_blocks(scan_segments<Exclusive, Op, T>);
  constexpr size_t max_partials = PARTIALS_BYTES / sizeof(T);

  const size_t tiles = (N + TILE - 1) / TILE;
  size_t segments = (resident_blocks + B - 1) / B;
  segments = std::min({segments, tiles, max_partials / B});
  if (tiles <= SERIAL_TILES || segments == 0) {
    segments = 1;
  }
  const size_t tiles_per_segment = (tiles + segments - 1) / segments;
  segments = (tiles + tiles_per_segment - 1) / tiles_per_segment;

  if (segments == 1) {
    scan_segments<Exclusive, Op><<<dim3(1, B), BLOCK_SIZE>>>(
        input, output, static_cast<const T *>(nullptr), N, tiles_per_segment);
    return;
  }

  static T *partials = [] {
    void *p = nullptr;
    cudaGetSymbolAddress(&p, g_partials);
    return static_cast<T *>(p);
  }();

  reduce_segments<Op><<<dim3(segments, B), BLOCK_SIZE>>>(input, partials, N,
                                                         tiles_per_segment);
  scan_segments<Exclusive, Op><<<dim3(segments, B), BLOCK_SIZE>>>(
      input, output, static_cast<const T *>(partials), N, tiles_per_segment);
}

// input, output are device pointers. output[i] = input[0] + ... + input[i]
// (inclusive).
extern "C" void solve(const float *input, float *output, int N) {
  scan(input, output, 1, N);
}
