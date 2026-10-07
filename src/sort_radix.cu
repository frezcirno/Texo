#include <cuda_runtime.h>

constexpr int THREADS = 256;
constexpr int RADIX = 16; // 每轮处理 4 bit
constexpr int MAX_N = 50000000;
constexpr int MAX_BLOCKS = (MAX_N + THREADS - 1) / THREADS;
constexpr int MAX_COUNTS = RADIX * MAX_BLOCKS;

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

constexpr int BLOCK_SIZE = 128;
constexpr int ITEMS =
    16; // 每个线程负责的连续元素数；tile 越大回看次数越少（A800 实测）
constexpr int TILE = BLOCK_SIZE * ITEMS;

// 每 32 个元素插入一个空位，使 tid * ITEMS + k 的按线程读取避开 bank conflict。
__host__ __device__ constexpr int padded(int i) { return i + i / 32; }

// tile 状态字：低 2 位是状态，高位是本次调用的 epoch。
// 上一次调用留下的状态 epoch 不同，自动视为未发布，所以不需要每次清零。
constexpr unsigned STATUS_AGGREGATE = 1; // 只知道本 tile 的总和
constexpr unsigned STATUS_PREFIX = 2; // 已知从行首到本 tile 的 inclusive 前缀
constexpr unsigned MAX_EPOCH = (1u << 30) - 1;

__host__ __device__ constexpr unsigned status_word(unsigned epoch,
                                                   unsigned status) {
  return (epoch << 2) | status;
}

// 每个 tile 的发布状态。不超过 4 字节的 T 把状态字和值打包进一个 64 位字，
// 一次读写同时拿到两者，不需要 fence；更大的 T 先写值、fence、再写状态。
template <typename T, bool Packed = (sizeof(T) <= 4)> struct TileStates;

template <typename T> struct TileStates<T, true> {
  unsigned long long *words =
      nullptr; // (B * tiles_per_row,)，nullptr 表示每行只有一个 tile

  __host__ __device__ bool enabled() const { return words != nullptr; }
  static size_t status_bytes(size_t tiles) {
    return tiles * sizeof(unsigned long long);
  }
  static size_t total_bytes(size_t tiles) { return status_bytes(tiles); }
  void bind(char *buffer, size_t) {
    words = reinterpret_cast<unsigned long long *>(buffer);
  }

  __device__ void publish(size_t tile, T value, unsigned status,
                          unsigned epoch) const {
    unsigned bits = 0;
    memcpy(&bits, &value, sizeof(T));
    *reinterpret_cast<volatile unsigned long long *>(words + tile) =
        (static_cast<unsigned long long>(status_word(epoch, status)) << 32) |
        bits;
  }
  // 等到 tile 在本次调用中发布过，返回状态，value 是对应的总和或前缀。
  __device__ unsigned wait(size_t tile, unsigned epoch, T &value) const {
    unsigned long long word;
    do {
      word = *reinterpret_cast<volatile unsigned long long *>(words + tile);
    } while ((unsigned(word >> 32) >> 2) != epoch);
    const unsigned bits = unsigned(word);
    memcpy(&value, &bits, sizeof(T));
    return unsigned(word >> 32) & 3;
  }
};

template <typename T> struct TileStates<T, false> {
  unsigned *status =
      nullptr; // (B * tiles_per_row,)，nullptr 表示每行只有一个 tile
  T *aggregate = nullptr;
  T *inclusive = nullptr;

  __host__ __device__ bool enabled() const { return status != nullptr; }
  static size_t status_bytes(size_t tiles) {
    return (tiles * sizeof(unsigned) + 15) / 16 * 16;
  }
  static size_t total_bytes(size_t tiles) {
    return status_bytes(tiles) + 2 * tiles * sizeof(T);
  }
  void bind(char *buffer, size_t tiles) {
    status = reinterpret_cast<unsigned *>(buffer);
    aggregate = reinterpret_cast<T *>(buffer + status_bytes(tiles));
    inclusive = aggregate + tiles;
  }

  // __threadfence 保证读到状态的线程也能读到之前写的值。
  __device__ void publish(size_t tile, T value, unsigned st,
                          unsigned epoch) const {
    T *slot = st == STATUS_PREFIX ? inclusive : aggregate;
    *reinterpret_cast<volatile T *>(slot + tile) = value;
    __threadfence();
    *reinterpret_cast<volatile unsigned *>(status + tile) =
        status_word(epoch, st);
  }
  __device__ unsigned wait(size_t tile, unsigned epoch, T &value) const {
    unsigned word;
    do {
      word = *reinterpret_cast<volatile unsigned *>(status + tile);
    } while ((word >> 2) != epoch);
    const unsigned st = word & 3;
    __threadfence();
    const T *slot = st == STATUS_PREFIX ? inclusive : aggregate;
    value = *reinterpret_cast<const volatile T *>(slot + tile);
    return st;
  }
};

// 按领取顺序编号，编号更小的 tile 一定已经在运行，look-back 不会死锁。
__device__ unsigned g_tile_counter = 0;

// warp 0 调用，返回 tile 之前（同一行内）所有元素的归约，即这个 tile 的 carry。
// 每轮 32 个 lane 各看一个前驱 tile，按 lane 顺序归约，所以只要求结合律。
template <template <typename> class Op, typename T>
__device__ T look_back(const TileStates<T> &states, const size_t row_first,
                       const size_t tile, const unsigned epoch) {
  const int lane = threadIdx.x & 31;
  T prefix = Op<T>::identity();
  long long window_end = static_cast<long long>(tile);
  while (true) {
    const long long pred = window_end - 32 + lane;
    unsigned status = STATUS_PREFIX; // 行首之前当作已知的 identity
    T value = Op<T>::identity();
    if (pred >= static_cast<long long>(row_first)) {
      status = states.wait(pred, epoch, value);
    }

    // 最靠后的 P 之前的 tile 已经包含在它的前缀里。
    const unsigned prefix_mask =
        __ballot_sync(0xffffffff, status == STATUS_PREFIX);
    const int last_prefix = prefix_mask ? 31 - __clz(prefix_mask) : -1;
    if (lane < last_prefix) {
      value = Op<T>::identity();
    }
    // 低 lane 在左合并，lane 0 得到整个窗口按顺序的归约。
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const T other = __shfl_down_sync(0xffffffff, value, off);
      if (lane + off < 32) {
        value = Op<T>::apply(value, other);
      }
    }
    // 窗口在已有前缀之前，所以放在左边。
    prefix = Op<T>::apply(__shfl_sync(0xffffffff, value, 0), prefix);
    if (prefix_mask != 0) {
      return prefix;
    }
    window_end -= 32;
  }
}

template <bool Exclusive, template <typename> class Op, typename T>
__global__ void __launch_bounds__(BLOCK_SIZE)
    scan_decoupled(const T *input, // (B, N,)
                   T *output,      // (B, N,)
                   const TileStates<T> states, const size_t N,
                   const size_t tiles_per_row, const unsigned total_tiles,
                   const unsigned epoch) {
  __shared__ T tile[padded(TILE)];
  __shared__ unsigned tile_id;
  __shared__ T carry;

  if (threadIdx.x == 0) {
    unsigned id = blockIdx.x;
    if (states.enabled()) {
      id = atomicAdd(&g_tile_counter, 1u);
      // 最后一个领号的 block 清零，其他 block 都已领过号，下次调用直接复用。
      if (id == total_tiles - 1) {
        atomicExch(&g_tile_counter, 0u);
      }
    }
    tile_id = id;
  }
  __syncthreads();

  const size_t id = tile_id;
  const size_t row = id / tiles_per_row, t = id % tiles_per_row;
  const size_t base = row * N + t * TILE;
  const int valid = int(min(size_t(TILE), N - t * TILE));

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
  T total;
  const T thread_prefix = block_scan<true, Op>(items[ITEMS - 1], &total);

  T tile_carry = Op<T>::identity();
  if (states.enabled() && t > 0) {
    // 先发布总和，让后面的 tile 尽早往前推进，再回看自己的前缀。
    if (threadIdx.x == 0) {
      states.publish(id, total, STATUS_AGGREGATE, epoch);
    }
    if (threadIdx.x < 32) {
      const T prefix = look_back<Op>(states, row * tiles_per_row, id, epoch);
      if (threadIdx.x == 0) {
        states.publish(id, Op<T>::apply(prefix, total), STATUS_PREFIX, epoch);
        carry = prefix;
      }
    }
    __syncthreads();
    tile_carry = carry;
  } else if (states.enabled() && threadIdx.x == 0) {
    states.publish(id, total, STATUS_PREFIX, epoch);
  }

  const T prefix = Op<T>::apply(tile_carry, thread_prefix);
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
      output[base + i] = tile[padded(i)];
    }
  }
}

// Op 是运算模板，T 由 input/output 的指针类型推导。支持 input == output。
template <bool Exclusive = false, template <typename> class Op = Add,
          typename T>
void scan(const T *input, T *output, const size_t B, const size_t N) {
  if (N <= 0 || B <= 0)
    return;

  const size_t tiles_per_row = (N + TILE - 1) / TILE;
  const unsigned total_tiles = unsigned(B * tiles_per_row);
  if (tiles_per_row == 1) {
    // 每行一个 tile，不需要跨 tile 传递前缀。
    scan_decoupled<Exclusive, Op><<<total_tiles, BLOCK_SIZE>>>(
        input, output, TileStates<T>{}, N, tiles_per_row, total_tiles, 0);
    return;
  }

  // 状态缓冲区按需增长，跨调用复用；靠 epoch 区分新旧状态，只在分配或 epoch
  // 用完时清零。
  static TileStates<T> states;
  static size_t capacity = 0;
  static unsigned epoch = MAX_EPOCH;
  static char *buffer = nullptr;
  if (total_tiles > capacity) {
    cudaFree(buffer); // 隐式同步，旧缓冲区不会还在被使用
    capacity = std::max(size_t(total_tiles), 2 * capacity);
    cudaMalloc(&buffer, TileStates<T>::total_bytes(capacity));
    states.bind(buffer, capacity);
    epoch = MAX_EPOCH;
  }
  if (epoch == MAX_EPOCH) {
    cudaMemset(buffer, 0, TileStates<T>::status_bytes(capacity));
    epoch = 0;
  }
  ++epoch;

  scan_decoupled<Exclusive, Op><<<total_tiles, BLOCK_SIZE>>>(
      input, output, states, N, tiles_per_row, total_tiles, epoch);
}

__device__ unsigned int g_tmp[MAX_N];
// count_digits 写入 [digit][block] 计数，原地 exclusive scan 后
// 变为每个 (digit, block) 在输出中的起始位置。
__device__ int g_block_counts[MAX_COUNTS];

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

    int pos = g_block_counts[digit * blocks + blockIdx.x] + rank;
    out[pos] = value;
  }
}

// input、output 均为 device pointer
extern "C" void solve(const unsigned int *input, unsigned int *output, int N) {
  int blocks = (N + THREADS - 1) / THREADS;

  // 需要在 host 端当指针传递的静态 buffer，取其设备地址（不分配内存）
  unsigned int *tmp;
  int *block_counts;
  cudaGetSymbolAddress((void **)&tmp, g_tmp);
  cudaGetSymbolAddress((void **)&block_counts, g_block_counts);

  const unsigned int *src = input;
  unsigned int *dst = tmp;

  for (int shift = 0; shift < 32; shift += 4) {
    count_digits<<<blocks, THREADS>>>(src, N, shift);
    // [digit][block] 布局下，一次 exclusive scan 同时得到
    // bucket 起点与 block 在 bucket 内的偏移。
    scan<true>(block_counts, block_counts, 1, RADIX * blocks);
    stable_scatter<<<blocks, THREADS>>>(src, dst, N, shift, blocks);

    src = dst;
    dst = (dst == tmp) ? output : tmp;
  }
}
