// 线性递推 h[l] = a[l] * h[l-1] + x[l]，h[0] = x[0]，按行独立。
// 把每一步看作仿射变换 (A, X)，对变换的复合做单遍 inclusive scan。
// Decoupled look-back（见 Merrill & Garland 2016）：每个 block 动态领取一个
// tile，扫描后先发布 tile 总和（A），再由 warp 0 一次检查前面 32 个 tile：
// 遇到已发布完整前缀（P）的 tile 就停，否则累加总和继续向前。
// 显存流量 12 字节/元素（读 a、x，写 h），1 次 kernel 启动。
// tile 状态缓冲区跨调用复用，不能在多个 stream 上并发调用。

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// Pack (A, X) into one 64-bit scan value.  The scan's shuffle operations then
// move both float components together.
__device__ inline uint64_t pack_affine(float A, float X) {
  return (uint64_t(__float_as_uint(A)) << 32) | __float_as_uint(X);
}

__device__ inline float affine_A(uint64_t v) {
  return __uint_as_float(static_cast<unsigned int>(v >> 32));
}

__device__ inline float affine_X(uint64_t v) {
  return __uint_as_float(static_cast<unsigned int>(v));
}

template <typename T> struct AffineCompose {
  __host__ __device__ static constexpr T identity() {
    return T(0x3f80000000000000ull); // packed (1.0f, 0.0f)
  }
  // Apply the left interval first, then the right interval:
  // (A2, X2) o (A1, X1) = (A2*A1, A2*X1 + X2).
  __device__ static T apply(T left, T right) {
    const float A1 = affine_A(left), X1 = affine_X(left);
    const float A2 = affine_A(right), X2 = affine_X(right);
    return pack_affine(A2 * A1, A2 * X1 + X2);
  }
};

// 假设一维 block，完整 warp 的 32 个线程一起调用。
template <bool Exclusive = false, template <typename> class Op, typename T>
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
template <bool Exclusive = false, template <typename> class Op, typename T>
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
  static constexpr size_t status_bytes(size_t tiles) {
    return tiles * sizeof(unsigned long long);
  }
  static constexpr size_t total_bytes(size_t tiles) {
    return status_bytes(tiles);
  }
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
  static constexpr size_t status_bytes(size_t tiles) {
    return (tiles * sizeof(unsigned) + 15) / 16 * 16;
  }
  static constexpr size_t total_bytes(size_t tiles) {
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

// h[l] = a[l] * h[l-1] + x[l] 看作仿射变换 (A, X) 的复合，做一遍 inclusive
// scan， 前缀的 X 分量就是 h。读 a、x，写 h，显存流量 12 字节/元素。
using Affine = uint64_t;

__global__ void __launch_bounds__(BLOCK_SIZE)
    linear_recurrence(const float *__restrict__ a, // (B, L)
                      const float *__restrict__ x, // (B, L)
                      float *__restrict__ h,       // (B, L)
                      const TileStates<Affine> states, const size_t L,
                      const size_t tiles_per_row, const unsigned total_tiles,
                      const unsigned epoch) {
  __shared__ float tile_a[padded(TILE)];
  __shared__ float tile_x[padded(TILE)]; // 也用来暂存输出 h
  __shared__ unsigned tile_id;
  __shared__ Affine carry;

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
  const size_t base = row * L + t * TILE;
  const int valid = int(min(size_t(TILE), L - t * TILE));

  // 越界位置填 identity (1, 0)。行首的 a 不参与计算（行首 carry 是 identity）。
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    const int i = k * BLOCK_SIZE + threadIdx.x;
    const bool in = i < valid;
    tile_a[padded(i)] = in ? a[base + i] : 1.0f;
    tile_x[padded(i)] = in ? x[base + i] : 0.0f;
  }
  __syncthreads();

  // 线程内对连续 ITEMS 个元素做 inclusive scan，A、X 分开放在寄存器里。
  float A[ITEMS], X[ITEMS];
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    A[k] = tile_a[padded(threadIdx.x * ITEMS + k)];
    X[k] = tile_x[padded(threadIdx.x * ITEMS + k)];
  }
#pragma unroll
  for (int k = 1; k < ITEMS; ++k) {
    X[k] = A[k] * X[k - 1] + X[k];
    A[k] = A[k] * A[k - 1];
  }

  // 只对每个线程的总和做 block scan；block_scan 结尾的同步保证 tile 可以复用。
  Affine total;
  const Affine thread_prefix = block_scan<true, AffineCompose>(
      pack_affine(A[ITEMS - 1], X[ITEMS - 1]), &total);

  Affine tile_carry = AffineCompose<Affine>::identity();
  if (states.enabled() && t > 0) {
    // 先发布总和，让后面的 tile 尽早往前推进，再回看自己的前缀。
    if (threadIdx.x == 0) {
      states.publish(id, total, STATUS_AGGREGATE, epoch);
    }
    if (threadIdx.x < 32) {
      const Affine prefix =
          look_back<AffineCompose>(states, row * tiles_per_row, id, epoch);
      if (threadIdx.x == 0) {
        states.publish(id, AffineCompose<Affine>::apply(prefix, total),
                       STATUS_PREFIX, epoch);
        carry = prefix;
      }
    }
    __syncthreads();
    tile_carry = carry;
  } else if (states.enabled() && threadIdx.x == 0) {
    states.publish(id, total, STATUS_PREFIX, epoch);
  }

  // 只需要前缀的 X 分量：h = A_local * X_prefix + X_local。
  const float h_prefix =
      affine_X(AffineCompose<Affine>::apply(tile_carry, thread_prefix));
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    tile_x[padded(threadIdx.x * ITEMS + k)] = A[k] * h_prefix + X[k];
  }
  __syncthreads();

#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    const int i = k * BLOCK_SIZE + threadIdx.x;
    if (i < valid) {
      h[base + i] = tile_x[padded(i)];
    }
  }
}

constexpr size_t MAX_B = 256, MAX_L = 65536;
constexpr size_t MAX_TILES = MAX_B * ((MAX_L + TILE - 1) / TILE);
__device__ char g_tile_states[TileStates<Affine>::total_bytes(MAX_TILES)];

// a, x, h are device pointers
extern "C" void solve(const float *a, // (B, L)
                      const float *x, // (B, L)
                      float *h,       // (B, L)
                      int B, int L) {
  if (B <= 0 || L <= 0)
    return;

  const size_t tiles_per_row = (size_t(L) + TILE - 1) / TILE;
  const unsigned total_tiles = unsigned(B * tiles_per_row);
  if (tiles_per_row == 1) {
    // 每行一个 tile，不需要跨 tile 传递前缀。
    linear_recurrence<<<total_tiles, BLOCK_SIZE>>>(
        a, x, h, TileStates<Affine>{}, L, tiles_per_row, total_tiles, 0);
    return;
  }

  // 靠 epoch 区分新旧状态，只在第一次或 epoch 用完时清零。
  static TileStates<Affine> states;
  static unsigned epoch = MAX_EPOCH;
  if (!states.enabled()) {
    void *buffer;
    cudaGetSymbolAddress(&buffer, g_tile_states);
    states.bind(static_cast<char *>(buffer), MAX_TILES);
  }
  if (epoch == MAX_EPOCH) {
    cudaMemset(states.status, 0, TileStates<Affine>::status_bytes(MAX_TILES));
    epoch = 0;
  }
  ++epoch;

  linear_recurrence<<<total_tiles, BLOCK_SIZE>>>(
      a, x, h, states, L, tiles_per_row, total_tiles, epoch);
}
