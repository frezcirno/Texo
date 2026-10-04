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

// 每个 block 独立扫描，并记录这个 block 的总和。
template <bool Exclusive = false, template <typename> class Op = Add,
          typename T>
__global__ void block_scan(const T *input, // (B, N,)
                           T *output,      // (B, N,)
                           T *block_sums,  // (B, block_num,)
                           const size_t block_num, const size_t N) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t b = blockIdx.y;
  const T val = tid < N ? input[b * N + tid] : Op<T>::identity();
  const T prefix_sum = block_scan<Exclusive, Op>(val);
  if (tid < N) {
    output[b * N + tid] = prefix_sum;
  }
  if (block_sums != nullptr && threadIdx.x == blockDim.x - 1) {
    // block last thread runs
    block_sums[b * block_num + blockIdx.x] =
        Exclusive ? Op<T>::apply(prefix_sum, val) : prefix_sum;
  }
}

template <template <typename> class Op = Add, typename T>
__global__ void add_block_offsets(T *output,               // (B, N,)
                                  const T *block_prefixes, // (B, block_num,)
                                  const size_t B, const size_t N,
                                  const size_t block_num) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t b = blockIdx.y;
  if (tid < N && blockIdx.x > 0) {
    // block_prefixes 是 inclusive：前一个位置就是当前 block 的偏移。
    output[b * N + tid] = Op<T>::apply(
        block_prefixes[b * block_num + blockIdx.x - 1], output[b * N + tid]);
  }
}

// Op 是运算模板，T 由 input/output 的指针类型推导。
template <bool Exclusive = false, template <typename> class Op = Add,
          typename T>
void scan(const T *input, T *output, const size_t B, const size_t N) {
  if (N <= 0)
    return;

  constexpr size_t BLOCK_SIZE = 256;
  const size_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // 递归终点：一个 block 就能完成扫描。
  if (block_num == 1) {
    block_scan<Exclusive, Op, T>
        <<<dim3(1, B), BLOCK_SIZE>>>(input, output, nullptr, block_num, N);
    return;
  }

  T *block_sums = nullptr;
  cudaMalloc(&block_sums, B * block_num * sizeof(T));

  block_scan<Exclusive, Op><<<dim3(block_num, B), BLOCK_SIZE>>>(
      input, output, block_sums, block_num, N);

  // block_sums 本身也可能超过一个 block，因此递归做 inclusive scan。
  scan<false, Op>(block_sums, block_sums, B, block_num);
  add_block_offsets<Op>
      <<<dim3(block_num, B), BLOCK_SIZE>>>(output, block_sums, B, N, block_num);

  cudaFree(block_sums);
}

__global__ inline void reverse_array(const float *__restrict__ input,
                                     float *__restrict__ output, const size_t B,
                                     const size_t L) {
  const size_t BL = blockIdx.x * blockDim.x + threadIdx.x;
  if (BL >= B * L)
    return;
  const size_t b = BL / L;
  const size_t l = BL % L;
  output[b * L + l] = input[b * L + L - l - 1];
}

template <size_t BLOCK_SIZE>
__global__ void network(const float *cum_a, // (B, L)
                        const float *x,     // (B, L)
                        float *h,           // (B, L)
                        const size_t B, const size_t L) {
  // h0 = x0
  // h1 = a1 x0 + x1
  // h2 = a2 a1 x0 + a2 x1 + x2
  // h3 = a3 a2 a1 x0 + a3 a2 x1 + a3 x2 + x3
  // h4 = a4 a3 a2 a1 x0 + a4 a3 a2 x1 + a4 a3 x2 + a4 x3 + x4

  // a[0] = 1.0
  // cum_a[i] = a[0] ... a[i]
  const size_t hl = blockIdx.x;
  const size_t b = blockIdx.y;
  float sum = 0;
  for (int xi = threadIdx.x; xi <= hl; xi += BLOCK_SIZE) {
    sum += (hl > xi ? x[b * L + xi] * cum_a[b * L + hl] / cum_a[b * L + xi]
                    : x[b * L + xi]);
  }
  __shared__ float total;
  if (threadIdx.x == 0) {
    total = 0;
  }
  __syncthreads();
  atomicAdd(&total, sum);
  __syncthreads();
  if (threadIdx.x == 0) {
    h[b * L + hl] = total;
  }
  __syncthreads();
}

__global__ inline void copy_kernel(const float *__restrict__ in,
                                   float *__restrict__ out, const size_t B,
                                   const size_t L) {
  //
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= B * L)
    return;
  const size_t b = tid / L;
  const size_t l = tid % L;
  out[b * L + l] = l == 0 ? 1.0 : in[b * L + l];
}

// a, x, h are device pointers
extern "C" void solve(const float *a, // (B, L)
                      const float *x, // (B, L)
                      float *h,       // (B, L)
                      int B, int L) {
  float *cum_a;
  cudaMalloc(&cum_a, B * L * sizeof(float));
  cudaMemset(cum_a, 0, B * L * sizeof(float));

  float *a1;
  cudaMalloc(&a1, B * L * sizeof(float));
  copy_kernel<<<(B * L + 255) / 256, 256>>>(a, a1, B, L);

  scan<false, Mul>(a1, cum_a, B, L);
  // a[0] = 0
  // cum_a[i] = a[0] ... a[i]

  network<256><<<dim3(L, B), 256>>>(cum_a, x, h, B, L);

  cudaFree(a1);
  cudaFree(cum_a);
}
