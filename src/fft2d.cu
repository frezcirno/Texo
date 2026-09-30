#include <cuda_runtime.h>
#include <math_constants.h>

__device__ __forceinline__ int reverse_bits(int x, int bits) {
  int r = 0;
  for (int i = 0; i < bits; ++i) {
    r = (r << 1) | (x & 1);
    x >>= 1;
  }
  return r;
}

__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
  return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Fallback for dimensions that cannot be represented by a radix-2 FFT.
// The two passes retain the row-column decomposition, so this is O(M*N*(M+N))
// rather than a full O(M^2*N^2) 2D DFT.
__global__ void dft_rows(const float *signal, float *temporary, int M, int N) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = M * N;
  if (index >= count)
    return;

  const int row = index / N;
  const int frequency = index - row * N;
  float sum_real = 0.0f;
  float sum_imag = 0.0f;
  for (int col = 0; col < N; ++col) {
    const int input_index = row * N + col;
    float sine, cosine;
    sincosf(-2.0f * CUDART_PI_F * float(frequency) * float(col) / float(N),
            &sine, &cosine);
    const float real = signal[2 * input_index];
    const float imag = signal[2 * input_index + 1];
    sum_real += real * cosine - imag * sine;
    sum_imag += real * sine + imag * cosine;
  }
  temporary[2 * index] = sum_real;
  temporary[2 * index + 1] = sum_imag;
}

__global__ void dft_columns(const float *temporary, float *spectrum, int M, int N) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = M * N;
  if (index >= count)
    return;

  const int frequency_row = index / N;
  const int col = index - frequency_row * N;
  float sum_real = 0.0f;
  float sum_imag = 0.0f;
  for (int row = 0; row < M; ++row) {
    const int input_index = row * N + col;
    float sine, cosine;
    sincosf(-2.0f * CUDART_PI_F * float(frequency_row) * float(row) / float(M),
            &sine, &cosine);
    const float real = temporary[2 * input_index];
    const float imag = temporary[2 * input_index + 1];
    sum_real += real * cosine - imag * sine;
    sum_imag += real * sine + imag * cosine;
  }
  spectrum[2 * index] = sum_real;
  spectrum[2 * index + 1] = sum_imag;
}

// 每个 block 做一条长度 L 的 FFT。
// 第 t 条向量的第 i 个元素在 base = t * transform_stride + i * element_stride。
__global__ void fft1d(const float *in, float *out, int L, int transform_stride,
                      int element_stride) {
  extern __shared__ float2 smem[];
  float2 *values = smem;      // L 个复数
  float2 *twiddle = smem + L; // 最多 L/2 个旋转因子

  const int tid = threadIdx.x;
  const int base = blockIdx.x * transform_stride;

  int logL = 0;
  for (int p = L; p > 1; p >>= 1)
    ++logL;

  // 读入 shared memory，同时完成 bit-reversal 排列。
  for (int i = tid; i < L; i += blockDim.x) {
    int src = base + i * element_stride;
    int dst = reverse_bits(i, logL);
    values[dst] = make_float2(in[2 * src], in[2 * src + 1]);
  }
  __syncthreads();

  // iterative radix-2 DIT FFT
  for (int len = 2; len <= L; len <<= 1) {
    int half = len >> 1;

    // 本 stage 的旋转因子只计算一次，供整个 block 复用。
    for (int k = tid; k < half; k += blockDim.x) {
      float s, c;
      float angle = -2.0f * CUDART_PI_F * float(k) / float(len);
      sincosf(angle, &s, &c);
      twiddle[k] = make_float2(c, s);
    }
    __syncthreads();

    for (int p = tid; p < L / 2; p += blockDim.x) {
      int group = p / half;
      int k = p - group * half;
      int i0 = group * len + k;
      int i1 = i0 + half;

      float2 u = values[i0];
      float2 v = cmul(values[i1], twiddle[k]);

      values[i0] = make_float2(u.x + v.x, u.y + v.y);
      values[i1] = make_float2(u.x - v.x, u.y - v.y);
    }
    __syncthreads();
  }

  for (int i = tid; i < L; i += blockDim.x) {
    int dst = base + i * element_stride;
    out[2 * dst] = values[i].x;
    out[2 * dst + 1] = values[i].y;
  }
}

// signal, spectrum are device pointers
extern "C" void solve(const float *signal, float *spectrum, int M, int N) {
  const bool radix2 = M > 0 && N > 0 && !(M & (M - 1)) && !(N & (N - 1));
  if (!radix2) {
    constexpr int THREADS = 256;
    const int count = M * N;
    float *temporary = nullptr;
    cudaMalloc(&temporary, 2 * size_t(count) * sizeof(float));
    const int blocks = (count + THREADS - 1) / THREADS;
    dft_rows<<<blocks, THREADS>>>(signal, temporary, M, N);
    dft_columns<<<blocks, THREADS>>>(temporary, spectrum, M, N);
    cudaFree(temporary);
    return;
  }

  constexpr int THREADS = 256;

  // 一行：N 个 complex；twiddle 最多 N/2 个 complex。
  size_t row_smem = size_t(N + N / 2) * sizeof(float2);
  fft1d<<<M, THREADS, row_smem>>>(signal, spectrum, N, N, 1);

  // 一列：元素地址为 col + row*N。
  // 各 block 读写不同列，因此 spectrum 可以原地作为输入和输出。
  size_t col_smem = size_t(M + M / 2) * sizeof(float2);
  fft1d<<<N, THREADS, col_smem>>>(spectrum, spectrum, M, 1, N);
}
