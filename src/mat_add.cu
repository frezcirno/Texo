#include <cuda_runtime.h>

__global__ void matrix_add(const float4 *__restrict__ A,
                           const float4 *__restrict__ B, float4 *__restrict__ C,
                           size_t n4, const float *__restrict__ At,
                           const float *__restrict__ Bt, float *__restrict__ Ct,
                           int tail) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) {
    float4 a = __ldcs(A + i), b = __ldcs(B + i);
    __stcs(C + i, make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w));
  }
  // 尾巴（最多 3 个元素）
  if (blockIdx.x == 0 && threadIdx.x < tail) {
    Ct[threadIdx.x] = At[threadIdx.x] + Bt[threadIdx.x];
  }
}

extern "C" void solve(const float *A, const float *B, float *C, int N) {
  const size_t total = (size_t)N * N;
  const size_t n4 = total / 4;
  const int tail = total % 4;
  const size_t off = n4 * 4;

  int dev, sms;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

  const int threads = 256;
  size_t need = (n4 + threads - 1) / threads;
  int blocks = (int)(need < (size_t)sms * 8 ? (need ? need : 1) : sms * 8);

  matrix_add<<<blocks, threads>>>(
      reinterpret_cast<const float4 *>(A), reinterpret_cast<const float4 *>(B),
      reinterpret_cast<float4 *>(C), n4, A + off, B + off, C + off, tail);
  // cudaDeviceSynchronize();
}
