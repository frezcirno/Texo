#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void mm_int4(const __half *__restrict__ x,      // (M, K)
                        const uint8_t *__restrict__ w_q,   // (N, K/2)
                        const __half *__restrict__ scales, // (N, K/G)
                        __half *__restrict__ y,            // (M, N)
                        const size_t M, const size_t N, const size_t K,
                        const size_t group_size) {
  const size_t n = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N)
    return;
  float dot = 0;
  for (size_t k = 0; k < K; k++) {
    dot += float(x[m * K + k]) *
           (float((k % 2 == 0 ? w_q[n * (K / 2) + k / 2] >> 4
                              : w_q[n * (K / 2) + k / 2] & 0x0f) -
                  8) *
            float(scales[n * (K / group_size) + k / group_size]));
  }
  y[m * N + n] = dot;
}

// x, w_q, scales, y are device pointers
extern "C" void solve(const __half *x,      // (M, K)
                      const uint8_t *w_q,   // (N, K/2)
                      const __half *scales, // (N, K/G)
                      __half *y,            // (M, N)
                      int M, int N, int K, int group_size) {
  dim3 blockDim(16, 16);
  dim3 gridDim((N + 15) / 16, (M + 15) / 16);
  mm_int4<<<gridDim, blockDim>>>(x, w_q, scales, y, M, N, K, group_size);
}
