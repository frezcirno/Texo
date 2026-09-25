#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void mm_int8(const int8_t *__restrict__ A, // (M, K)
                        const int8_t *__restrict__ B, // (K, N)
                        int8_t *__restrict__ C,       // (M, N)
                        int M, int N, int K, float scale_A, float scale_B,
                        float scale_C, int zero_point_A, int zero_point_B,
                        int zero_point_C) {
  const int cx = blockIdx.x * blockDim.x + threadIdx.x;
  const int cy = blockIdx.y * blockDim.y + threadIdx.y;
  if (cy >= M || cx >= N)
    return;
  // Accumulate quantized products exactly; scaling each term in FP32 can
  // introduce a whole output-unit error after long, cancelling reductions.
  int64_t dot = 0;
  for (int i = 0; i < K; i++) {
    dot += int64_t(A[size_t(cy) * K + i] - zero_point_A) *
           (B[size_t(i) * N + cx] - zero_point_B);
  }
  // Keep the FP32 operations in this order; precombining the scale ratio or
  // promoting it to double can change decisions near a half-integer boundary.
  float sum = float(dot) * scale_A;
  sum = sum * scale_B;
  sum = sum / scale_C;
  // Quantization uses round-to-nearest with halfway values rounded to even.
  sum = nearbyintf(sum) + zero_point_C;
  if (sum > 127)
    sum = 127;
  if (sum < -128)
    sum = -128;
  const size_t index = size_t(cy) * N + cx;
  C[index] = sum;
}

// A, B, C are device pointers
extern "C" void solve(const int8_t *A, // (M, K)
                      const int8_t *B, // (K, N)
                      int8_t *C,       // (M, N)
                      int M, int N, int K, float scale_A, float scale_B,
                      float scale_C, int zero_point_A, int zero_point_B,
                      int zero_point_C) {
  dim3 blockDim(16, 16);
  dim3 gridDim((N + 15) / 16, (M + 15) / 16);
  mm_int8<<<gridDim, blockDim>>>(A, B, C, M, N, K, scale_A, scale_B, scale_C,
                                 zero_point_A, zero_point_B, zero_point_C);
}
