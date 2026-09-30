#include <cuda_runtime.h>

__global__ void rope_kernel(float *Q,      // (M, D)
                            float *cos,    // (M, D)
                            float *sin,    // (M, D)
                            float *output, // (M, D)
                            const size_t M, const size_t D) {
  // output[row][i] = Q * cos + rotate_half(Q) * sin
  const size_t row = blockIdx.x;
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  const auto rotate_half = [&](size_t i) {
    if (i < D / 2)
      return -Q[row * D + i + D / 2];
    return Q[row * D + i - D / 2];
  };

  for (int i = tid; i < D; i += stride) {
    output[row * D + i] =
        Q[row * D + i] * cos[row * D + i] + rotate_half(i) * sin[row * D + i];
  }
}

// Q, cos, sin, output are device pointers
extern "C" void solve(float *Q,      // (M, D)
                      float *cos,    // (M, D)
                      float *sin,    // (M, D)
                      float *output, // (M, D)
                      int M, int D) {
  rope_kernel<<<M, 256>>>(Q, cos, sin, output, M, D);
}
