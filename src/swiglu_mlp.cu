#include <cuda_runtime.h>

template <typename T> __device__ inline T sigmoid(const T &val) {
  return 1 / (1 + expf(-val));
}

template <typename T> __device__ inline T silu(const T &val) {
  return val * sigmoid(val);
}

template <size_t BLOCK_SIZE>
__global__ void fused_kernel(const float *__restrict__ x,      // (M, D)
                             const float *__restrict__ W_gate, // (D, D2)
                             const float *__restrict__ W_up,   // (D, D2)
                             const float *__restrict__ W_down, // (D2, D)
                             float *__restrict__ output,       // (M, D)
                             const size_t M, const size_t D, const size_t D2) {
  const size_t d = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t m = blockIdx.y * blockDim.y + threadIdx.y;
  if (d >= D || m >= M)
    return;

  float sum = 0;
  for (int d2 = 0; d2 < D2; d2++) {
    // A = x @ W_gate
    // B = x @ W_up
    float a = 0; // (M, D2)
    float b = 0; // (M, D2)
    for (int d = 0; d < D; d++) {
      const auto xx = x[m * D + d];
      a += xx * W_gate[d * D2 + d2];
      b += xx * W_up[d * D2 + d2];
    }
    // output = (silu(A) . B) @ W_down
    sum += silu(a) * b * W_down[d2 * D + d];
  }
  output[m * D + d] = sum;
}

// x, W_gate, W_up, W_down, output are device pointers
extern "C" void solve(const float *x,      // (M, D)
                      const float *W_gate, // (D, D2)
                      const float *W_up,   // (D, D2)
                      const float *W_down, // (D2, D)
                      float *output,       // (M, D)
                      int M, int d_model, int d_ffn) {
  fused_kernel<256><<<dim3((d_model + 15) / 16, (M + 15) / 16), dim3(16, 16)>>>(
      x, W_gate, W_up, W_down, output, M, d_model, d_ffn);
}
