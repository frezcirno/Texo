#include <cmath>
#include <cuda_runtime.h>

__global__ void max_pooling_2d(const float *input, // (N, C, H, W)
                               float *output, // (N, C, H+2p-s*k+1, W+2p-s*k+1)
                               int N, int C, int H, int W, int kernel_size,
                               int stride, int padding) {
  //
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t Hout = (H + 2 * padding - kernel_size) / stride + 1;
  const size_t Wout = (W + 2 * padding - kernel_size) / stride + 1;
  if (y >= Hout || x >= Wout)
    return;
  const size_t n = (blockIdx.z * blockDim.z) / C;
  const size_t c = (blockIdx.z * blockDim.z) % C;

  auto access = [&](int h, int w) {
    if (h < 0 || h >= H || w < 0 || w >= W) {
      return -INFINITY;
    }
    return input[n * C * H * W + c * H * W + h * W + w];
  };

  float maxv = -INFINITY;
  for (int row = 0; row < kernel_size; row++) {
    for (int col = 0; col < kernel_size; col++) {
      maxv = max(
          maxv, access(y * stride + row - padding, x * stride + col - padding));
    }
  }

  const size_t stride_h = Wout;
  const size_t stride_c = Hout * stride_h;
  const size_t stride_n = C * stride_c;
  output[n * stride_n + c * stride_c + y * stride_h + x] = maxv;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, // (N, C, H, W)
                      float *output,      // (N, C, H+2p-s*k+1, W+2p-s*k+1)
                      int N, int C, int H, int W, int kernel_size, int stride,
                      int padding) {
  dim3 work;
  work.y = (H + 2 * padding - kernel_size) / stride + 1;
  work.x = (W + 2 * padding - kernel_size) / stride + 1;
  max_pooling_2d<<<dim3((work.x + 15) / 16, (work.y + 15) / 16, N * C),
                   dim3(16, 16)>>>(input, output, N, C, H, W, kernel_size,
                                   stride, padding);
}
