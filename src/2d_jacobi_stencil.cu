#include <cuda_runtime.h>

__global__ void solve_kernel(const float *__restrict__ input, // (rows, cols)
                             float *__restrict__ output,      // (rows, cols)
                             const size_t rows, const size_t cols) {
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cols || y >= rows)
    return;
  if (x == 0 || x == cols - 1 || y == 0 || y == rows - 1) {
    output[y * cols + x] = input[y * cols + x];
    return;
  }
  output[y * cols + x] = (input[y * cols + x - 1] +   //
                          input[y * cols + x + 1] +   //
                          input[(y - 1) * cols + x] + //
                          input[(y + 1) * cols + x]) /
                         4;
}

// input, output are device pointers
extern "C" void solve(const float *input, // (rows, cols)
                      float *output,      // (rows, cols)
                      int rows, int cols) {
  solve_kernel<<<dim3((cols + 15) / 16, (rows + 15) / 16), dim3(16, 16)>>>(
      input, output, rows, cols);
}
