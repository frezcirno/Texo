#include <cuda_runtime.h>

__global__ void blur(const float *input, const float *kernel, float *output,
                     const int input_rows, const int input_cols,
                     const int kernel_rows, const int kernel_cols) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= input_cols || y >= input_rows)
    return;
  float result = 0;
  for (int ky = 0; ky < kernel_rows; ++ky) {
    for (int kx = 0; kx < kernel_cols; ++kx) {
      int iy = y + ky - kernel_rows / 2;
      int ix = x + kx - kernel_cols / 2;
      if (iy >= 0 && iy < input_rows && ix >= 0 && ix < input_cols)
        result += input[size_t(iy) * input_cols + ix] *
                  kernel[size_t(ky) * kernel_cols + kx];
    }
  }
  output[y * input_cols + x] = result;
}

// input, kernel, output are device pointers
extern "C" void solve(const float *input, const float *kernel, float *output,
                      int input_rows, int input_cols, int kernel_rows,
                      int kernel_cols) {
  if (input_rows <= 0 || input_cols <= 0 || kernel_rows <= 0 || kernel_cols <= 0)
    return;
  blur<<<dim3((input_cols + 7) / 8, (input_rows + 31) / 32), dim3(8, 32)>>>(
      input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
}
