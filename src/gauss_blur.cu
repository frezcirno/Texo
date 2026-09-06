#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

__global__ void blur(const float *input, const float *kernel, float *output,
                     const int input_rows, const int input_cols,
                     const int kernel_rows, const int kernel_cols) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= input_cols || y >= input_rows)
    return;
  float result = 0;
  for (int j = -kernel_rows / 2; j <= kernel_rows / 2; j++) {
    for (int i = -kernel_cols / 2; i <= kernel_cols / 2; i++) {
      float this_input = 0;
      if (y + j >= 0 && y + j < input_rows && x + i >= 0 && x + i < input_cols)
        this_input = input[(y + j) * input_cols + x + i];
      result +=
          this_input *
          kernel[(kernel_rows / 2 + j) * kernel_cols + kernel_cols / 2 + i];
    }
  }
  output[y * input_cols + x] = result;
}

// input, kernel, output are device pointers
extern "C" void solve(const float *input, const float *kernel, float *output,
                      int input_rows, int input_cols, int kernel_rows,
                      int kernel_cols) {
  //   cudaMemset(output, 0, input_rows * input_cols * sizeof(float));
  blur<<<dim3((input_cols + 7) / 8, (input_rows + 31) / 32), dim3(8, 32)>>>(
      input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
}
