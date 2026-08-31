#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float *__restrict__ input,  // (M, N)
                              const float *__restrict__ kernel, // (X, Y)
                              float *__restrict__ output, // (M-X+1, N-Y+1)
                              int M, int N, int X, int Y) {
  const int output_rows = M - X + 1;
  const int output_cols = N - Y + 1;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= output_rows || col >= output_cols) {
    return;
  }
  float sum = 0;
  for (int r = 0; r < X; r++) {
    for (int c = 0; c < Y; c++) {
      sum += input[(row + r) * N + col + c] * kernel[r * Y + c];
    }
  }
  output[row * output_cols + col] = sum;
}

// input, kernel, output are device pointers
extern "C" void solve(const float *input, const float *kernel, float *output,
                      int input_rows, int input_cols, int kernel_rows,
                      int kernel_cols) {
  if (input_rows <= 0 || input_cols <= 0 || kernel_rows <= 0 ||
      kernel_cols <= 0 || kernel_rows > input_rows ||
      kernel_cols > input_cols) {
    return;
  }
  constexpr int TILE_ROW = 8;
  constexpr int TILE_COL = 32;
  int output_rows = input_rows - kernel_rows + 1;
  int output_cols = input_cols - kernel_cols + 1;
  conv2d_kernel<<<dim3((output_cols + TILE_COL - 1) / TILE_COL,
                       (output_rows + TILE_ROW - 1) / TILE_ROW),
                  dim3(TILE_COL, TILE_ROW)>>>(
      input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
}
