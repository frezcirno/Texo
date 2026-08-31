#include <cuda_runtime.h>

__global__ void
conv3d_kernel_rdc(const float *__restrict__ input,  // (IR, ID, IC)
                  const float *__restrict__ kernel, // (KR, KD, KC)
                  float *__restrict__ output,       //
                  int input_depth, int input_rows, int input_cols,
                  int kernel_depth, int kernel_rows, int kernel_cols) {
  const int output_cols = input_cols - kernel_cols + 1;
  const int output_rows = input_rows - kernel_rows + 1;
  const int output_depth = input_depth - kernel_depth + 1;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int depth = blockIdx.z * blockDim.z + threadIdx.z;
  if (col >= output_cols || row >= output_rows || depth >= output_depth) {
    return;
  }
  float sum = 0;
  for (int r = 0; r < kernel_rows; r++) {
    for (int d = 0; d < kernel_depth; d++) {
      for (int c = 0; c < kernel_cols; c++) {
        int d1 = depth + d;
        int r1 = row + r;
        int c1 = col + c;
        sum += input[(r1 * input_depth + d1) * input_cols + c1] *
               kernel[(r * kernel_depth + d) * kernel_cols + c];
      }
    }
  }
  output[(row * output_depth + depth) * output_cols + col] = sum;
}

__global__ void
conv3d_kernel_drc(const float *__restrict__ input,  // (ID, IR, IC)
                  const float *__restrict__ kernel, // (KD, KR, KC)
                  float *__restrict__ output,       //
                  int input_depth, int input_rows, int input_cols,
                  int kernel_depth, int kernel_rows, int kernel_cols) {
  const int output_cols = input_cols - kernel_cols + 1;
  const int output_rows = input_rows - kernel_rows + 1;
  const int output_depth = input_depth - kernel_depth + 1;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int depth = blockIdx.z * blockDim.z + threadIdx.z;
  if (col >= output_cols || row >= output_rows || depth >= output_depth) {
    return;
  }
  float sum = 0;
  for (int d = 0; d < kernel_depth; d++) {
    for (int r = 0; r < kernel_rows; r++) {
      for (int c = 0; c < kernel_cols; c++) {
        int d1 = depth + d;
        int r1 = row + r;
        int c1 = col + c;
        sum += input[(d1 * input_rows + r1) * input_cols + c1] *
               kernel[(d * kernel_rows + r) * kernel_cols + c];
      }
    }
  }
  output[(depth * output_rows + row) * output_cols + col] = sum;
}

// input, kernel, output are device pointers
extern "C" void solve(const float *input,  //
                      const float *kernel, //
                      float *output,       //
                      int input_depth, int input_rows, int input_cols,
                      int kernel_depth, int kernel_rows, int kernel_cols) {
  constexpr int TILE_COL = 32;
  constexpr int TILE_ROW = 4;
  constexpr int TILE_DEPTH = 2;
  int output_rows = input_rows - kernel_rows + 1;
  int output_cols = input_cols - kernel_cols + 1;
  int output_depth = input_depth - kernel_depth + 1;
  conv3d_kernel_drc<<<dim3((output_cols + TILE_COL - 1) / TILE_COL,
                           (output_rows + TILE_ROW - 1) / TILE_ROW,
                           (output_depth + TILE_DEPTH - 1) / TILE_DEPTH),
                      dim3(TILE_COL, TILE_ROW, TILE_DEPTH)>>>(
      input, kernel, output, input_depth, input_rows, input_cols, kernel_depth,
      kernel_rows, kernel_cols);
}
