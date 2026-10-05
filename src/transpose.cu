#include <cuda_runtime.h>

#define TILE 32
#define BLOCK_ROWS 8

__global__ void
matrix_transpose_kernel(const float *__restrict__ input, // (R, C)
                        float *__restrict__ output,      // (C, R)
                        size_t R, size_t C) {
  // 32*32 buf
  __shared__ float tile[TILE][TILE + 1]; // +1 padding 消除 bank conflict

  size_t x = blockIdx.x * TILE + threadIdx.x; // input 列
  size_t y = blockIdx.y * TILE + threadIdx.y; // input 行

  // load from input
#pragma unroll
  for (int i = 0; i < TILE; i += BLOCK_ROWS) {
    if (x < C && y + i < R) {
      tile[threadIdx.y + i][threadIdx.x] = input[(y + i) * C + x];
    }
  }

  __syncthreads();

  x = blockIdx.y * TILE + threadIdx.x; // output 列 (= input 行)
  y = blockIdx.x * TILE + threadIdx.y; // output 行 (= input 列)

  // save to output
#pragma unroll
  for (int i = 0; i < TILE; i += BLOCK_ROWS) {
    if (x < R && y + i < C) {
      output[(y + i) * R + x] = tile[threadIdx.x][threadIdx.y + i];
    }
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, float *output, int rows, int cols) {
  matrix_transpose_kernel<<<dim3((cols + TILE - 1) / TILE,
                                 (rows + TILE - 1) / TILE),
                            dim3(TILE, BLOCK_ROWS)>>>(input, output, rows,
                                                      cols);
  cudaDeviceSynchronize();
}
