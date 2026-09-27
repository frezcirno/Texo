#include <cuda_runtime.h>

__global__ void gemm(const float *__restrict__ A, // (M, K)
                     const float *__restrict__ B, // (K, N)
                     float *__restrict__ C,       // (M, N)
                     size_t M, size_t N, size_t K) {
  const size_t cx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t cy = blockIdx.y * blockDim.y + threadIdx.y;
  if (cy >= M || cx >= N)
    return;
  float sum = 0.0f;
  for (int i = 0; i < K; i++) {
    sum += float(A[cy * K + i]) * float(B[i * N + cx]);
  }
  C[cy * N + cx] = sum;
}

// input, output are device pointers
extern "C" void solve(const float *input, // (N, N)
                      float *output,      // (N, N)
                      int N, int P) {
  if (P <= 1) {
    cudaMemcpy(output, input, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
    return;
  }

  if (P == 2) {
    gemm<<<dim3((N + 15) / 16, (N + 15) / 16), //
           dim3(16, 16)>>>(input, input, output, N, N, N);
    return;
  }

  if (P % 2 == 0) {
    float *input2;
    cudaMalloc(&input2, N * N * sizeof(*input2));

    gemm<<<dim3((N + 15) / 16, (N + 15) / 16), //
           dim3(16, 16)>>>(input, input, input2, N, N, N);

    solve(input2, output, N, P >> 1);

    cudaFree(input2);
    return;
  }

  solve(input, output, N, P - 1);

  float *inputP1;
  cudaMalloc(&inputP1, N * N * sizeof(*inputP1));
  cudaMemcpy(inputP1, output, N * N * sizeof(float), cudaMemcpyDeviceToDevice);

  gemm<<<dim3((N + 15) / 16, (N + 15) / 16), //
         dim3(16, 16)>>>(input, inputP1, output, N, N, N);

  cudaFree(inputP1);
}
