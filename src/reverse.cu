#include <cuda_runtime.h>

__global__ void reverse_array(float *input, float *buffer, int N) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
    return;
  buffer[tid] = input[N - tid - 1];
}

// input is device pointer
extern "C" void solve(float *input, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  float *buffer;
  cudaMalloc(&buffer, N * sizeof(float));
  reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, buffer, N);
  cudaMemcpy(input, buffer, N * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaFree(buffer);
  cudaDeviceSynchronize();
}
