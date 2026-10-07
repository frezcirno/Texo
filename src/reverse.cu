#include <cuda_runtime.h>

__global__ void reverse_array(float *input, int N) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N / 2)
    return;

  float v = input[tid];
  input[tid] = input[N - tid - 1];
  input[N - tid - 1] = v;
}

// input is device pointer
extern "C" void solve(float *input, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N / 2 + threadsPerBlock - 1) / threadsPerBlock;

  reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
}
