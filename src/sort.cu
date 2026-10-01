#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void pad(float *tmp, const float *data, int N, int P) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < P)
    tmp[i] = (i < N) ? data[i] : CUDART_INF_F;
}

// 一个线程负责一对 (i, i + j)
__global__ void bitonic_step(float *a, int j, int k, int P) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= P / 2)
    return;

  int i = (t / j) * (2 * j) + (t % j);
  int partner = i + j;

  float x = a[i];
  float y = a[partner];
  bool ascending = (i & k) == 0;

  if ((x > y) == ascending) {
    a[i] = y;
    a[partner] = x;
  }
}

// data is a device pointer
extern "C" void solve(float *data, int N) {
  int P = 1;
  while (P < N)
    P <<= 1;

  float *tmp;
  cudaMalloc(&tmp, P * sizeof(float));

  constexpr int THREADS = 256;
  pad<<<(P + THREADS - 1) / THREADS, THREADS>>>(tmp, data, N, P);

  for (int k = 2; k <= P; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      bitonic_step<<<(P / 2 + THREADS - 1) / THREADS, THREADS>>>(tmp, j, k, P);
    }
  }

  cudaMemcpy(data, tmp, N * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaFree(tmp);
}