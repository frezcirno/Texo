#include <cuda_runtime.h>

__device__ float pow2(float x) { return x * x; }

__global__ void bn_kernel(const float *__restrict__ input, // (N, C)
                          const float *__restrict__ gamma, // (C,)
                          const float *__restrict__ beta,  // (C,)
                          float *__restrict__ output,      // (N, C)
                          int N, int C, float eps) {
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= C)
    return;
  // mu, sigma = f(input)
  // xhat = norm(input)
  // y = gamma * xhat + beta

  float mu = 0;
  for (int i = 0; i < N; i++) {
    mu += input[i * C + col];
  }
  mu /= N;

  float sigma2 = 0;
  for (int i = 0; i < N; i++) {
    sigma2 += pow2(input[i * C + col] - mu);
  }
  sigma2 /= N;

  for (int i = 0; i < N; i++) {
    output[i * C + col] =
        gamma[col] * (input[i * C + col] - mu) / sqrt(sigma2 + eps) + beta[col];
  }
}

// input, gamma, beta, output are device pointers
extern "C" void solve(const float *input, // (N, C)
                      const float *gamma, // (C,)
                      const float *beta,  // (C,)
                      float *output,      // (N, C)
                      int N, int C, float eps) {
  // mu, sigma = f(input)
  // xhat = norm(input)
  // y = gamma * xhat + beta
  bn_kernel<<<(C + 255) / 256, 256>>>(input, gamma, beta, output, N, C, eps);
}
