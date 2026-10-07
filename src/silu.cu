#include <cuda_runtime.h>

template <typename T> __host__ __device__ __forceinline__ T silu(T x) {
  return x / (1 + __expf(-x));
}

__global__ void silu_kernel(const float *__restrict__ input,
                            float *__restrict__ output, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
    return;

  int stride = gridDim.x * blockDim.x;

  const float4 *input4 = reinterpret_cast<const float4 *>(input);
  float4 *output4 = reinterpret_cast<float4 *>(output);
  int N4 = N / 4;

  for (int i = tid; i < N4; i += stride) {
    auto v = input4[i];
    v.x = silu(v.x);
    v.y = silu(v.y);
    v.z = silu(v.z);
    v.w = silu(v.w);
    output4[i] = v;
  }

  int tail = N4 * 4 + tid;
  if (tail < N) {
    output[tail] = silu(input[tail]);
  }
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (max(N / 4, 1) + threadsPerBlock - 1) / threadsPerBlock;

  silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
}
