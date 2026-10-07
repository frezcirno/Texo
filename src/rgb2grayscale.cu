#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(const float3 *input, float *output,
                                        int N) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
    return;
  float3 v = input[tid];
  __stcs(output + tid, 0.299f * v.x + 0.587f * v.y + 0.114f * v.z);
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int width,
                      int height) {
  int total_pixels = width * height;
  int threadsPerBlock = 256;
  int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

  rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      reinterpret_cast<const float3 *>(input), output, total_pixels);
}
