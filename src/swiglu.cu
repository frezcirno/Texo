#include <cuda_runtime.h>

__global__ void swiglu_float4(const float *input1, const float *input2,
                              float *output, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
    return;
  const float4 *input14 = reinterpret_cast<const float4 *>(input1);
  const float4 *input24 = reinterpret_cast<const float4 *>(input2);
  float4 *output4 = reinterpret_cast<float4 *>(output);
  int N4 = N / 4;
  int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < N4; i += stride) {
    float4 x1 = input14[i];
    float4 x2 = input24[i];
    x2.x = x2.x * x1.x / (1 + __expf(-x1.x));
    x2.y = x2.y * x1.y / (1 + __expf(-x1.y));
    x2.z = x2.z * x1.z / (1 + __expf(-x1.z));
    x2.w = x2.w * x1.w / (1 + __expf(-x1.w));
    output4[i] = x2;
  }

  int tail = N4 * 4 + threadIdx.x;
  if (tail < N) {
    float x1 = input1[tail];
    float x2 = input2[tail];
    output[tail] = x2 * x1 / (1 + __expf(-x1));
  }
}

__global__ void swiglu(const float *input, float *output, int halfN) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= halfN)
    return;
  float x1 = input[tid];
  float x2 = input[halfN + tid];
  output[tid] = x2 * x1 / (1 + __expf(-x1));
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N) {
  int halfN = N / 2;
  int threadsPerBlock = 256;

  if (halfN % 4 == 0) {
    // input + halfN is 16-byte aligned only when halfN is a multiple of 4
    int N4 = halfN / 4;
    int blocksPerGrid = (N4 + threadsPerBlock - 1) / threadsPerBlock;
    swiglu_float4<<<blocksPerGrid, threadsPerBlock>>>(input, input + halfN,
                                                      output, halfN);
  } else {
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;
    swiglu<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
  }
}
