#include <cuda_runtime.h>

// 前提：N % 4 == 0 && T % 4 == 0
__global__ void dequant_rows(const float4 *__restrict__ X4, // (M, N4)
                             const float *__restrict__ S,   // (MT, NT)
                             float4 *__restrict__ Y4,       // (M, N4)
                             int N4, int T, int NT) {
  const int m = blockIdx.y;
  const float *S_base = S + (m / T) * NT; // 每个 block 只算一次
  const float4 *X_base = X4 + (size_t)m * N4;
  float4 *Y_base = Y4 + (size_t)m * N4;
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < N4;
       c += gridDim.x * blockDim.x) {
    const float s = __ldg(S_base + (c * 4) / T);
    float4 v = __ldcs(X_base + c);
    v.x *= s;
    v.y *= s;
    v.z *= s;
    v.w *= s;
    __stcs(Y_base + c, v);
  }
}

__global__ void generic(const float *__restrict__ X, // (M, N)
                        const float *__restrict__ S, // (MT, NT)
                        float *__restrict__ Y,       // (M, N)
                        const int N, const int T, const int NT) {
  const int m = blockIdx.y;
  const float *S_base = S + (m / T) * NT; // 每个 block 只算一次
  const float *X_base = X + (size_t)m * N;
  float *Y_base = Y + (size_t)m * N;
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < N;
       c += gridDim.x * blockDim.x) {
    const float s = __ldg(S_base + c / T);
    float v = __ldcs(X_base + c);
    __stcs(Y_base + c, v * s);
  }
}

// X, S, Y are device pointers
extern "C" void solve(const float *X, // (M, N)
                      const float *S, // (ceil(M/T), ceil(N/T))
                      float *Y,       // (M, N)
                      int M, int N, int T) {
  if (N % 4 == 0 && T % 4 == 0) {
    dequant_rows<<<dim3((N + 1023) / 1024, M), 256>>>(
        reinterpret_cast<const float4 *>(X), S, reinterpret_cast<float4 *>(Y),
        N / 4, T, (N + T - 1) / T);
    return;
  }
  generic<<<dim3((N + 255) / 256, M), 256>>>(X, S, Y, N, T, (N + T - 1) / T);
}
