#include <cuda_runtime.h>

__device__ float pow2(float x) { return x * x; }

__global__ void nn_kernel(const float *__restrict__ points, // (N, 3)
                          int *__restrict__ indices,        // (N,)
                          int N) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
    return;

  float point[3];
  point[0] = points[tid * 3];
  point[1] = points[tid * 3 + 1];
  point[2] = points[tid * 3 + 2];

  float min_dist = MAXFLOAT;
  int min_index = -1;
  for (int i = 0; i < N; i++) {
    float dist = pow2(point[0] - points[i * 3]) +
                 pow2(point[1] - points[i * 3 + 1]) +
                 pow2(point[2] - points[i * 3 + 2]);
    if (i != tid && dist < min_dist) {
      min_dist = dist;
      min_index = i;
    }
  }

  indices[tid] = min_index;
}

// Nearest Neighbor
// points and indices are device pointers
extern "C" void solve(const float *points, // (N, 3)
                      int *indices,        // (N,)
                      int N) {
  nn_kernel<<<(N + 255) / 256, 256>>>(points, indices, N);
}
