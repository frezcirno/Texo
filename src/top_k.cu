#include <cuda_runtime.h>

__device__ void block_argmax(float best_value, int best_index,
                             float *shared_values, int *shared_indices) {
  const int tid = threadIdx.x;

  shared_values[tid] = best_value;
  shared_indices[tid] = best_index;

  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      if (shared_values[tid + stride] > shared_values[tid]) {
        shared_values[tid] = shared_values[tid + stride];
        shared_indices[tid] = shared_indices[tid + stride];
      }
    }

    __syncthreads();
  }
}

__global__ void topk(const float *__restrict__ input, // (N,)
                     float *__restrict__ output,      // (k,)
                     float *__restrict__ work,        // (N,)
                     int N, int k) {
  __shared__ float shared_values[256];
  __shared__ int shared_indices[256];

  for (int rank = 0; rank < k; ++rank) {
    float best_value = -INFINITY;
    int best_index = -1;

    // ① 每个线程处理多个元素。
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
      if (work[i] > best_value) {
        best_value = work[i];
        best_index = i;
      }
    }

    // ② 将各线程的 (value, index) 归约。
    // 结果放到 shared_values[0] 和 shared_indices[0]。
    block_argmax(best_value, best_index, shared_values, shared_indices);

    // ③ 输出本轮最大值，排除它的位置。
    if (threadIdx.x == 0) {
      output[rank] = shared_values[0];
      work[shared_indices[0]] = -INFINITY;
    }

    // 等线程 0 修改完 work，再找下一个最大值。
    __syncthreads();
  }
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N, int k) {
  float *work;
  cudaMalloc(&work, N * sizeof(float));
  cudaMemcpy(work, input, N * sizeof(float), cudaMemcpyDeviceToDevice);
  topk<<<1, 256>>>(input, output, work, N, k);
  cudaFree(work);
}
