#include <cuda_runtime.h>

#define add_during_load_scaling 8
#define BLOCKSIZE 256

// __device__ void warp_unroll(volatile float* sdata, int tid) {
//     // 把32以下的都unroll
//     sdata[tid] += sdata[tid + 32];
//     sdata[tid] += sdata[tid + 16];
//     sdata[tid] += sdata[tid + 8];
//     sdata[tid] += sdata[tid + 4];
//     sdata[tid] += sdata[tid + 2];
//     sdata[tid] += sdata[tid + 1];
// }

__global__ void reduct_kernel(const float *input, float *output, int N) {
  __shared__ float smem[BLOCKSIZE];
  // extern __shared__ float smem[];

  int tid = threadIdx.x;
  // int global_idx =(blockDim.x * blockIdx.x * add_during_load_scaling)+ tid;

  // 每個 Block 的起跑線
  int block_offset = blockDim.x * blockIdx.x * add_during_load_scaling;

  float sum = 0.0f;

  // 根據 scaling 自動展開加法 (編譯器會把這個常數次數的 for 迴圈直接 Unroll 掉)
  for (int step = 0; step < add_during_load_scaling; ++step) {
    int target_idx = block_offset + (step * blockDim.x) + tid;
    sum += (target_idx < N) ? input[target_idx] : 0.0f;
  }

  // 把算好的 4 份總和寫進白板
  smem[tid] = sum;
  __syncthreads();

  //  Unroll the iterations with operation in a Warp

  // complete unroll
  // for (int s = blockDim.x / 2; s > 32; s >>= 1) {
  //     if (tid < s) {
  //         smem[tid] += smem[tid + s];
  //     }
  //     __syncthreads();
  // }

  if (BLOCKSIZE >= 512)
    if (tid < 256) {
      smem[tid] += smem[tid + 256];
      __syncthreads();
    }
  if (BLOCKSIZE >= 256)
    if (tid < 128) {
      smem[tid] += smem[tid + 128];
      __syncthreads();
    }
  if (BLOCKSIZE >= 128)
    if (tid < 64) {
      smem[tid] += smem[tid + 64];
      __syncthreads();
    }

  // // 剩下的在這邊處理
  // if (tid < 32) {
  //     warp_unroll(smem, tid);
  // }

  // if (tid == 0) {
  //     atomicAdd(output, smem[0]);
  // }

  // 剩下的 32 個元素，使用現代的 Warp Shuffle 黑科技處理
  if (tid < 32) {
    // 1. 把 Shared Memory 裡的最終局部結果，讀進 Thread 自己私有的暫存器裡

    float val = smem[tid] + smem[tid + 32];

    // 2. 暫存器級別的隔空交換！(不再寫回 smem)
    // 0xffffffff 代表 Warp 內的 32 個人全部參與這場交換
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    // 3. 結帳：現在 Thread 0 手上的 val 就是整個 Block 的總和了
    // 我們直接把這個暫存器裡的 val 寫回 Global Memory，完全不用再碰 smem[0]！
    if (tid == 0) {
      atomicAdd(output, val);
    }
  }
}

// input, output are device pointers
extern "C" void reduce_stellar(const float *input, float *output, int N) {
  cudaMemset(output, 0, sizeof(float));
  int block_size = BLOCKSIZE;
  int new_block_capacity = block_size * add_during_load_scaling;
  int grid_size = (N + new_block_capacity - 1) / new_block_capacity;
  // size_t smem_size = block_size * sizeof(float);

  // reduct_kernel<<<grid_size, block_size, smem_size>>> (input, output, N);
  reduct_kernel<<<grid_size, block_size>>>(input, output, N);
}
