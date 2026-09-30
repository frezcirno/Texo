#include <cmath>
#include <cuda_runtime.h>

__device__ inline float warp_max(float val) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    val = max(val, __shfl_down_sync(0xffffffff, val, off));
  }
  return val;
}

__device__ inline float warp_sum(float value) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, off);
  }
  return value;
}

// 保留你已有的 warp_sum()
constexpr int THREADS = 256;
constexpr int KEYS_PER_TILE = 8; // 256 threads = 8 warps

__global__ void gqa_kernel(const float *__restrict__ Q,
                           const float *__restrict__ K,
                           const float *__restrict__ V, float *__restrict__ O,
                           int num_q_heads, int num_kv_heads, int S, int d) {
  const int q_head = blockIdx.x;
  const int q_pos = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const int group_size = num_q_heads / num_kv_heads;
  const int kv_head = q_head / group_size;

  const size_t head_stride = size_t(S) * d;
  const float *q = Q + (size_t(q_head) * S + q_pos) * d;
  const float *k_head = K + size_t(kv_head) * head_stride;
  const float *v_head = V + size_t(kv_head) * head_stride;
  float *out = O + (size_t(q_head) * S + q_pos) * d;

  __shared__ float q_smem[256];
  __shared__ float score[KEYS_PER_TILE];
  __shared__ float running_max, running_sum, rescale;

  if (tid < d)
    q_smem[tid] = q[tid];

  if (tid == 0) {
    running_max = -INFINITY;
    running_sum = 0.0f;
  }
  __syncthreads();

  float acc = 0.0f;
  const float scale = rsqrtf(float(d));

  for (int base = 0; base < S; base += KEYS_PER_TILE) {
    // 每个 warp 算一个 Q·K_j
    const int key = base + warp;
    float dot = 0.0f;

    if (key < S) {
      for (int col = lane; col < d; col += 32)
        dot += q_smem[col] * k_head[size_t(key) * d + col];
    }

    dot = warp_sum(dot);

    if (lane == 0)
      score[warp] = (key < S) ? dot * scale : -INFINITY;
    __syncthreads();

    // 在线、数值稳定 softmax
    if (tid == 0) {
      float tile_max = running_max;
      for (int j = 0; j < KEYS_PER_TILE && base + j < S; ++j)
        tile_max = fmaxf(tile_max, score[j]);

      rescale = expf(running_max - tile_max);
      float new_sum = running_sum * rescale;

      for (int j = 0; j < KEYS_PER_TILE && base + j < S; ++j)
        new_sum += expf(score[j] - tile_max);

      running_max = tile_max;
      running_sum = new_sum;
    }
    __syncthreads();

    // 每个线程负责一个输出维度
    if (tid < d) {
      float tile_value = 0.0f;
      for (int j = 0; j < KEYS_PER_TILE && base + j < S; ++j) {
        float w = expf(score[j] - running_max);
        tile_value += w * v_head[size_t(base + j) * d + tid];
      }
      acc = acc * rescale + tile_value;
    }
    __syncthreads();
  }

  if (tid < d)
    out[tid] = acc / running_sum;
}

extern "C" void solve(const float *Q, const float *K, const float *V,
                      float *output, int num_q_heads, int num_kv_heads,
                      int seq_len, int head_dim) {
  dim3 grid(num_q_heads, seq_len);
  gqa_kernel<<<grid, THREADS>>>(Q, K, V, output, num_q_heads, num_kv_heads,
                                seq_len, head_dim);
  cudaDeviceSynchronize();
}
