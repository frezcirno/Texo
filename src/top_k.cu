#include <cuda_runtime.h>

namespace {
constexpr int THREADS = 256;
constexpr int SORT_TILE = 1024;

// For non-NaN floats, larger keys mean larger values (including negatives).
__device__ unsigned float_key(float value) {
  unsigned bits = __float_as_uint(value);
  return bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}

__device__ float key_float(unsigned key) {
  unsigned bits = key ^ ((key & 0x80000000u) ? 0x80000000u : 0xffffffffu);
  return __uint_as_float(bits);
}

struct Selection {
  unsigned prefix;
  unsigned mask;
  unsigned rank; // One-based rank within the currently selected prefix.
  unsigned written;
};

// Count the next byte only for values matching previously selected bytes.
__global__ void byte_histogram(const float *input, int n, int shift,
                               const Selection *selection, unsigned *counts) {
  __shared__ unsigned local[256];
  int tid = threadIdx.x;
  int lane = tid & 31;
  local[tid] = 0;
  __syncthreads();
  unsigned mask = selection->mask;
  unsigned prefix = selection->prefix;
  for (int base = blockIdx.x * THREADS; base < n;
       base += gridDim.x * THREADS) {
    int i = base + tid;
    unsigned key = i < n ? float_key(input[i]) : 0;
    bool eligible = i < n && (key & mask) == prefix;
    unsigned active = __ballot_sync(0xffffffffu, eligible);
    if (eligible) {
      unsigned digit = (key >> shift) & 255u;
      // Combine identical digits within each warp before the shared atomic.
      // __match_any_sync is supported on T4 (sm_75) and newer GPUs.
      unsigned peers = __match_any_sync(active, digit);
      if (lane == __ffs(peers) - 1)
        atomicAdd(local + digit, unsigned(__popc(peers)));
    }
  }
  __syncthreads();
  if (local[tid]) atomicAdd(counts + tid, local[tid]);
}

__global__ void choose_byte(const unsigned *counts, int shift,
                            Selection *selection) {
  unsigned rank = selection->rank;
  for (int digit = 255; digit >= 0; --digit) {
    if (rank > counts[digit]) {
      rank -= counts[digit];
    } else {
      selection->prefix |= unsigned(digit) << shift;
      selection->mask |= 255u << shift;
      selection->rank = rank;
      break;
    }
  }
}

// Fill with the kth value first; strictly larger values overwrite the front.
// This keeps exactly k values even when the threshold occurs many times.
__global__ void fill_threshold(float *output, int k, const Selection *selection) {
  float threshold = key_float(selection->prefix);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k;
       i += gridDim.x * blockDim.x)
    output[i] = threshold;
}

__global__ void collect_larger(const float *input, float *output, int n,
                               Selection *selection) {
  int lane = threadIdx.x & 31;
  unsigned threshold = selection->prefix;
  for (int base = blockIdx.x * THREADS; base < n;
       base += gridDim.x * THREADS) {
    int i = base + threadIdx.x;
    float value = i < n ? input[i] : 0.0f;
    bool eligible = i < n && float_key(value) > threshold;
    unsigned active = __ballot_sync(0xffffffffu, eligible);
    if (active) {
      int leader = __ffs(active) - 1;
      unsigned offset = 0;
      if (lane == leader)
        offset = atomicAdd(&selection->written, unsigned(__popc(active)));
      offset = __shfl_sync(0xffffffffu, offset, leader);
      if (eligible) {
        unsigned before = active & ((1u << lane) - 1u);
        output[offset + __popc(before)] = value;
      }
    }
  }
}

// Bitonic sort of at most 1024 keys per block. The measured k=100 fits one tile.
__global__ void sort_tiles(float *values, int n) {
  __shared__ unsigned keys[SORT_TILE];
  int start = blockIdx.x * SORT_TILE;
  int count = min(SORT_TILE, n - start);
  int padded = 1;
  while (padded < count) padded <<= 1;
  for (int i = threadIdx.x; i < padded; i += THREADS)
    keys[i] = i < count ? float_key(values[start + i]) : 0;
  __syncthreads();
  for (int size = 2; size <= padded; size <<= 1) {
    for (int stride = size / 2; stride > 0; stride >>= 1) {
      for (int i = threadIdx.x; i < padded; i += THREADS) {
        int other = i ^ stride;
        if (other > i) {
          bool descending = (i & size) == 0;
          unsigned a = keys[i], b = keys[other];
          if (descending ? a < b : a > b) {
            keys[i] = b;
            keys[other] = a;
          }
        }
      }
      __syncthreads();
    }
  }
  for (int i = threadIdx.x; i < count; i += THREADS)
    values[start + i] = key_float(keys[i]);
}

// For k > 1024, merge adjacent sorted runs. Ties put the left run first,
// so every element computes a distinct destination without atomics.
__global__ void merge_runs(const float *input, float *output, int n, int width) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    int start = (i / (2 * width)) * (2 * width);
    int middle = min(start + width, n);
    int end = min(start + 2 * width, n);
    bool left = i < middle;
    int other_start = left ? middle : start;
    int lo = other_start, hi = left ? end : middle;
    unsigned key = float_key(input[i]);
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2;
      unsigned other = float_key(input[mid]);
      if (left ? other > key : other >= key)
        lo = mid + 1;
      else
        hi = mid;
    }
    int position = start + (i - (left ? start : middle)) + (lo - other_start);
    output[position] = input[i];
  }
}
} // namespace

// Device pointers; input contains no NaNs; 1 <= k <= N.
// Output contains the largest k values in descending order; input is unchanged.
extern "C" void solve(const float *input, float *output, int N, int k) {
  if (N <= 0 || k <= 0 || k > N) return;
  int blocks = (N + THREADS - 1) / THREADS;
  blocks = blocks < 512 ? blocks : 512;
  Selection initial = {0, 0, unsigned(k), 0};
  Selection *selection;
  unsigned *counts;
  cudaMalloc(&selection, sizeof(Selection));
  cudaMalloc(&counts, 256 * sizeof(unsigned));
  cudaMemcpy(selection, &initial, sizeof(initial), cudaMemcpyHostToDevice);
  for (int shift = 24; shift >= 0; shift -= 8) {
    cudaMemsetAsync(counts, 0, 256 * sizeof(unsigned));
    byte_histogram<<<blocks, THREADS>>>(input, N, shift, selection, counts);
    choose_byte<<<1, 1>>>(counts, shift, selection);
  }
  fill_threshold<<<blocks, THREADS>>>(output, k, selection);
  collect_larger<<<blocks, THREADS>>>(input, output, N, selection);
  sort_tiles<<<(k + SORT_TILE - 1) / SORT_TILE, THREADS>>>(output, k);
  if (k > SORT_TILE) {
    float *scratch;
    cudaMalloc(&scratch, size_t(k) * sizeof(float));
    float *from = output, *to = scratch;
    for (int width = SORT_TILE; width < k; width *= 2) {
      merge_runs<<<blocks, THREADS>>>(from, to, k, width);
      float *swap = from;
      from = to;
      to = swap;
    }
    if (from != output)
      cudaMemcpyAsync(output, from, size_t(k) * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(scratch);
  }
  cudaFree(counts);
  cudaFree(selection);
}
