#include <cstdint>
#include <cuda_runtime.h>

template <size_t BLOCK_SIZE>
__global__ void invert_kernel(uint32_t *image, int N) {
  const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const int stride = gridDim.x * BLOCK_SIZE;

  const int N4 = N / 4;
  uint4 *image4 = reinterpret_cast<uint4 *>(image);
  for (int i = tid; i < N4; i += stride) {
    uint4 v = image4[i];
    v.x ^= 0x00FFFFFFu;
    v.y ^= 0x00FFFFFFu;
    v.z ^= 0x00FFFFFFu;
    v.w ^= 0x00FFFFFFu;
    image4[i] = v;
  }

  const int tail = N4 * 4 + tid;
  if (tail < N) {
    image[tail] ^= 0x00FFFFFFu;
  }
}

// image_input, image_output are device pointers (i.e. pointers to memory on the
// GPU)
extern "C" void solve(unsigned char *image, int width, int height) {
  const int N = width * height;
  invert_kernel<256><<<(max(N / 4, 1) + 255) / 256, 256>>>(
      reinterpret_cast<uint32_t *>(image), N);
}
