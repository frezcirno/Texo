#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      std::exit(EXIT_FAILURE);                                              \
    }                                                                       \
  } while (0)

extern "C" void reduce_manual(const float *input, float *output, int N);
extern "C" void reduce_cg(const float *input, float *output, int N);
extern "C" void reduce_cub(const float *input, float *output, int N);

using ReduceFn = void (*)(const float *, float *, int);

static bool bench(const char *name, ReduceFn fn,
                  const float *device_input, float *device_output,
                  int64_t N, int repeat, float host_result) {
  cudaEvent_t start_event, stop_event;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  fn(device_input, device_output, static_cast<int>(N));
  CUDA_CHECK(cudaDeviceSynchronize());

  float milliseconds = 0.0f;
  for (int i = 0; i < repeat; ++i) {
    CUDA_CHECK(cudaEventRecord(start_event));
    fn(device_input, device_output, static_cast<int>(N));
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_event, stop_event));
    milliseconds += elapsed;
  }
  milliseconds /= repeat;
  float bandwidth_gb =
      (static_cast<double>(N) * sizeof(float)) / (milliseconds * 1e-3) / 1e9;

  float device_result = 0.0f;
  CUDA_CHECK(cudaMemcpy(&device_result, device_output, sizeof(float),
                        cudaMemcpyDeviceToHost));
  float error = std::fabs(device_result - host_result);
  float relative_error = error / (std::fabs(host_result) + 1e-9f);

  std::printf("[%s]\n", name);
  std::printf("  average latency    = %.4f ms\n", milliseconds);
  std::printf("  throughput         = %.4f GB/s\n", bandwidth_gb);
  std::printf("  device output sum  = %.6f\n", device_result);
  std::printf("  absolute error     = %.6e\n", error);
  std::printf("  relative error     = %.6e\n", relative_error);

  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));
  return std::isfinite(device_result) && error <= 1e-4f + 1e-5f * std::fabs(host_result);
}

int main(int argc, char **argv) {
  int64_t N = 1 << 24; // 16M elements by default
  int repeat = 20;

  if (argc > 1) {
    N = std::stoll(argv[1]);
  }
  if (argc > 2) {
    repeat = std::atoi(argv[2]);
  }

  if (N <= 0 || N > (1LL << 30) || repeat <= 0) {
    std::fprintf(stderr, "Expected 0 < N <= 2^30 and repeat > 0\n");
    return EXIT_FAILURE;
  }
  std::vector<float> host_input(N);
  for (int64_t i = 0; i < N; ++i) {
    host_input[i] = 1.0f;
  }

  float *device_input = nullptr;
  float *device_output = nullptr;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc(&device_input, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, host_input.data(), N * sizeof(float),
                        cudaMemcpyHostToDevice));

  float host_result = static_cast<float>(N); // Input contains only ones.

  std::printf("CUDA reduce benchmark\n");
  std::printf("  elements           = %lld\n", static_cast<long long>(N));
  std::printf("  repeats            = %d\n", repeat);
  std::printf("  host reference sum = %.6f\n", host_result);

  bool passed = true;
  passed &= bench("reduce_manual", reduce_manual, device_input, device_output, N, repeat,
        host_result);
  passed &= bench("reduce_cg", reduce_cg, device_input, device_output, N, repeat,
        host_result);
  passed &= bench("reduce_cub", reduce_cub, device_input, device_output, N, repeat,
        host_result);

  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
