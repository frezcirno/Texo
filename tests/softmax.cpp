#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(err));                                  \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                        \
  } while (0)

extern "C" void softmax_3kernel(const float *input, float *output, int N);
extern "C" void softmax_4kernel(const float *input, float *output, int N);

using SoftmaxFn = void (*)(const float *, float *, int);

struct Implementation {
  const char *name;
  SoftmaxFn function;
};

struct ValidationResult {
  bool passed = false;
  bool all_finite = true;
  double output_sum = 0.0;
  double max_absolute_error = 0.0;
  double max_relative_error = 0.0;
  double l1_error = 0.0;
};

static std::vector<float> reference_softmax(const std::vector<float> &input) {
  const float maximum = *std::max_element(input.begin(), input.end());

  double total = 0.0;
  for (float value : input) {
    total += std::exp(static_cast<double>(value) - maximum);
  }

  std::vector<float> output(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = static_cast<float>(
        std::exp(static_cast<double>(input[i]) - maximum) / total);
  }
  return output;
}

static ValidationResult validate(const std::vector<float> &actual,
                                 const std::vector<float> &reference) {
  ValidationResult result;

  for (size_t i = 0; i < actual.size(); ++i) {
    const double value = actual[i];
    const double expected = reference[i];
    if (!std::isfinite(value)) {
      result.all_finite = false;
      continue;
    }

    result.output_sum += value;
    const double absolute_error = std::fabs(value - expected);
    const double relative_error =
        absolute_error / std::max(std::fabs(expected), 1.0e-12);
    result.max_absolute_error =
        std::max(result.max_absolute_error, absolute_error);
    result.max_relative_error =
        std::max(result.max_relative_error, relative_error);
    result.l1_error += absolute_error;
  }

  constexpr double MAX_ABSOLUTE_ERROR = 5.0e-5;
  constexpr double MAX_L1_ERROR = 5.0e-3;
  constexpr double MAX_SUM_ERROR = 5.0e-3;
  result.passed = result.all_finite &&
                  result.max_absolute_error <= MAX_ABSOLUTE_ERROR &&
                  result.l1_error <= MAX_L1_ERROR &&
                  std::fabs(result.output_sum - 1.0) <= MAX_SUM_ERROR;
  return result;
}

static bool benchmark(const Implementation &implementation,
                      const float *device_input, float *device_output,
                      const std::vector<float> &reference, int N, int warmup,
                      int repeat) {
  for (int i = 0; i < warmup; ++i) {
    implementation.function(device_input, device_output, N);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  std::vector<float> actual(static_cast<size_t>(N));
  CUDA_CHECK(cudaMemset(device_output, 0xff,
                        static_cast<size_t>(N) * sizeof(float)));
  implementation.function(device_input, device_output, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(actual.data(), device_output,
                        static_cast<size_t>(N) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  const ValidationResult validation = validate(actual, reference);

  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  double gpu_milliseconds = 0.0;
  double wall_milliseconds = 0.0;
  for (int i = 0; i < repeat; ++i) {
    const auto wall_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(start_event));
    implementation.function(device_input, device_output, N);
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    const auto wall_stop = std::chrono::steady_clock::now();

    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_event, stop_event));
    gpu_milliseconds += elapsed;
    wall_milliseconds +=
        std::chrono::duration<double, std::milli>(wall_stop - wall_start)
            .count();
  }

  gpu_milliseconds /= repeat;
  wall_milliseconds /= repeat;
  const double giga_elements_per_second =
      static_cast<double>(N) / (gpu_milliseconds * 1.0e-3) / 1.0e9;

  std::printf("[%s] %s\n", implementation.name,
              validation.passed ? "PASS" : "FAIL");
  std::printf("  GPU event latency  = %.6f ms\n", gpu_milliseconds);
  std::printf("  end-to-end latency = %.6f ms\n", wall_milliseconds);
  std::printf("  throughput         = %.4f Gelem/s\n",
              giga_elements_per_second);
  std::printf("  output sum         = %.9f\n", validation.output_sum);
  std::printf("  max absolute error = %.6e\n",
              validation.max_absolute_error);
  std::printf("  max relative error = %.6e\n",
              validation.max_relative_error);
  std::printf("  L1 error           = %.6e\n", validation.l1_error);
  std::printf("  all finite         = %s\n\n",
              validation.all_finite ? "yes" : "no");

  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));
  return validation.passed;
}

int main(int argc, char **argv) {
  int N = 1 << 20;
  int repeat = 20;
  int warmup = 5;

  if (argc > 1)
    N = std::stoi(argv[1]);
  if (argc > 2)
    repeat = std::stoi(argv[2]);
  if (argc > 3)
    warmup = std::stoi(argv[3]);

  if (N <= 0 || repeat <= 0 || warmup < 0) {
    std::fprintf(stderr,
                 "Usage: %s [positive-N] [positive-repeat] "
                 "[nonnegative-warmup]\n",
                 argv[0]);
    return EXIT_FAILURE;
  }

  std::vector<float> host_input(static_cast<size_t>(N));
  std::mt19937 rng(0xC0FFEE);
  std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);

  // The large common offset catches implementations that fail to subtract the
  // maximum while keeping the reference softmax well conditioned.
  constexpr float INPUT_OFFSET = 1000.0f;
  for (float &value : host_input)
    value = INPUT_OFFSET + distribution(rng);

  const std::vector<float> reference = reference_softmax(host_input);

  float *device_input = nullptr;
  float *device_output = nullptr;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc(&device_input,
                        static_cast<size_t>(N) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output,
                        static_cast<size_t>(N) * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, host_input.data(),
                        static_cast<size_t>(N) * sizeof(float),
                        cudaMemcpyHostToDevice));

  const Implementation implementations[] = {
      {"softmax_3kernel", softmax_3kernel},
      {"softmax_4kernel", softmax_4kernel},
  };

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  std::printf("CUDA softmax benchmark\n");
  std::printf("  device             = %s\n", properties.name);
  std::printf("  elements           = %d\n", N);
  std::printf("  repeats            = %d\n", repeat);
  std::printf("  warmup             = %d\n\n", warmup);

  bool passed = true;
  for (const Implementation &implementation : implementations) {
    passed &= benchmark(implementation, device_input, device_output, reference, N,
              warmup, repeat);
  }

  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
