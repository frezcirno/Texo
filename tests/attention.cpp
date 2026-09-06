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

extern "C" void solve(const float *Q, const float *K, const float *V,
                      float *output, int M, int N, int d);

struct ValidationResult {
  bool passed = false;
  bool all_finite = true;
  double max_absolute_error = 0.0;
  double max_relative_error = 0.0;
  double rmse = 0.0;
};

static std::vector<float> reference_attention(const std::vector<float> &Q,
                                              const std::vector<float> &K,
                                              const std::vector<float> &V,
                                              int M, int N, int d) {
  std::vector<double> accumulator(static_cast<size_t>(M) * d, 0.0);
  std::vector<double> scores(static_cast<size_t>(N));
  const double scale = 1.0 / std::sqrt(static_cast<double>(d));

  for (int row = 0; row < M; ++row) {
    double maximum = -INFINITY;
    for (int column = 0; column < N; ++column) {
      double dot = 0.0;
      for (int k = 0; k < d; ++k) {
        dot += static_cast<double>(Q[static_cast<size_t>(row) * d + k]) *
               K[static_cast<size_t>(column) * d + k];
      }
      scores[column] = dot * scale;
      maximum = std::max(maximum, scores[column]);
    }

    double total = 0.0;
    for (int column = 0; column < N; ++column) {
      scores[column] = std::exp(scores[column] - maximum);
      total += scores[column];
    }

    for (int column = 0; column < N; ++column) {
      const double probability = scores[column] / total;
      for (int k = 0; k < d; ++k) {
        accumulator[static_cast<size_t>(row) * d + k] +=
            probability * V[static_cast<size_t>(column) * d + k];
      }
    }
  }

  std::vector<float> output(accumulator.size());
  for (size_t i = 0; i < accumulator.size(); ++i)
    output[i] = static_cast<float>(accumulator[i]);
  return output;
}

static ValidationResult validate(const std::vector<float> &actual,
                                 const std::vector<float> &reference) {
  ValidationResult result;
  double squared_error = 0.0;

  for (size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i])) {
      result.all_finite = false;
      continue;
    }

    const double absolute_error =
        std::fabs(static_cast<double>(actual[i]) - reference[i]);
    const double relative_error =
        absolute_error / std::max(std::fabs(static_cast<double>(reference[i])),
                                  1.0e-6);
    result.max_absolute_error =
        std::max(result.max_absolute_error, absolute_error);
    result.max_relative_error =
        std::max(result.max_relative_error, relative_error);
    squared_error += absolute_error * absolute_error;
  }

  result.rmse = std::sqrt(squared_error / actual.size());
  constexpr double MAX_ABSOLUTE_ERROR = 5.0e-4;
  constexpr double MAX_RMSE = 1.0e-4;
  result.passed = result.all_finite &&
                  result.max_absolute_error <= MAX_ABSOLUTE_ERROR &&
                  result.rmse <= MAX_RMSE;
  return result;
}

static ValidationResult run_validation(const float *device_Q,
                                       const float *device_K,
                                       const float *device_V,
                                       float *device_output,
                                       const std::vector<float> &reference,
                                       int M, int N, int d,
                                       float initial_output) {
  const size_t output_bytes = static_cast<size_t>(M) * d * sizeof(float);
  std::vector<float> initial(static_cast<size_t>(M) * d, initial_output);
  CUDA_CHECK(cudaMemcpy(device_output, initial.data(), output_bytes,
                        cudaMemcpyHostToDevice));

  solve(device_Q, device_K, device_V, device_output, M, N, d);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  std::vector<float> actual(static_cast<size_t>(M) * d);
  CUDA_CHECK(cudaMemcpy(actual.data(), device_output, output_bytes,
                        cudaMemcpyDeviceToHost));
  return validate(actual, reference);
}

static void benchmark(const float *device_Q, const float *device_K,
                      const float *device_V, float *device_output, int M, int N,
                      int d, int warmup, int repeat) {
  const size_t output_bytes = static_cast<size_t>(M) * d * sizeof(float);

  for (int i = 0; i < warmup; ++i) {
    CUDA_CHECK(cudaMemset(device_output, 0, output_bytes));
    solve(device_Q, device_K, device_V, device_output, M, N, d);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  double gpu_milliseconds = 0.0;
  double wall_milliseconds = 0.0;
  for (int i = 0; i < repeat; ++i) {
    CUDA_CHECK(cudaMemset(device_output, 0, output_bytes));
    const auto wall_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(start_event));
    solve(device_Q, device_K, device_V, device_output, M, N, d);
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
  const double operations = 4.0 * M * N * d;
  const double tflops = operations / (gpu_milliseconds * 1.0e-3) / 1.0e12;

  std::printf("  GPU event latency  = %.6f ms\n", gpu_milliseconds);
  std::printf("  end-to-end latency = %.6f ms\n", wall_milliseconds);
  std::printf("  approximate TFLOP/s = %.4f\n", tflops);

  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));
}

int main(int argc, char **argv) {
  int M = 128;
  int N = 256;
  int d = 64;
  int repeat = 20;
  int warmup = 5;

  if (argc > 1)
    M = std::atoi(argv[1]);
  if (argc > 2)
    N = std::atoi(argv[2]);
  if (argc > 3)
    d = std::atoi(argv[3]);
  if (argc > 4)
    repeat = std::atoi(argv[4]);
  if (argc > 5)
    warmup = std::atoi(argv[5]);

  if (M <= 0 || N <= 0 || d <= 0 || repeat <= 0 || warmup < 0) {
    std::fprintf(stderr,
                 "Usage: %s [M] [N] [d] [positive-repeat] "
                 "[nonnegative-warmup]\n",
                 argv[0]);
    return EXIT_FAILURE;
  }

  const size_t q_elements = static_cast<size_t>(M) * d;
  const size_t kv_elements = static_cast<size_t>(N) * d;
  const size_t output_elements = static_cast<size_t>(M) * d;

  std::mt19937 rng(0xC0FFEE);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  std::vector<float> host_Q(q_elements);
  std::vector<float> host_K(kv_elements);
  std::vector<float> host_V(kv_elements);
  for (float &value : host_Q)
    value = distribution(rng);
  for (float &value : host_K)
    value = distribution(rng);
  for (float &value : host_V)
    value = distribution(rng);

  const std::vector<float> reference =
      reference_attention(host_Q, host_K, host_V, M, N, d);

  float *device_Q = nullptr;
  float *device_K = nullptr;
  float *device_V = nullptr;
  float *device_output = nullptr;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc(&device_Q, q_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_K, kv_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_V, kv_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, output_elements * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_Q, host_Q.data(), q_elements * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_K, host_K.data(), kv_elements * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_V, host_V.data(), kv_elements * sizeof(float),
                        cudaMemcpyHostToDevice));

  const ValidationResult zero_initialized =
      run_validation(device_Q, device_K, device_V, device_output, reference, M,
                     N, d, 0.0f);
  const ValidationResult overwrite_check =
      run_validation(device_Q, device_K, device_V, device_output, reference, M,
                     N, d, 1.0f);

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  std::printf("CUDA attention benchmark\n");
  std::printf("  device             = %s\n", properties.name);
  std::printf("  Q                  = (%d, %d)\n", M, d);
  std::printf("  K, V               = (%d, %d)\n", N, d);
  std::printf("  repeats            = %d\n", repeat);
  std::printf("  warmup             = %d\n", warmup);
  std::printf("  zero-init result   = %s\n",
              zero_initialized.passed ? "PASS" : "FAIL");
  std::printf("  overwrite result   = %s\n",
              overwrite_check.passed ? "PASS" : "FAIL");
  std::printf("  all finite         = %s\n",
              zero_initialized.all_finite ? "yes" : "no");
  std::printf("  max absolute error = %.6e\n",
              zero_initialized.max_absolute_error);
  std::printf("  max relative error = %.6e\n",
              zero_initialized.max_relative_error);
  std::printf("  RMSE               = %.6e\n",
              zero_initialized.rmse);
  benchmark(device_Q, device_K, device_V, device_output, M, N, d, warmup,
            repeat);

  CUDA_CHECK(cudaFree(device_Q));
  CUDA_CHECK(cudaFree(device_K));
  CUDA_CHECK(cudaFree(device_V));
  CUDA_CHECK(cudaFree(device_output));
  return zero_initialized.passed && overwrite_check.passed ? EXIT_SUCCESS
                                                           : EXIT_FAILURE;
}
