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
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(err));                                 \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

extern "C" void solve(const float *input, const float *kernel, float *output,
                      int input_depth, int input_rows, int input_cols,
                      int kernel_depth, int kernel_rows, int kernel_cols);

struct ValidationResult {
  bool passed = false;
  bool all_finite = true;
  double max_absolute_error = 0.0;
  double max_relative_error = 0.0;
  double rmse = 0.0;
};

static size_t index3d(int depth, int row, int col, int rows, int cols) {
  return (static_cast<size_t>(depth) * rows + row) * cols + col;
}

static std::vector<float> reference_conv3d(
    const std::vector<float> &input, const std::vector<float> &kernel,
    int input_depth, int input_rows, int input_cols, int kernel_depth,
    int kernel_rows, int kernel_cols) {
  const int output_depth = input_depth - kernel_depth + 1;
  const int output_rows = input_rows - kernel_rows + 1;
  const int output_cols = input_cols - kernel_cols + 1;
  std::vector<float> output(static_cast<size_t>(output_depth) * output_rows *
                            output_cols);

  for (int depth = 0; depth < output_depth; ++depth) {
    for (int row = 0; row < output_rows; ++row) {
      for (int col = 0; col < output_cols; ++col) {
        double sum = 0.0;
        for (int kd = 0; kd < kernel_depth; ++kd) {
          for (int kr = 0; kr < kernel_rows; ++kr) {
            for (int kc = 0; kc < kernel_cols; ++kc) {
              sum += static_cast<double>(
                         input[index3d(depth + kd, row + kr, col + kc,
                                       input_rows, input_cols)]) *
                     kernel[index3d(kd, kr, kc, kernel_rows, kernel_cols)];
            }
          }
        }
        output[index3d(depth, row, col, output_rows, output_cols)] =
            static_cast<float>(sum);
      }
    }
  }
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
  constexpr double MAX_ABSOLUTE_ERROR = 1.0e-3;
  constexpr double MAX_RMSE = 2.0e-5;
  result.passed = result.all_finite &&
                  result.max_absolute_error <= MAX_ABSOLUTE_ERROR &&
                  result.rmse <= MAX_RMSE;
  return result;
}

static ValidationResult run_validation(
    const float *device_input, const float *device_kernel, float *device_output,
    const std::vector<float> &reference, int input_depth, int input_rows,
    int input_cols, int kernel_depth, int kernel_rows, int kernel_cols) {
  const size_t output_bytes = reference.size() * sizeof(float);

  // NaNs expose output coordinates that the launch failed to cover.
  CUDA_CHECK(cudaMemset(device_output, 0xff, output_bytes));
  solve(device_input, device_kernel, device_output, input_depth, input_rows,
        input_cols, kernel_depth, kernel_rows, kernel_cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> actual(reference.size());
  CUDA_CHECK(cudaMemcpy(actual.data(), device_output, output_bytes,
                        cudaMemcpyDeviceToHost));
  return validate(actual, reference);
}

static void benchmark(const float *device_input, const float *device_kernel,
                      float *device_output, int input_depth, int input_rows,
                      int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols, int repeat, int warmup) {
  for (int i = 0; i < warmup; ++i) {
    solve(device_input, device_kernel, device_output, input_depth, input_rows,
          input_cols, kernel_depth, kernel_rows, kernel_cols);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  double gpu_milliseconds = 0.0;
  double wall_milliseconds = 0.0;
  for (int i = 0; i < repeat; ++i) {
    const auto wall_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(start_event));
    solve(device_input, device_kernel, device_output, input_depth, input_rows,
          input_cols, kernel_depth, kernel_rows, kernel_cols);
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
  const int output_depth = input_depth - kernel_depth + 1;
  const int output_rows = input_rows - kernel_rows + 1;
  const int output_cols = input_cols - kernel_cols + 1;
  const double operations =
      2.0 * output_depth * output_rows * output_cols * kernel_depth *
      kernel_rows * static_cast<double>(kernel_cols);
  const double gflops = operations / (gpu_milliseconds * 1.0e-3) / 1.0e9;

  std::printf("  GPU event latency   = %.6f ms\n", gpu_milliseconds);
  std::printf("  end-to-end latency  = %.6f ms\n", wall_milliseconds);
  std::printf("  approximate GFLOP/s = %.3f\n", gflops);

  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));
}

int main(int argc, char **argv) {
  int input_depth = 17;
  int input_rows = 65;
  int input_cols = 97;
  int kernel_depth = 3;
  int kernel_rows = 5;
  int kernel_cols = 7;
  int repeat = 20;
  int warmup = 5;

  if (argc > 1)
    input_depth = std::atoi(argv[1]);
  if (argc > 2)
    input_rows = std::atoi(argv[2]);
  if (argc > 3)
    input_cols = std::atoi(argv[3]);
  if (argc > 4)
    kernel_depth = std::atoi(argv[4]);
  if (argc > 5)
    kernel_rows = std::atoi(argv[5]);
  if (argc > 6)
    kernel_cols = std::atoi(argv[6]);
  if (argc > 7)
    repeat = std::atoi(argv[7]);
  if (argc > 8)
    warmup = std::atoi(argv[8]);

  if (input_depth <= 0 || input_rows <= 0 || input_cols <= 0 ||
      kernel_depth <= 0 || kernel_rows <= 0 || kernel_cols <= 0 ||
      kernel_depth > input_depth || kernel_rows > input_rows ||
      kernel_cols > input_cols || repeat <= 0 || warmup < 0) {
    std::fprintf(stderr,
                 "Usage: %s [input-depth] [input-rows] [input-cols] "
                 "[kernel-depth] [kernel-rows] [kernel-cols] "
                 "[positive-repeat] [nonnegative-warmup]\n",
                 argv[0]);
    return EXIT_FAILURE;
  }

  const int output_depth = input_depth - kernel_depth + 1;
  const int output_rows = input_rows - kernel_rows + 1;
  const int output_cols = input_cols - kernel_cols + 1;
  const size_t input_elements =
      static_cast<size_t>(input_depth) * input_rows * input_cols;
  const size_t kernel_elements =
      static_cast<size_t>(kernel_depth) * kernel_rows * kernel_cols;
  const size_t output_elements =
      static_cast<size_t>(output_depth) * output_rows * output_cols;

  std::mt19937 rng(0xC03D);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  std::vector<float> host_input(input_elements);
  std::vector<float> host_kernel(kernel_elements);
  for (float &value : host_input)
    value = distribution(rng);
  for (float &value : host_kernel)
    value = distribution(rng);
  const std::vector<float> reference = reference_conv3d(
      host_input, host_kernel, input_depth, input_rows, input_cols,
      kernel_depth, kernel_rows, kernel_cols);

  float *device_input = nullptr;
  float *device_kernel = nullptr;
  float *device_output = nullptr;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc(&device_input, input_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_kernel, kernel_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, output_elements * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, host_input.data(),
                        input_elements * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_kernel, host_kernel.data(),
                        kernel_elements * sizeof(float), cudaMemcpyHostToDevice));

  const ValidationResult validation = run_validation(
      device_input, device_kernel, device_output, reference, input_depth,
      input_rows, input_cols, kernel_depth, kernel_rows, kernel_cols);

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  std::printf("CUDA conv3d benchmark\n");
  std::printf("  device              = %s\n", properties.name);
  std::printf("  layout              = (depth, row, col)\n");
  std::printf("  input               = (%d, %d, %d)\n", input_depth,
              input_rows, input_cols);
  std::printf("  kernel              = (%d, %d, %d)\n", kernel_depth,
              kernel_rows, kernel_cols);
  std::printf("  output              = (%d, %d, %d)\n", output_depth,
              output_rows, output_cols);
  std::printf("  repeats             = %d\n", repeat);
  std::printf("  warmup              = %d\n", warmup);
  std::printf("  correctness         = %s\n",
              validation.passed ? "PASS" : "FAIL");
  std::printf("  all finite          = %s\n",
              validation.all_finite ? "yes" : "no");
  std::printf("  max absolute error  = %.6e\n",
              validation.max_absolute_error);
  std::printf("  max relative error  = %.6e\n",
              validation.max_relative_error);
  std::printf("  RMSE                = %.6e\n", validation.rmse);
  benchmark(device_input, device_kernel, device_output, input_depth, input_rows,
            input_cols, kernel_depth, kernel_rows, kernel_cols, repeat, warmup);

  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_kernel));
  CUDA_CHECK(cudaFree(device_output));
  return validation.passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
