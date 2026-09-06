#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void solve(const float*, const float*, float*, int);

#define CUDA_CHECK(call) do { \
  cudaError_t error = (call); \
  if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

enum Pattern { Random, Identical, Constant, TailOnly };
static const char* names[] = {"random", "identical", "constant", "tail-only"};

static bool run_case(int n, Pattern pattern) {
  std::vector<float> predictions(n), targets(n);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
  double reference = 0;
  for (int i = 0; i < n; ++i) {
    float a = dist(rng), b = dist(rng);
    if (pattern == Identical) b = a;
    if (pattern == Constant) { a = 3.0f; b = 1.0f; }
    // A missing final element or block then changes the entire answer.
    if (pattern == TailOnly) { a = i == n - 1 ? 8.0f : 0.0f; b = 0.0f; }
    predictions[i] = a;
    targets[i] = b;
    double diff = double(a) - b;
    reference += diff * diff;
  }
  // Current solve contract explicitly returns zero for an empty input.
  if (n > 0) reference /= n;

  float *dp = nullptr, *dt = nullptr, *out = nullptr;
  if (n > 0) {
    CUDA_CHECK(cudaMalloc(&dp, size_t(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dt, size_t(n) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dp, predictions.data(), size_t(n) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dt, targets.data(), size_t(n) * sizeof(float), cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMalloc(&out, 3 * sizeof(float)));
  float values[3] = {123.0f, 456.0f, -123.0f};
  CUDA_CHECK(cudaMemcpy(out, values, sizeof(values), cudaMemcpyHostToDevice));

  bool passed = true;
  double max_error = 0;
  // Deliberately do not reset output between calls: catches atomic accumulation.
  for (int repeat = 0; repeat < 3; ++repeat) {
    solve(dp, dt, out + 1, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(values, out, sizeof(values), cudaMemcpyDeviceToHost));
    double error = std::abs(double(values[1]) - reference);
    max_error = std::max(max_error, error);
    double tolerance = 1e-7 + 1e-6 * std::abs(reference);
    if (!std::isfinite(values[1]) || error > tolerance ||
        values[0] != 123.0f || values[2] != -123.0f) {
      std::fprintf(stderr, "FAIL N=%d pattern=%s call=%d got=%.9g expected=%.12g guards=%g,%g\n",
                   n, names[pattern], repeat + 1, values[1], reference, values[0], values[2]);
      passed = false;
    }
  }
  std::printf("%s N=%d %-9s mse=%.9g max_abs_error=%.3g\n",
              passed ? "PASS" : "FAIL", n, names[pattern], values[1], max_error);
  if (dp) CUDA_CHECK(cudaFree(dp));
  if (dt) CUDA_CHECK(cudaFree(dt));
  CUDA_CHECK(cudaFree(out));
  return passed;
}

int main(int argc, char** argv) {
  if (argc > 2) {
    std::fprintf(stderr, "Usage: %s [N]\n", argv[0]);
    return 2;
  }
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s\nMSE CPU double reference; atol=1e-7 rtol=1e-6; 3 calls per case\n", prop.name);
  bool passed = true;
  if (argc == 2) {
    try {
      size_t end;
      long long n = std::stoll(argv[1], &end);
      if (argv[1][end] || n < 0 || n > INT_MAX - 255)
        throw std::invalid_argument("N must be in [0, INT_MAX-255]");
      for (Pattern pattern : {Random, Identical, Constant, TailOnly})
        passed &= run_case(int(n), pattern);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "Invalid argument: %s\n", e.what());
      return 2;
    }
  } else {
    for (int n : {0, 1, 8, 31, 32, 33, 255, 256, 257, 511, 512, 513, 1025, 65537, 1000000, 50000000})
      for (Pattern pattern : {Random, Identical, Constant, TailOnly})
        passed &= run_case(n, pattern);
  }
  std::puts(passed ? "ALL PASSED" : "FAILED");
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
