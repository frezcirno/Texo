#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void solve(const float*, float*, int, int);
#define CUDA_CHECK(call) do { \
  cudaError_t error = (call); \
  if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

static bool run_case(const char* name, const std::vector<float>& input, int k) {
  int n = int(input.size());
  std::vector<float> reference = input;
  if (k < n)
    std::nth_element(reference.begin(), reference.begin() + k, reference.end(),
                     std::greater<float>());
  reference.resize(k);
  std::sort(reference.begin(), reference.end(), std::greater<float>());
  float *di, *dout;
  CUDA_CHECK(cudaMalloc(&di, size_t(n) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, size_t(k + 2) * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(di, input.data(), size_t(n) * sizeof(float), cudaMemcpyHostToDevice));
  std::vector<float> output(k + 2);
  bool passed = true;
  double best_ms = std::numeric_limits<double>::infinity();
  for (int call = 0; call < 3; ++call) {
    std::fill(output.begin(), output.end(), std::numeric_limits<float>::quiet_NaN());
    output.front() = 12345.0f;
    output.back() = -12345.0f;
    CUDA_CHECK(cudaMemcpy(dout, output.data(), output.size() * sizeof(float), cudaMemcpyHostToDevice));
    auto start = std::chrono::steady_clock::now();
    solve(di, dout + 1, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    // First call warms up; wall time includes wrapper allocation and freeing.
    if (call > 0) best_ms = std::min(best_ms, ms);
    CUDA_CHECK(cudaMemcpy(output.data(), dout, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (output.front() != 12345.0f || output.back() != -12345.0f) {
      std::fprintf(stderr, "Output guard overwritten\n");
      passed = false;
    }
    for (int i = 0; i < k; ++i) {
      if (output[i + 1] != reference[i]) {
        std::fprintf(stderr, "call=%d rank=%d got=%.9g expected=%.9g\n",
                     call, i, output[i + 1], reference[i]);
        passed = false;
        break;
      }
    }
  }
  std::vector<float> after(n);
  CUDA_CHECK(cudaMemcpy(after.data(), di, size_t(n) * sizeof(float), cudaMemcpyDeviceToHost));
  if (after != input) {
    std::fprintf(stderr, "Input modified\n");
    passed = false;
  }
  CUDA_CHECK(cudaFree(di));
  CUDA_CHECK(cudaFree(dout));
  std::printf("%s %-12s N=%d k=%d best_wrapper_ms=%.3f\n",
              passed ? "PASS" : "FAIL", name, n, k, best_ms);
  std::fflush(stdout);
  return passed;
}

static std::vector<float> random_input(int n) {
  std::vector<float> input(n);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
  for (float& value : input) value = dist(rng);
  return input;
}

int main(int argc, char** argv) {
  if (argc != 1 && argc != 3) {
    std::fprintf(stderr, "Usage: %s [N k], 1 <= k <= N <= 100000000\n", argv[0]);
    return 2;
  }
  bool passed = true;
  if (argc == 3) {
    try {
      auto parse = [](const char* arg) {
        size_t used;
        std::string text(arg);
        long long value = std::stoll(text, &used);
        if (used != text.size() || value < 1 || value > 100000000)
          throw std::invalid_argument("expected a size in [1, 100000000]");
        return int(value);
      };
      int n = parse(argv[1]), k = parse(argv[2]);
      if (k > n) throw std::invalid_argument("k must not exceed N");
      passed = run_case("random", random_input(n), k);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "%s\n", error.what());
      return 2;
    }
  } else {
    passed &= run_case("duplicates", {3, 8, 2, 8, 5}, 3);
    passed &= run_case("negative", {-3, -8, -2, -8, -5}, 5);
    passed &= run_case("equal", std::vector<float>(4097, 7.0f), 2049);
    passed &= run_case("extremes", {-INFINITY, INFINITY, -0.0f, 0.0f,
        -1.0e30f, 1.0e30f, -1.0e-30f, 1.0e-30f}, 8);
    for (int n : {1, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 4097}) {
      auto input = random_input(n);
      passed &= run_case("random", input, std::min(n, 100));
      passed &= run_case("all", input, n);
    }
    auto input = random_input(65537);
    passed &= run_case("multi-merge", input, 5000);
    passed &= run_case("all", input, 65537);
  }
  std::puts(passed ? "ALL PASSED" : "FAILED");
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
