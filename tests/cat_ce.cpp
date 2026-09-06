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

extern "C" void solve(const float*, const int*, float*, int, int);

#define CUDA_CHECK(call) do { \
  cudaError_t error = (call); \
  if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

enum Pattern { Random, Uniform, Confident, Wrong };

static bool run_case(int n, int c, Pattern pattern, float shift) {
  std::vector<float> logits(size_t(n) * c);
  std::vector<int> labels(n);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  std::uniform_int_distribution<int> label_dist(0, c - 1);
  double reference = 0.0;
  for (int row = 0; row < n; ++row) {
    labels[row] = label_dist(rng);
    size_t base = size_t(row) * c;
    for (int col = 0; col < c; ++col) {
      float v = pattern == Random ? dist(rng) : 0.0f;
      if (pattern == Confident || pattern == Wrong) {
        int winner = pattern == Confident ? labels[row] : (labels[row] + 1) % c;
        v = col == winner ? 1000.0f : -1000.0f;
      }
      logits[base + col] = v + shift;
    }
    double max_logit = -INFINITY;
    for (int col = 0; col < c; ++col)
      max_logit = std::max(max_logit, double(logits[base + col]));
    double sum = 0.0;
    for (int col = 0; col < c; ++col)
      sum += std::exp(double(logits[base + col]) - max_logit);
    reference += std::log(sum) + (max_logit - logits[base + labels[row]]);
  }
  reference /= n;

  float *device_logits, *device_output;
  int* device_labels;
  CUDA_CHECK(cudaMalloc(&device_logits, logits.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_labels, labels.size() * sizeof(int)));
  // Guard elements before and after the single scalar output.
  CUDA_CHECK(cudaMalloc(&device_output, 3 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_labels, labels.data(), labels.size() * sizeof(int), cudaMemcpyHostToDevice));
  float output[3] = {123.0f, 456.0f, -123.0f};
  CUDA_CHECK(cudaMemcpy(device_output, output, sizeof(output), cudaMemcpyHostToDevice));

  bool passed = true;
  double max_error = 0.0;
  // No resetting between calls: catches missing initialization of atomic output.
  for (int repeat = 0; repeat < 3; ++repeat) {
    solve(device_logits, device_labels, device_output + 1, n, c);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output, device_output, sizeof(output), cudaMemcpyDeviceToHost));
    double error = std::abs(double(output[1]) - reference);
    max_error = std::max(max_error, error);
    if (!std::isfinite(output[1]) || error > 1e-5 + 1e-5 * std::abs(reference) ||
        output[0] != 123.0f || output[2] != -123.0f) {
      std::fprintf(stderr, "FAIL N=%d C=%d pattern=%d shift=%g call=%d got=%.9g expected=%.12g guards=%g,%g\n",
                   n, c, int(pattern), shift, repeat + 1, output[1], reference, output[0], output[2]);
      passed = false;
    }
  }
  std::printf("%s N=%d C=%d pattern=%d shift=%g loss=%.8g max_abs_error=%.3g\n",
              passed ? "PASS" : "FAIL", n, c, int(pattern), shift, output[1], max_error);
  CUDA_CHECK(cudaFree(device_logits));
  CUDA_CHECK(cudaFree(device_labels));
  CUDA_CHECK(cudaFree(device_output));
  return passed;
}

static bool empty_case() {
  // Preserve current solve contract: N=0 is a no-op, not a defined mean loss.
  float* output;
  CUDA_CHECK(cudaMalloc(&output, sizeof(float)));
  float value = 123.0f;
  CUDA_CHECK(cudaMemcpy(output, &value, sizeof(float), cudaMemcpyHostToDevice));
  solve(nullptr, nullptr, output, 0, 3);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(&value, output, sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(output));
  bool passed = value == 123.0f;
  std::printf("%s N=0 no-op (output unchanged)\n", passed ? "PASS" : "FAIL");
  return passed;
}

int main(int argc, char** argv) {
  if (argc != 1 && argc != 3) {
    std::fprintf(stderr, "Usage: %s [N C]\n", argv[0]);
    return 2;
  }
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s\nMean categorical cross entropy; CPU double reference; atol=1e-5 rtol=1e-5\n", prop.name);
  std::puts("Patterns: 0=random, 1=uniform, 2=confident correct, 3=confident wrong; 3 calls per case.");
  bool passed = true;
  if (argc == 3) {
    try {
      auto parse = [](const char* text) {
        size_t end;
        long long v = std::stoll(text, &end);
        if (text[end] || v <= 0 || v > INT_MAX) throw std::invalid_argument("expected positive int");
        return int(v);
      };
      int n = parse(argv[1]), c = parse(argv[2]);
      if (size_t(n) * c > INT_MAX || n > INT_MAX - 255)
        throw std::invalid_argument("exceeds kernel int indexing");
      passed = run_case(n, c, Random, 0.0f);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "Invalid arguments: %s\n", e.what());
      return 2;
    }
  } else {
    passed &= empty_case();
    for (int n : {1, 8, 255, 256, 257, 1025})
      for (int c : {1, 3, 65})
        for (float shift : {0.0f, 1000.0f, -1000.0f})
          passed &= run_case(n, c, Random, shift);
    passed &= run_case(513, 17, Uniform, 1000.0f); // Expected log(17).
    passed &= run_case(257, 17, Confident, 0.0f); // Expected approximately 0.
    passed &= run_case(257, 17, Wrong, 0.0f);     // Expected approximately 2000.
    passed &= run_case(33, 4097, Random, 0.0f);   // Long class dimension.
    passed &= run_case(65537, 3, Random, 0.0f);  // Many blocks and partial tail.
  }
  std::puts(passed ? "ALL PASSED" : "FAILED");
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
