#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void solve(const float*, const float*, float*, int, int, int, int);
#define CUDA_CHECK(call) do { \
  cudaError_t error = (call); \
  if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

// Contract: row-major arrays, zero padding, no kernel flip; anchor=(KH/2,KW/2).
// The suite includes odd/even kernels and empty images.
enum Pattern { RandomGaussian, Constant, Impulse, Asymmetric, Zero };
static const char* names[] = {"random-gaussian", "constant", "impulse", "asymmetric", "zero"};

static bool run_case(int h, int w, int kh, int kw, Pattern pattern) {
  size_t count = size_t(h) * w, taps = size_t(kh) * kw;
  std::vector<float> input(count), kernel(taps), output(count + 2);
  std::vector<double> reference(count, 0.0);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  for (float& v : input) v = dist(rng);
  if (pattern == Constant) std::fill(input.begin(), input.end(), 1.0f);
  if (pattern == Zero || pattern == Impulse) std::fill(input.begin(), input.end(), 0.0f);
  if (pattern == Impulse && count) input[size_t(h / 2) * w + w / 2] = 1.0f;
  double weight_sum = 0;
  for (int y = 0; y < kh; ++y) for (int x = 0; x < kw; ++x) {
    double dy = y - kh / 2, dx = x - kw / 2;
    float v = pattern == Asymmetric ? dist(rng) : float(std::exp(-(dx*dx + dy*dy) / 8.0));
    kernel[size_t(y) * kw + x] = v;
    weight_sum += v;
  }
  if (pattern != Asymmetric) for (float& v : kernel) v = float(v / weight_sum);

  // Independent CPU double accumulation of the actual float inputs/weights.
  for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
    double sum = 0;
    for (int ky = 0; ky < kh; ++ky) for (int kx = 0; kx < kw; ++kx) {
      int iy = y + ky - kh / 2, ix = x + kx - kw / 2;
      if (iy >= 0 && iy < h && ix >= 0 && ix < w)
        sum += double(input[size_t(iy) * w + ix]) * kernel[size_t(ky) * kw + kx];
    }
    reference[size_t(y) * w + x] = sum;
  }

  float *di, *dk, *dout;
  CUDA_CHECK(cudaMalloc(&di, std::max(size_t(1), count) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dk, taps * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, output.size() * sizeof(float)));
  if (count) CUDA_CHECK(cudaMemcpy(di, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, kernel.data(), taps * sizeof(float), cudaMemcpyHostToDevice));
  std::fill(output.begin(), output.end(), std::numeric_limits<float>::quiet_NaN());
  output.front() = 123.0f;
  output.back() = -123.0f;
  CUDA_CHECK(cudaMemcpy(dout, output.data(), output.size() * sizeof(float), cudaMemcpyHostToDevice));

  bool passed = true;
  double max_error = 0;
  for (int repeat = 0; repeat < 2; ++repeat) {
    solve(di, dk, dout + 1, h, w, kh, kw);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), dout, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (output.front() != 123.0f || output.back() != -123.0f) {
      std::fprintf(stderr, "Output guard overwritten\n");
      passed = false;
    }
    size_t bad = 0;
    for (size_t i = 0; i < count; ++i) {
      double error = std::abs(double(output[i + 1]) - reference[i]);
      max_error = std::max(max_error, error);
      if (!std::isfinite(output[i + 1]) || error > 1e-5 + 1e-5 * std::abs(reference[i])) {
        if (bad++ == 0) std::fprintf(stderr, "call=%d row=%zu col=%zu got=%.9g expected=%.12g\n",
                                    repeat + 1, i / w, i % w, output[i + 1], reference[i]);
      }
    }
    if (bad) { std::fprintf(stderr, "mismatches=%zu/%zu\n", bad, count); passed = false; }
  }
  std::printf("%s image=%dx%d kernel=%dx%d %-15s max_abs_error=%.3g\n",
              passed ? "PASS" : "FAIL", h, w, kh, kw, names[pattern], max_error);
  CUDA_CHECK(cudaFree(di));
  CUDA_CHECK(cudaFree(dk));
  CUDA_CHECK(cudaFree(dout));
  return passed;
}

int main(int argc, char** argv) {
  if (argc != 1 && argc != 5) {
    std::fprintf(stderr, "Usage: %s [rows cols kernel_rows kernel_cols]\n", argv[0]);
    return 2;
  }
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s\nZero padding, no kernel flip; CPU double reference; atol=1e-5 rtol=1e-5\n", prop.name);
  bool passed = true;
  if (argc == 5) {
    try {
      auto parse = [](const char* text) {
        size_t end;
        long long v = std::stoll(text, &end);
        if (text[end] || v < 0 || v > INT_MAX / 2) throw std::invalid_argument("dimension out of range");
        return int(v);
      };
      int h = parse(argv[1]), w = parse(argv[2]), kh = parse(argv[3]), kw = parse(argv[4]);
      if (!kh || !kw || size_t(h)*w > INT_MAX || size_t(kh)*kw > INT_MAX)
        throw std::invalid_argument("positive kernel dimensions and int-sized arrays required");
      passed = run_case(h, w, kh, kw, RandomGaussian);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "Invalid arguments: %s\n", e.what());
      return 2;
    }
  } else {
    const int shapes[][4] = {
      {0,5,3,3}, {5,0,3,3}, {3,5,2,2}, {17,9,2,4}, {1,1,1,1}, {1,1,3,3}, {3,5,3,3}, {31,7,3,3}, {32,8,3,3},
      {33,9,5,5}, {1,37,1,5}, {37,1,5,1}, {5,3,7,9},
      {65,17,3,5}, {17,65,5,3}, {127,129,7,7}, {256,257,3,3}
    };
    for (auto& s : shapes)
      for (Pattern pattern : {RandomGaussian, Constant, Impulse, Asymmetric, Zero})
        passed &= run_case(s[0], s[1], s[2], s[3], pattern);
  }
  std::puts(passed ? "ALL PASSED" : "FAILED");
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
