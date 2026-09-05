#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Include once so the benchmark can instantiate and compare the actual kernels.
#include "../src/mat-vec-mul.cu"

#define CUDA_CHECK(call) do { \
  cudaError_t error = (call); \
  if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

static const char* names[] = {"solve", "block256", "warp256", "warp128"};

static void launch(int version, const float* a, const float* x, float* y,
                   int m, int n) {
  if (version == 0) {
    solve(a, x, y, m, n, 0);
  } else if (m > 0) {
    if (version == 1)
      mat_vec_mul<256><<<m, 256>>>(a, x, y, m, n, 0);
    else {
      int threads = version == 2 ? 256 : 128;
      int warps = threads / 32;
      mat_vec_mul_warp<<<1 + (m - 1) / warps, threads>>>(a, x, y, m, n);
    }
  }
  CUDA_CHECK(cudaGetLastError());
}

static bool run_case(int m, int n, int repeats) {
  const size_t count = size_t(m) * n;
  std::vector<float> a(count), x(n), output(size_t(m) + 2);
  std::vector<double> reference(m, 0.0), magnitude(m, 0.0);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& v : a) v = dist(rng);
  for (float& v : x) v = dist(rng);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      double product = double(a[size_t(row) * n + col]) * x[col];
      reference[row] += product;
      magnitude[row] += std::abs(product);
    }
  }

  float *da, *dx, *dy;
  CUDA_CHECK(cudaMalloc(&da, std::max(size_t(1), count) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dx, std::max(1, n) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, output.size() * sizeof(float)));
  if (count) CUDA_CHECK(cudaMemcpy(da, a.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  if (n) CUDA_CHECK(cudaMemcpy(dx, x.data(), size_t(n) * sizeof(float), cudaMemcpyHostToDevice));

  bool passed = true;
  std::printf("M=%d N=%d\n", m, n);
  for (int version = 0; version < 4; ++version) {
    // Nonzero initial output catches accidental accumulation; canaries catch
    // writes just outside y. Call twice without resetting y to test overwrite.
    std::fill(output.begin(), output.end(), 12345.0f);
    CUDA_CHECK(cudaMemcpy(dy, output.data(), output.size() * sizeof(float), cudaMemcpyHostToDevice));
    double max_error = 0.0;
    bool correct = true;
    for (int invocation = 0; invocation < 2; ++invocation) {
      launch(version, da, dx, dy + 1, m, n);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(output.data(), dy, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
      if (output.front() != 12345.0f || output.back() != 12345.0f) correct = false;
      for (int row = 0; row < m; ++row) {
        double error = std::abs(double(output[row + 1]) - reference[row]);
        // Scale by sum(abs(products)) to handle cancellation near zero.
        double tolerance = 1e-5 + 2e-6 * magnitude[row];
        if (!std::isfinite(output[row + 1]) || error > tolerance) {
          if (correct) std::fprintf(stderr, "%s row=%d got=%.9g expected=%.12g tolerance=%.3g\n",
                                    names[version], row, output[row + 1], reference[row], tolerance);
          correct = false;
        }
        max_error = std::max(max_error, error);
      }
    }
    passed &= correct;
    std::printf("  %-8s %s max_abs_error=%.3g", names[version], correct ? "PASS" : "FAIL", max_error);
    if (correct && repeats > 0 && m > 0) {
      for (int i = 0; i < 10; ++i) launch(version, da, dx, dy + 1, m, n);
      CUDA_CHECK(cudaDeviceSynchronize());
      cudaEvent_t start, stop;
      CUDA_CHECK(cudaEventCreate(&start));
      CUDA_CHECK(cudaEventCreate(&stop));
      std::vector<float> times;
      for (int batch = 0; batch < 5; ++batch) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeats; ++i) launch(version, da, dx, dy + 1, m, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms * 1000.0f / repeats);
      }
      std::sort(times.begin(), times.end());
      std::printf(" median_batch_us=%.3f", times[2]);
      CUDA_CHECK(cudaEventDestroy(start));
      CUDA_CHECK(cudaEventDestroy(stop));
    }
    std::puts("");
  }
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dy));
  return passed;
}

int main(int argc, char** argv) {
  bool check_only = argc == 2 && std::string(argv[1]) == "--check-only";
  if (argc != 1 && !check_only && argc != 3 && argc != 4) {
    std::fprintf(stderr, "Usage: %s [M N [repeats]] | --check-only\n", argv[0]);
    return 2;
  }
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s\nCPU reference: double; tolerance: 1e-5 + 2e-6 * sum(abs(A*x))\n", prop.name);
  std::puts("Timing: warmed, reused inputs; median of 5 batches, kernel launches only (no allocation/copies).");
  bool passed = true;
  if (argc >= 3) {
    try {
      auto parse = [](const char* text) {
        size_t end;
        long long v = std::stoll(text, &end);
        if (text[end] != '\0' || v < 0 || v > INT_MAX) throw std::invalid_argument("range");
        return int(v);
      };
      int m = parse(argv[1]), n = parse(argv[2]);
      int repeats = argc == 4 ? parse(argv[3]) : 50;
      // Current kernels use int for row * N.
      if (size_t(m) * n > INT_MAX) throw std::invalid_argument("matrix exceeds int indexing");
      passed = run_case(m, n, repeats);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "Invalid arguments: %s\n", e.what());
      return 2;
    }
  } else {
    const std::pair<int, int> edges[] = {
      {0, 8}, {1, 0}, {1, 1}, {7, 31}, {8, 32}, {9, 33},
      {17, 127}, {31, 128}, {33, 129}, {9, 255}, {17, 256}, {19, 257}, {3, 4097}
    };
    for (auto shape : edges) passed &= run_case(shape.first, shape.second, 0);
    const std::pair<int, int> shapes[] = {{4096, 64}, {4096, 1024}, {1024, 4096}, {16, 16384}};
    for (auto shape : shapes) passed &= run_case(shape.first, shape.second, check_only ? 0 : 50);
  }
  std::puts(passed ? "ALL PASSED" : "FAILED");
  return passed ? 0 : 1;
}
