#include <cuda_fp16.h>
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

extern "C" void solve(const half*, const half*, half*, int, int, int, float, float);

#define CUDA_CHECK(call) do { \
  cudaError_t error = (call); \
  if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

// Reference uses the actual FP16 inputs, double accumulation, then FP16 output.
// Tolerance is explicit: abs_error <= 0.01 + 0.01 * abs(reference).
static bool run_case(const char* name, int m, int n, int k,
                     float alpha, float beta, bool ones, int repeats) {
  size_t na = size_t(m) * k, nb = size_t(k) * n, nc = size_t(m) * n;
  std::vector<half> a(na), b(nb), initial(nc), actual(nc + 2);
  std::vector<double> product(nc, 0.0);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : a) v = __float2half(ones ? 1.0f : dist(rng));
  for (auto& v : b) v = __float2half(ones ? 1.0f : dist(rng));
  for (auto& v : initial) v = __float2half(dist(rng));
  for (int row = 0; row < m; ++row)
    for (int col = 0; col < n; ++col)
      for (int i = 0; i < k; ++i)
        product[size_t(row) * n + col] +=
          double(__half2float(a[size_t(row) * k + i])) * __half2float(b[size_t(i) * n + col]);

  half *da, *db, *dc;
  CUDA_CHECK(cudaMalloc(&da, std::max(size_t(1), na) * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&db, std::max(size_t(1), nb) * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dc, (nc + 2) * sizeof(half)));
  if (na) CUDA_CHECK(cudaMemcpy(da, a.data(), na * sizeof(half), cudaMemcpyHostToDevice));
  if (nb) CUDA_CHECK(cudaMemcpy(db, b.data(), nb * sizeof(half), cudaMemcpyHostToDevice));
  std::fill(actual.begin(), actual.end(), __float2half(123.0f));
  std::copy(initial.begin(), initial.end(), actual.begin() + 1);
  CUDA_CHECK(cudaMemcpy(dc, actual.data(), (nc + 2) * sizeof(half), cudaMemcpyHostToDevice));

  bool passed = true;
  double max_error = 0;
  std::vector<half> previous = initial;
  // Two calls test overwrite for beta=0, and use of existing C for beta!=0.
  for (int invocation = 0; invocation < 2; ++invocation) {
    solve(da, db, dc + 1, m, n, k, alpha, beta);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      std::printf("  launch failed: %s\n", cudaGetErrorString(error));
      passed = false;
      break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(actual.data(), dc, (nc + 2) * sizeof(half), cudaMemcpyDeviceToHost));
    if (__half2float(actual.front()) != 123.0f || __half2float(actual.back()) != 123.0f) {
      std::printf("  output guard overwritten\n");
      passed = false;
    }
    size_t bad = 0;
    for (size_t i = 0; i < nc; ++i) {
      double ref = __half2float(__float2half(float(alpha * product[i] + beta * __half2float(previous[i]))));
      double got = __half2float(actual[i + 1]);
      double diff = std::abs(got - ref);
      max_error = std::max(max_error, diff);
      if (!std::isfinite(got) || diff > 0.01 + 0.01 * std::abs(ref)) {
        if (bad++ == 0)
          std::printf("  call=%d row=%zu col=%zu got=%g expected=%g\n", invocation + 1, i / n, i % n, got, ref);
      }
      // Compare each call against its actual input C, isolating new errors.
      previous[i] = actual[i + 1];
    }
    if (bad) {
      std::printf("  mismatches=%zu/%zu\n", bad, nc);
      passed = false;
    }
  }
  std::printf("%-18s %s M=%d N=%d K=%d alpha=%g beta=%g max_abs_error=%.6g",
              name, passed ? "PASS" : "FAIL", m, n, k, alpha, beta, max_error);

  if (passed && repeats > 0 && nc) {
    // Benchmark beta=0 separately, so repeated launches do not grow C.
    for (int i = 0; i < 5; ++i) solve(da, db, dc + 1, m, n, k, alpha, 0.0f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times;
    for (int batch = 0; batch < 5; ++batch) {
      CUDA_CHECK(cudaEventRecord(start));
      for (int i = 0; i < repeats; ++i) solve(da, db, dc + 1, m, n, k, alpha, 0.0f);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float ms;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      times.push_back(ms / repeats);
    }
    std::sort(times.begin(), times.end());
    std::printf(" beta0_us=%.3f GFLOP/s=%.3f", times[2] * 1000, 2.0 * m * n * k / (times[2] * 1e6));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }
  std::puts("");
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dc));
  return passed;
}

int main(int argc, char** argv) {
  bool check_only = argc == 2 && std::string(argv[1]) == "--check-only";
  if (argc != 1 && !check_only && argc != 4 && argc != 5) {
    std::fprintf(stderr, "Usage: %s [M N K [repeats]] | --check-only\n", argv[0]);
    return 2;
  }
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s\nFP16 inputs/output, CPU double reference; atol=0.01 rtol=0.01\n", prop.name);
  std::puts("Timing: warmed inputs, beta=0, median of 5 batches; no allocation/copies included.");
  bool passed = true;
  if (argc >= 4) {
    try {
      auto parse = [](const char* text) {
        size_t end;
        long long v = std::stoll(text, &end);
        if (text[end] || v < 0 || v > INT_MAX) throw std::invalid_argument("expected nonnegative int");
        return int(v);
      };
      int m = parse(argv[1]), n = parse(argv[2]), k = parse(argv[3]);
      int repeats = argc == 5 ? parse(argv[4]) : 20;
      if (size_t(m)*n > INT_MAX || size_t(m)*k > INT_MAX || size_t(k)*n > INT_MAX)
        throw std::invalid_argument("exceeds kernel int indexing");
      passed = run_case("custom", m, n, k, 1, 0, false, repeats);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "Invalid arguments: %s\n", e.what());
      return 2;
    }
  } else {
    passed &= run_case("scalar", 1, 1, 1, 1, 0, false, 0);
    passed &= run_case("tile-aligned", 16, 16, 16, 1, 0, false, 0);
    passed &= run_case("rectangular-tail", 17, 33, 19, 1, 0, false, 0);
    passed &= run_case("single-row", 1, 35, 31, 0.5f, 0.25f, false, 0);
    passed &= run_case("single-column", 35, 1, 33, -1, 1, false, 0);
    passed &= run_case("alpha-beta", 19, 7, 65, -0.75f, 0.5f, false, 0);
    passed &= run_case("alpha-zero", 7, 19, 17, 0, -0.5f, false, 0);
    passed &= run_case("K-zero", 3, 5, 0, 1, 0.5f, false, 0);
    passed &= run_case("M-zero-no-op", 0, 7, 3, 1, 0, false, 0);
    passed &= run_case("N-zero-no-op", 7, 0, 3, 1, 0, false, 0);
    passed &= run_case("long-sum-ones", 1, 1, 4096, 1, 0, true, 0);
    passed &= run_case("random-long-K", 9, 17, 1025, 1, 0, false, 0);
    passed &= run_case("benchmark", 128, 128, 128, 1, 0, false, check_only ? 0 : 20);
  }
  std::puts(passed ? "ALL PASSED" : "FAILED (see cases above)");
  return passed ? 0 : 1;
}
