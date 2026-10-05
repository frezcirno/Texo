#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void solve(const float*, const float*, float*, int, int, int, float, float);

// Export the same cases for the optional Python/Triton runner, without a GPU.
static bool list_cases = false;

#define CUDA_CHECK(call) do { \
  cudaError_t error = (call); \
  if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)
// Reference uses the actual FP32 inputs and double accumulation, rounded to FP32.
// Reference uses the actual FP16 inputs, double accumulation, then FP16 output.
// Tolerance is explicit: abs_error <= 1e-4 + 1e-4 * abs(reference).
static bool run_case(const char* name, int m, int n, int k,
                     float alpha, float beta, bool ones, int repeats,
                     size_t c_offset = 128, size_t a_offset = 0,
                     size_t b_offset = 0, bool nan_c = false) {
  if (list_cases) {
    std::printf("{\"name\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,"
                "\"alpha\":%#.9g,\"beta\":%#.9g,\"ones\":%s,\"c_offset\":%zu,"
                "\"a_offset\":%zu,\"b_offset\":%zu,\"nan_c\":%s}\n",
                name, m, n, k, alpha, beta, ones ? "true" : "false",
                c_offset, a_offset, b_offset, nan_c ? "true" : "false");
    return true;
  }
  size_t na = size_t(m) * k, nb = size_t(k) * n, nc = size_t(m) * n;
  // A 256-byte prefix preserves cudaMalloc alignment while guarding C.
  // Keep a separate offset=1 correctness case for unaligned output buffers.
  std::vector<float> a(na), b(nb), initial(nc), actual(c_offset + nc + 1);
  std::vector<double> product(nc, 0.0);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : a) v = ones ? 1.0f : dist(rng);
  for (auto& v : b) v = ones ? 1.0f : dist(rng);
  for (auto& v : initial)
    v = nan_c ? std::numeric_limits<float>::quiet_NaN() : dist(rng);
  for (int row = 0; row < m; ++row)
    for (int col = 0; col < n; ++col)
      for (int i = 0; i < k; ++i)
        product[size_t(row) * n + col] +=
          double(a[size_t(row) * k + i]) * b[size_t(i) * n + col];

  float *a_storage, *b_storage, *dc;
  CUDA_CHECK(cudaMalloc(&a_storage, (a_offset + std::max(size_t(1), na)) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b_storage, (b_offset + std::max(size_t(1), nb)) * sizeof(float)));
  float *da = a_storage + a_offset, *db = b_storage + b_offset;
  CUDA_CHECK(cudaMalloc(&dc, actual.size() * sizeof(float)));
  if (na) CUDA_CHECK(cudaMemcpy(da, a.data(), na * sizeof(float), cudaMemcpyHostToDevice));
  if (nb) CUDA_CHECK(cudaMemcpy(db, b.data(), nb * sizeof(float), cudaMemcpyHostToDevice));
  std::fill(actual.begin(), actual.end(), 123.0f);
  std::copy(initial.begin(), initial.end(), actual.begin() + c_offset);
  CUDA_CHECK(cudaMemcpy(dc, actual.data(), actual.size() * sizeof(float), cudaMemcpyHostToDevice));

  bool passed = true;
  double max_error = 0;
  std::vector<float> previous = initial;
  // Two calls test overwrite for beta=0, and use of existing C for beta!=0.
  for (int invocation = 0; invocation < 2; ++invocation) {
    solve(da, db, dc + c_offset, m, n, k, alpha, beta);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      std::printf("  launch failed: %s\n", cudaGetErrorString(error));
      passed = false;
      break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(actual.data(), dc, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    bool prefix_ok = std::all_of(actual.begin(), actual.begin() + c_offset,
        [](float value) { return value == 123.0f; });
    if (!prefix_ok || actual.back() != 123.0f) {
      std::printf("  output guard overwritten\n");
      passed = false;
    }
    size_t bad = 0;
    for (size_t i = 0; i < nc; ++i) {
      double ref = float(alpha * product[i] +
          (beta == 0.0f ? 0.0 : beta * previous[i]));
      double got = actual[i + c_offset];
      double diff = std::abs(got - ref);
      max_error = std::max(max_error, diff);
      if (!std::isfinite(got) || diff > 1e-4 + 1e-4 * std::abs(ref)) {
        if (bad++ == 0)
          std::printf("  call=%d row=%zu col=%zu got=%g expected=%g\n", invocation + 1, i / n, i % n, got, ref);
      }
      // Compare each call against its actual input C, isolating new errors.
      previous[i] = actual[i + c_offset];
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
    for (int i = 0; i < 5; ++i) solve(da, db, dc + c_offset, m, n, k, alpha, 0.0f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times;
    for (int batch = 0; batch < 5; ++batch) {
      CUDA_CHECK(cudaEventRecord(start));
      for (int i = 0; i < repeats; ++i) solve(da, db, dc + c_offset, m, n, k, alpha, 0.0f);
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
  CUDA_CHECK(cudaFree(a_storage));
  CUDA_CHECK(cudaFree(b_storage));
  CUDA_CHECK(cudaFree(dc));
  return passed;
}

int main(int argc, char** argv) {
  list_cases = argc == 2 && std::string(argv[1]) == "--list-cases";
  bool check_only = list_cases || (argc == 2 && std::string(argv[1]) == "--check-only");
  if (argc != 1 && !check_only && argc != 4 && argc != 5) {
    std::fprintf(stderr, "Usage: %s [M N K [repeats]] | --check-only | --list-cases\n", argv[0]);
    return 2;
  }
  if (!list_cases) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s\nFP32 inputs/output, CPU double reference; atol=1e-4 rtol=1e-4\n", prop.name);
    std::puts("Timing: warmed inputs, beta=0, median of 5 batches; no allocation/copies included.");
    std::puts("Benchmark buffers are 256-byte aligned; includes host submission gaps between kernels.");
  }
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
    passed &= run_case("multi-tile-tail", 65, 129, 67, 0.75f, -0.25f, false, 0);
    passed &= run_case("unaligned-output", 17, 33, 19, 1, 0, false, 0, 1);
    // Aligned strides with misaligned base pointers must avoid 16-byte copies.
    passed &= run_case("unaligned-A", 64, 64, 96, 1, 0, false, 0, 128, 1, 0);
    passed &= run_case("unaligned-B", 64, 64, 96, 1, 0, false, 0, 128, 0, 1);
    passed &= run_case("unaligned-A-B", 64, 64, 96, 0.75f, -0.25f, false, 0, 128, 1, 1);
    // BK=8: one to four K tiles leave the four-stage ring partly empty.
    passed &= run_case("one-K-tile", 64, 64, 8, 0.75f, -0.25f, false, 0);
    passed &= run_case("two-K-tiles", 64, 64, 16, 0.75f, -0.25f, false, 0);
    passed &= run_case("three-K-tiles", 64, 64, 24, -0.75f, 0.5f, false, 0);
    passed &= run_case("four-K-tiles", 64, 64, 32, 1, 0, false, 0);
    passed &= run_case("aligned-multiblock", 128, 192, 96, -0.75f, 0.5f, false, 0);
    passed &= run_case("aligned-unalign-C", 64, 64, 96, 1, 0, false, 0, 1);
    passed &= run_case("full-output-K-zero", 64, 64, 0, 1, 0.5f, false, 0);
    // Each tail alone; K%4 != 0 and N%4 != 0 select scalar copies/stores.
    passed &= run_case("M-tail-only", 63, 64, 96, 1, 0, false, 0);
    passed &= run_case("N-tail-only", 64, 63, 96, 1, 0, false, 0);
    passed &= run_case("K-tail-only", 64, 64, 95, 1, 0, false, 0);
    passed &= run_case("K-tail-vector", 64, 64, 92, 1, 0, false, 0);
    passed &= run_case("N-tail-vector", 64, 68, 96, 0.5f, 0.25f, false, 0);
    // Enough 128x128 tiles to select the large configuration, and grid-stride
    // reuse of shared stages. Includes partial 128-row/column tiles.
    passed &= run_case("large-one-unalign-C", 1024, 2048, 8, 0.75f, -0.25f, false, 0, 1);
    passed &= run_case("large-three", 1024, 2048, 24, -0.75f, 0.5f, false, 0);
    passed &= run_case("large-unaligned-A", 1024, 2048, 40, 1, 0, false, 0, 128, 1, 0);
    passed &= run_case("large-tail", 1100, 1900, 77, 1, 0, false, 0);
    passed &= run_case("large-ring-wrap", 1536, 1536, 200, 0.75f, 0.5f, false, 0);
    // Grouped rasterization with a partial last row group.
    passed &= run_case("grouped-partial", 2200, 640, 64, 1, 0, false, 0);
    // beta=0 must not read C; NaN in the old output must disappear.
    passed &= run_case("alpha-one-beta", 128, 128, 96, 1, 0.5f, false, 0);
    passed &= run_case("alpha-other-beta0", 128, 128, 96, -0.75f, 0, false, 0);
    passed &= run_case("beta0-nan", 128, 128, 64, 1, 0, false, 0, 128, 0, 0, true);
    passed &= run_case("beta0-nan-tail", 65, 129, 67, 1, 0, false, 0, 128, 0, 0, true);
    passed &= run_case("beta0-nan-K0", 64, 64, 0, 1, 0, false, 0, 128, 0, 0, true);
    passed &= run_case("beta0-nan-large", 1024, 1024, 128, 1, -0.0f, false, 0, 1, 0, 0, true);
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
  if (!list_cases) std::puts(passed ? "ALL PASSED" : "FAILED (see cases above)");
  return passed ? 0 : 1;
}
