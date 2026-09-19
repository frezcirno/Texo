// Large-shape timings with a cuBLAS FP32-reduction reference. tests/gemm.cu
// supplies the independent CPU reference and general alpha/beta coverage.
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" void solve(const half*, const half*, half*, int, int, int, float, float);

#define CUDA_CHECK(call) do { \
  cudaError_t error = (call); \
  if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    return 1; \
  } \
} while (0)
#define CUBLAS_CHECK(call) do { \
  cublasStatus_t error = (call); \
  if (error != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "%s:%d: cuBLAS error %d\n", __FILE__, __LINE__, int(error)); \
    return 1; \
  } \
} while (0)

static bool positive_int(const char* text, int& value) {
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(text, &end, 10);
  if (errno || end == text || *end || parsed <= 0 || parsed > INT_MAX)
    return false;
  value = int(parsed);
  return true;
}

int main(int argc, char** argv) {
  const bool profile = argc == 5 && std::strcmp(argv[4], "--profile") == 0;
  int m = 0, n = 0, k = 0, iterations = 100;
  if ((argc != 4 && argc != 5) || !positive_int(argv[1], m) ||
      !positive_int(argv[2], n) || !positive_int(argv[3], k) ||
      (argc == 5 && !profile && !positive_int(argv[4], iterations))) {
    std::fprintf(stderr, "Usage: %s M N K [iterations | --profile]\n", argv[0]);
    return 2;
  }

  const size_t na = size_t(m) * k, nb = size_t(k) * n, nc = size_t(m) * n;
  constexpr size_t PREFIX = 128;
  std::vector<half> a(na), b(nb), c(nc + PREFIX + 1, __float2half(123)), reference(nc);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1, 1);
  for (auto& value : a) value = __float2half(dist(rng));
  for (auto& value : b) value = __float2half(dist(rng));

  half *da, *db, *dc, *dr;
  CUDA_CHECK(cudaMalloc(&da, na * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&db, nb * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dc, c.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dr, nc * sizeof(half)));
  CUDA_CHECK(cudaMemcpy(da, a.data(), na * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, b.data(), nb * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dc, c.data(), c.size() * sizeof(half), cudaMemcpyHostToDevice));

  cublasHandle_t reference_handle;
  CUBLAS_CHECK(cublasCreate(&reference_handle));
  CUBLAS_CHECK(cublasSetMathMode(reference_handle,
                                CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
  const float alpha = 1, beta = 0;
  CUBLAS_CHECK(cublasGemmEx(reference_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
      &alpha, db, CUDA_R_16F, n, da, CUDA_R_16F, k, &beta, dr, CUDA_R_16F, n,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CUDA_CHECK(cudaMemcpy(reference.data(), dr, nc * sizeof(half), cudaMemcpyDeviceToHost));

  solve(da, db, dc + PREFIX, m, n, k, alpha, beta);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(c.data(), dc, c.size() * sizeof(half), cudaMemcpyDeviceToHost));
  float max_error = 0;
  size_t bad = 0;
  for (size_t i = 0; i < c.size(); ++i) {
    const float got = __half2float(c[i]);
    if (i < PREFIX || i >= nc + PREFIX) {
      if (got != 123) ++bad;
      continue;
    }
    const float ref = __half2float(reference[i - PREFIX]);
    const float diff = std::abs(got - ref);
    max_error = std::max(max_error, diff);
    if (!std::isfinite(got) || diff > 0.01f + 0.01f * std::abs(ref)) ++bad;
  }
  if (bad) {
    std::fprintf(stderr, "FAIL mismatches=%zu max_error=%g\n", bad, max_error);
    return 1;
  }

  for (int i = 0; i < 5; ++i) solve(da, db, dc + PREFIX, m, n, k, alpha, beta);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  if (profile) {
    CUDA_CHECK(cudaProfilerStart());
    solve(da, db, dc + PREFIX, m, n, k, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaProfilerStop());
    std::printf("PASS %d %d %d max_error=%g (profiled one warmed launch)\n",
                m, n, k, max_error);
  } else {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times;
    for (int batch = 0; batch < 5; ++batch) {
      CUDA_CHECK(cudaEventRecord(start));
      for (int i = 0; i < iterations; ++i)
        solve(da, db, dc + PREFIX, m, n, k, alpha, beta);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float ms;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      times.push_back(ms * 1000 / iterations);
    }
    std::sort(times.begin(), times.end());
    std::printf("PASS %d %d %d median_us=%.3f max_error=%g (cuBLAS reference)\n",
                m, n, k, times[2], max_error);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }
  CUBLAS_CHECK(cublasDestroy(reference_handle));
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dc));
  CUDA_CHECK(cudaFree(dr));
}
