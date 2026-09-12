#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CUDA_CHECK(call) do { \
  cudaError_t error = (call); \
  if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

namespace test {
template <typename T> class DeviceArray {
 public:
  explicit DeviceArray(size_t count) : count_(count) {
    CUDA_CHECK(cudaMalloc(&data_, std::max(size_t(1), count) * sizeof(T)));
  }
  explicit DeviceArray(const std::vector<T>& host) : DeviceArray(host.size()) {
    upload(host);
  }
  ~DeviceArray() { CUDA_CHECK(cudaFree(data_)); }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  T* data() const { return data_; }
  void upload(const std::vector<T>& host) {
    CUDA_CHECK(cudaMemcpy(data_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> download() const {
    std::vector<T> host(count_);
    CUDA_CHECK(cudaMemcpy(host.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
  }
  bool unchanged(const std::vector<T>& before) const {
    auto after = download();
    bool passed = std::memcmp(after.data(), before.data(), count_ * sizeof(T)) == 0;
    if (!passed) std::fprintf(stderr, "Input modified\n");
    return passed;
  }
 private:
  size_t count_;
  T* data_ = nullptr;
};

inline std::vector<float> random_values(size_t n, unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
  std::vector<float> values(n);
  for (float& value : values) value = dist(rng);
  return values;
}

// Four guard elements preserve 16-byte alignment for float/uint32 outputs.
// Clear output before each call, matching the exercise calling convention.
template <typename T, typename Launch>
bool check_output(const char* name, const std::vector<double>& reference,
                  Launch launch, double atol = 0, double rtol = 0) {
  constexpr size_t GUARD = 4;
  const T marker = T(1234567);
  size_t n = reference.size();
  std::vector<T> initial(n + 2 * GUARD, T(0));
  std::fill(initial.begin(), initial.begin() + GUARD, marker);
  std::fill(initial.end() - GUARD, initial.end(), marker);
  DeviceArray<T> device(initial);
  bool passed = true;
  double max_error = 0;
  for (int call = 0; call < 2; ++call) {
    device.upload(initial);
    launch(device.data() + GUARD);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto actual = device.download();
    for (size_t i = 0; i < GUARD; ++i) {
      if (actual[i] != marker || actual[GUARD + n + i] != marker) {
        std::fprintf(stderr, "%s call=%d output guard overwritten\n", name, call + 1);
        passed = false;
        break;
      }
    }
    for (size_t i = 0; i < n; ++i) {
      double got = double(actual[GUARD + i]);
      double error = std::abs(got - reference[i]);
      max_error = std::max(max_error, error);
      if (!std::isfinite(got) || error > atol + rtol * std::abs(reference[i])) {
        std::fprintf(stderr, "%s call=%d index=%zu got=%.12g expected=%.12g\n",
                     name, call + 1, i, got, reference[i]);
        passed = false;
        break;
      }
    }
  }
  std::printf("%s %-16s elements=%zu max_abs_error=%.3g\n",
              passed ? "PASS" : "FAIL", name, n, max_error);
  return passed;
}

inline int finish(bool passed) {
  std::puts(passed ? "ALL PASSED" : "FAILED");
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
} // namespace test
