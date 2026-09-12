#include "test_utils.h"

extern "C" void solve(float*, int);

int main() {
  bool passed = true;
  constexpr size_t GUARD = 4;
  for (int n : {1, 2, 3, 31, 32, 33, 255, 256, 257, 1025, 65537}) {
    auto input = test::random_values(n);
    std::vector<float> initial(size_t(n) + 2 * GUARD, 1234567.0f);
    std::copy(input.begin(), input.end(), initial.begin() + GUARD);
    test::DeviceArray<float> device(initial);
    auto reference = initial;
    bool correct = true;
    // In-place API: the second reversal must restore the original input.
    for (int call = 0; call < 2; ++call) {
      std::reverse(reference.begin() + GUARD, reference.end() - GUARD);
      solve(device.data() + GUARD, n);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      auto actual = device.download();
      if (std::memcmp(actual.data(), reference.data(), actual.size() * sizeof(float)) != 0) {
        std::fprintf(stderr, "reverse N=%d call=%d result or guards differ\n", n, call + 1);
        correct = false;
      }
    }
    std::printf("%s reverse N=%d (two reversals)\n", correct ? "PASS" : "FAIL", n);
    passed &= correct;
  }
  return test::finish(passed);
}
