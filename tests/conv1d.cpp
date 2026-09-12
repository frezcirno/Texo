#include "test_utils.h"
#include <utility>

extern "C" void solve(const float*, const float*, float*, int, int);

static bool run_case(int n, int k, int pattern) {
  auto input = test::random_values(n);
  auto kernel = test::random_values(k, 123);
  if (pattern == 1) std::fill(input.begin(), input.end(), 0);
  if (pattern == 2) {
    std::fill(input.begin(), input.end(), 0);
    input[n / 2] = 1;
    for (int j = 0; j < k; ++j) kernel[j] = float(j + 1);
  }
  if (pattern == 3) {
    std::fill(input.begin(), input.end(), 1);
    std::fill(kernel.begin(), kernel.end(), 1);
  }
  // Valid cross-correlation: no padding and no reversal of the kernel.
  std::vector<double> reference(n - k + 1, 0);
  for (size_t i = 0; i < reference.size(); ++i)
    for (int j = 0; j < k; ++j)
      reference[i] += double(input[i + j]) * kernel[j];
  test::DeviceArray<float> di(input), dk(kernel);
  const char* names[] = {"random", "zero", "impulse", "ones"};
  bool passed = test::check_output<float>(names[pattern], reference, [&](float* output) {
    solve(di.data(), dk.data(), output, n, k);
  }, 1e-4, 1e-5);
  passed &= di.unchanged(input);
  passed &= dk.unchanged(kernel);
  return passed;
}

int main() {
  bool passed = true;
  const std::pair<int, int> shapes[] = {
    {1, 1}, {7, 1}, {8, 3}, {33, 4}, {257, 2}, {258, 3},
    {259, 3}, {257, 257}, {1025, 5}, {65537, 7}
  };
  for (auto shape : shapes) passed &= run_case(shape.first, shape.second, 0);
  for (int pattern : {1, 2, 3}) passed &= run_case(33, 4, pattern);
  return test::finish(passed);
}
