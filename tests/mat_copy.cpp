#include "test_utils.h"

extern "C" void solve(const float*, float*, int);

int main() {
  bool passed = true;
  for (int n : {1, 2, 15, 16, 17, 31, 32, 33, 257}) {
    auto input = test::random_values(size_t(n) * n);
    std::vector<double> reference(input.begin(), input.end());
    test::DeviceArray<float> di(input);
    passed &= test::check_output<float>("square-copy", reference, [&](float* output) {
      solve(di.data(), output, n); // N is the side length, not the element count.
    });
    passed &= di.unchanged(input);
  }
  return test::finish(passed);
}
