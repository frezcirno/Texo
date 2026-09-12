#include "test_utils.h"

extern "C" void solve(const float*, const float*, float*, int);

static bool run_case(int n, int pattern) {
  auto a = test::random_values(size_t(n) * n, 42);
  auto b = test::random_values(size_t(n) * n, 123);
  std::vector<double> reference(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    if (pattern == 1) b[i] = -a[i];
    if (pattern == 2) b[i] = 0;
    reference[i] = float(a[i] + b[i]);
  }
  test::DeviceArray<float> da(a), db(b);
  const char* names[] = {"random", "cancellation", "add-zero"};
  bool passed = test::check_output<float>(names[pattern], reference, [&](float* output) {
    solve(da.data(), db.data(), output, n);
  });
  passed &= da.unchanged(a);
  passed &= db.unchanged(b);
  return passed;
}

int main() {
  bool passed = true;
  for (int n : {1, 2, 7, 8, 9, 31, 32, 33, 127, 257})
    passed &= run_case(n, 0);
  passed &= run_case(33, 1);
  passed &= run_case(33, 2);
  return test::finish(passed);
}
