#include "test_utils.h"

extern "C" void solve(const float*, const float*, float*, int);

int main() {
  bool passed = true;
  for (int n : {1, 2, 7, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1025, 65537}) {
    std::vector<float> a(n), b(n);
    std::vector<double> reference;
    reference.reserve(size_t(2) * n);
    for (int i = 0; i < n; ++i) {
      a[i] = float(i) + 0.25f;
      b[i] = -float(i) - 0.75f;
      reference.push_back(a[i]);
      reference.push_back(b[i]);
    }
    test::DeviceArray<float> da(a), db(b);
    passed &= test::check_output<float>("A0-B0-A1-B1", reference, [&](float* output) {
      solve(da.data(), db.data(), output, n);
    });
    passed &= da.unchanged(a);
    passed &= db.unchanged(b);
  }
  return test::finish(passed);
}
