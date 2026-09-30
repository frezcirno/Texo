#include "test_utils.h"

extern "C" void solve(const float*, const float*, float*, int, int, int, int);

enum class Pattern { Sparse, Zero, Identity };

static bool run_case(const char* name, int rows, int inner, int cols, Pattern pattern) {
  std::vector<float> a(size_t(rows) * inner), b(size_t(inner) * cols);
  for (int row = 0; row < rows; ++row)
    for (int n = 0; n < inner; ++n) {
      float value = float((row * 13 + n * 7) % 19 - 9) * 0.25f;
      if (pattern == Pattern::Sparse && (row * 17 + n * 5) % 7 < 5) value = 0.0f;
      if (pattern == Pattern::Zero) value = 0.0f;
      if (pattern == Pattern::Identity) value = row == n ? 1.0f : 0.0f;
      a[size_t(row) * inner + n] = value;
    }
  for (int n = 0; n < inner; ++n)
    for (int col = 0; col < cols; ++col)
      b[size_t(n) * cols + col] = float((n * 3 + col * 11) % 23 - 11) * 0.125f;

  int nnz = 0;
  for (float value : a) nnz += value != 0.0f;
  std::vector<double> expected(size_t(rows) * cols, 0.0);
  for (int row = 0; row < rows; ++row)
    for (int col = 0; col < cols; ++col)
      for (int n = 0; n < inner; ++n)
        expected[size_t(row) * cols + col] +=
            double(a[size_t(row) * inner + n]) * b[size_t(n) * cols + col];

  test::DeviceArray<float> device_a(a), device_b(b);
  bool passed = test::check_output<float>(name, expected, [&](float* output) {
    solve(device_a.data(), device_b.data(), output, rows, inner, cols, nnz);
  }, 1e-4, 1e-4);
  passed &= device_a.unchanged(a);
  passed &= device_b.unchanged(b);
  return passed;
}

int main() {
  bool passed = true;
  passed &= run_case("scalar", 1, 1, 1, Pattern::Sparse);
  passed &= run_case("problem-example-shape", 3, 4, 2, Pattern::Sparse);
  passed &= run_case("zero-a", 7, 19, 23, Pattern::Zero);
  passed &= run_case("identity-a", 17, 17, 31, Pattern::Identity);
  passed &= run_case("partial-output-tile", 17, 33, 31, Pattern::Sparse);
  passed &= run_case("rectangular", 33, 17, 35, Pattern::Sparse);
  return test::finish(passed);
}
