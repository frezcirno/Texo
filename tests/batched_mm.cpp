#include "test_utils.h"

extern "C" void solve(const float*, const float*, float*, int, int, int, int);

enum Pattern { Random, BatchConstants, Zero, LeftIdentity, RightIdentity, TailOnly };

static bool run_case(int batches, int m, int n, int k, Pattern pattern) {
  auto a = test::random_values(size_t(batches) * m * k, 42);
  auto b = test::random_values(size_t(batches) * k * n, 123);
  for (float& value : a) value /= 8.0f;
  for (float& value : b) value /= 8.0f;

  for (int batch = 0; batch < batches; ++batch) {
    size_t a_start = size_t(batch) * m * k;
    size_t b_start = size_t(batch) * k * n;
    for (int row = 0; row < m; ++row) {
      for (int inner = 0; inner < k; ++inner) {
        float& value = a[a_start + size_t(row) * k + inner];
        if (pattern == BatchConstants) value = float(batch + 1);
        if (pattern == Zero) value = 0;
        if (pattern == LeftIdentity) value = row == inner ? 1 : 0;
        if (pattern == TailOnly) value = inner == k - 1 ? float(batch + row + 1) : 0;
      }
    }
    for (int inner = 0; inner < k; ++inner) {
      for (int col = 0; col < n; ++col) {
        float& value = b[b_start + size_t(inner) * n + col];
        if (pattern == BatchConstants) value = 0.25f * float(batch + 2);
        if (pattern == RightIdentity) value = inner == col ? 1 : 0;
      }
    }
  }

  // Independent CPU double reference for each row-major A[batch] * B[batch].
  std::vector<double> reference(size_t(batches) * m * n, 0);
  for (int batch = 0; batch < batches; ++batch) {
    size_t a_start = size_t(batch) * m * k;
    size_t b_start = size_t(batch) * k * n;
    size_t c_start = size_t(batch) * m * n;
    for (int row = 0; row < m; ++row)
      for (int col = 0; col < n; ++col)
        for (int inner = 0; inner < k; ++inner)
          reference[c_start + size_t(row) * n + col] +=
              double(a[a_start + size_t(row) * k + inner]) *
              b[b_start + size_t(inner) * n + col];
  }

  test::DeviceArray<float> da(a), db(b);
  const char* names[] = {"random", "batch-constants", "zero", "left-identity",
                         "right-identity", "tail-only"};
  char label[128];
  std::snprintf(label, sizeof(label), "%s B=%d M=%d N=%d K=%d",
                names[pattern], batches, m, n, k);
  // The common checker clears C before each of two calls, then checks values
  // and output guards. The exercise permits the kernel to accumulate into C.
  bool passed = test::check_output<float>(label, reference, [&](float* output) {
    solve(da.data(), db.data(), output, batches, m, n, k);
  }, 1e-5, 1e-5);
  passed &= da.unchanged(a);
  passed &= db.unchanged(b);
  return passed;
}

int main() {
  bool passed = true;
  struct Shape { int batches, m, n, k; };
  // The current block covers N=4, M=8, BATCH=8. Exercise both full and
  // partial blocks, rectangular matrices, multiple batches and long reductions.
  const Shape shapes[] = {
    {1, 1, 1, 1}, {1, 1, 17, 7}, {1, 17, 1, 7}, {2, 3, 5, 7},
    {7, 7, 3, 5}, {8, 8, 4, 8}, {9, 9, 5, 9}, {16, 16, 8, 16},
    {17, 5, 7, 33}, {3, 17, 33, 19}, {2, 3, 9, 257},
    {2, 3, 5, 1025}, {4, 65, 67, 31}
  };
  for (auto shape : shapes)
    passed &= run_case(shape.batches, shape.m, shape.n, shape.k, Random);
  passed &= run_case(9, 9, 5, 7, BatchConstants);
  passed &= run_case(3, 17, 9, 19, Zero);
  passed &= run_case(3, 17, 9, 17, LeftIdentity);
  passed &= run_case(9, 7, 5, 5, RightIdentity);
  passed &= run_case(3, 9, 5, 257, TailOnly);
  return test::finish(passed);
}
