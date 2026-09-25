#include "test_utils.h"

extern "C" void solve(const float*, const float*, float*, int, int, int, int);

enum Pattern { Random, BatchConstants, Zero, LeftIdentity, RightIdentity, TailOnly,
               ZeroB, LastBatchOnly };

static bool run_case(int batches, int m, int n, int k, Pattern pattern,
                     bool accumulate = false, int a_offset = 0,
                     int b_offset = 0, int c_offset = 0) {
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
        if (pattern == LastBatchOnly)
          value = batch == batches - 1 ? 0.125f * float(row + 1) : 0;
      }
    }
    for (int inner = 0; inner < k; ++inner) {
      for (int col = 0; col < n; ++col) {
        float& value = b[b_start + size_t(inner) * n + col];
        if (pattern == BatchConstants) value = 0.25f * float(batch + 2);
        if (pattern == RightIdentity) value = inner == col ? 1 : 0;
        if (pattern == ZeroB) value = 0;
        if (pattern == LastBatchOnly)
          value = batch == batches - 1 ? 0.25f * float(col + 2) : 0;
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

  constexpr size_t GUARD = 4;
  constexpr float MARKER = 1234567.0f;
  // Keep nonempty inputs unpadded at the end so memcheck catches tail reads.
  a.insert(a.begin(), a_offset, MARKER);
  b.insert(b.begin(), b_offset, MARKER);
  if (a.empty()) a.push_back(MARKER);
  if (b.empty()) b.push_back(MARKER);
  test::DeviceArray<float> da(a), db(b);
  const char* names[] = {"random", "batch-constants", "zero", "left-identity",
                         "right-identity", "tail-only", "zero-B", "last-batch-only"};
  char label[128];
  std::snprintf(label, sizeof(label), "%s B=%d M=%d N=%d K=%d %s offsets=%d/%d/%d",
                names[pattern], batches, m, n, k, accumulate ? "accumulate" : "clear",
                a_offset, b_offset, c_offset);
  const size_t prefix = GUARD + c_offset;
  std::vector<float> initial(prefix + reference.size() + GUARD, MARKER);
  std::vector<double> expected(reference.size());
  for (size_t i = 0; i < reference.size(); ++i) {
    // Distinct, nonzero, exactly representable values expose overwritten C.
    float value = accumulate ? float(int(i % 17) + 1) * (i % 2 ? -0.125f : 0.125f) : 0;
    initial[prefix + i] = value;
    expected[i] = value;
  }
  test::DeviceArray<float> dc(initial);
  bool passed = true;
  double max_error = 0;
  const int calls = accumulate ? 3 : 2;
  for (int call = 0; call < calls; ++call) {
    if (!accumulate) dc.upload(initial);
    for (size_t i = 0; i < reference.size(); ++i) {
      // Account for the float output between calls without using GPU results
      // as the reference for the next invocation.
      expected[i] = accumulate ? double(float(expected[i] + reference[i])) : reference[i];
    }
    // A/B must not be read when the reduction is empty.
    solve(k == 0 ? nullptr : da.data() + a_offset,
          k == 0 ? nullptr : db.data() + b_offset, dc.data() + prefix,
          batches, m, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto actual = dc.download();
    for (size_t i = 0; i < actual.size(); ++i) {
      if ((i < prefix || i >= prefix + reference.size()) && actual[i] != MARKER) {
        std::fprintf(stderr, "%s call=%d output guard overwritten at %zu\n",
                     label, call + 1, i);
        passed = false;
        break;
      }
    }
    for (size_t i = 0; i < reference.size(); ++i) {
      double got = actual[prefix + i];
      double error = std::abs(got - expected[i]);
      max_error = std::max(max_error, error);
      bool exact = k == 0 || pattern == Zero || pattern == ZeroB ||
                   (pattern == LastBatchOnly && i < size_t(batches - 1) * m * n);
      double tolerance = exact ? 0 : 1e-5 + 1e-5 * std::abs(expected[i]);
      if (!std::isfinite(got) || error > tolerance) {
        std::fprintf(stderr, "%s call=%d index=%zu got=%.12g expected=%.12g\n",
                     label, call + 1, i, got, expected[i]);
        passed = false;
        break;
      }
    }
    passed &= da.unchanged(a);
    passed &= db.unchanged(b);
  }
  std::printf("%s %s calls=%d max_abs_error=%.3g\n",
              passed ? "PASS" : "FAIL", label, calls, max_error);
  return passed;
}

static bool run_empty_case(int batches, int m, int n, int k) {
  const std::vector<float> before = {-0.0f, 1.25f, -2.5f, 1234567.0f};
  test::DeviceArray<float> da(before), db(before), dc(before);
  bool passed = true;
  // Check untouched storage, then ensure an empty output accepts null pointers.
  for (int call = 0; call < 2; ++call) {
    solve(call == 0 ? da.data() : nullptr, call == 0 ? db.data() : nullptr,
          call == 0 ? dc.data() : nullptr, batches, m, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    passed &= da.unchanged(before);
    passed &= db.unchanged(before);
    passed &= dc.unchanged(before);
  }
  std::printf("%s empty-no-op B=%d M=%d N=%d K=%d\n",
              passed ? "PASS" : "FAIL", batches, m, n, k);
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
    {2, 3, 5, 1025}, {4, 65, 67, 31},
    // Isolate each grid axis below/above one full block.
    {8, 8, 3, 7}, {8, 7, 4, 7}, {7, 8, 4, 7},
    {8, 8, 5, 7}, {8, 9, 4, 7}, {9, 8, 4, 7},
    {33, 3, 5, 1}, {2, 33, 65, 1}
  };
  for (auto shape : shapes)
    passed &= run_case(shape.batches, shape.m, shape.n, shape.k, Random);
  passed &= run_case(9, 9, 5, 7, BatchConstants);
  passed &= run_case(3, 17, 9, 19, Zero);
  passed &= run_case(3, 17, 9, 17, LeftIdentity);
  passed &= run_case(9, 7, 5, 5, RightIdentity);
  passed &= run_case(3, 9, 5, 257, TailOnly);
  passed &= run_case(3, 17, 9, 19, ZeroB);
  passed &= run_case(17, 9, 5, 7, LastBatchOnly);
  passed &= run_case(9, 9, 5, 17, Random, true);
  passed &= run_case(9, 9, 5, 7, BatchConstants, true);
  passed &= run_case(3, 7, 5, 9, Zero, true);
  passed &= run_case(3, 7, 5, 9, ZeroB, true);
  passed &= run_case(17, 9, 7, 3, LastBatchOnly, true);
  passed &= run_case(9, 9, 5, 7, Random, true, 1, 0, 0);
  passed &= run_case(9, 9, 5, 7, Random, true, 0, 1, 0);
  passed &= run_case(9, 9, 5, 7, Random, true, 0, 0, 1);
  passed &= run_case(9, 9, 5, 7, Random, true, 1, 2, 3);
  passed &= run_case(8, 8, 4, 0, Random);
  passed &= run_case(9, 9, 5, 0, Random, true);
  passed &= run_case(9, 9, 5, 0, Random, true, 0, 0, 1);
  passed &= run_empty_case(0, 9, 5, 7);
  passed &= run_empty_case(9, 0, 5, 7);
  passed &= run_empty_case(9, 9, 0, 7);
  passed &= run_empty_case(0, 0, 0, 0);
  return test::finish(passed);
}
