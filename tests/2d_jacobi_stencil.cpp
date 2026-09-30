#include "test_utils.h"
#include <string>

extern "C" void solve(const float*, float*, int, int);

static bool run_case(const char* name, int rows, int cols, int input_offset = 0,
                     int output_offset = 0) {
  constexpr size_t GUARD = 4;
  constexpr float MARKER = 1234567.0f;
  const size_t count = size_t(rows) * cols;
  std::vector<float> values(count);
  for (int row = 0; row < rows; ++row)
    for (int col = 0; col < cols; ++col)
      values[size_t(row) * cols + col] =
          float((row * 17 + col * 7) % 29 - 14) * 0.125f;

  std::vector<double> expected(values.begin(), values.end());
  for (int row = 1; row + 1 < rows; ++row)
    for (int col = 1; col + 1 < cols; ++col)
      expected[size_t(row) * cols + col] =
          (double(values[size_t(row) * cols + col - 1]) +
           double(values[size_t(row) * cols + col + 1]) +
           double(values[size_t(row - 1) * cols + col]) +
           double(values[size_t(row + 1) * cols + col])) /
          4.0;

  std::vector<float> stored_input(input_offset + count, MARKER);
  std::copy(values.begin(), values.end(), stored_input.begin() + input_offset);
  test::DeviceArray<float> input(stored_input);
  const size_t prefix = GUARD + output_offset;
  std::vector<float> initial(prefix + count + GUARD, MARKER);
  test::DeviceArray<float> output(initial);
  bool passed = true;
  double max_error = 0;

  for (int call = 0; call < 2; ++call) {
    output.upload(initial);
    solve(input.data() + input_offset, output.data() + prefix, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto actual = output.download();
    for (size_t i = 0; i < prefix; ++i)
      if (actual[i] != MARKER) {
        std::fprintf(stderr, "%s call=%d output prefix overwritten at %zu\n", name, call + 1, i);
        passed = false;
        break;
      }
    for (size_t i = prefix + count; i < actual.size(); ++i)
      if (actual[i] != MARKER) {
        std::fprintf(stderr, "%s call=%d output suffix overwritten at %zu\n", name, call + 1, i);
        passed = false;
        break;
      }
    for (size_t i = 0; i < count; ++i) {
      const double error = std::abs(double(actual[prefix + i]) - expected[i]);
      max_error = std::max(max_error, error);
      if (!std::isfinite(actual[prefix + i]) || error > 1e-6) {
        std::fprintf(stderr, "%s call=%d index=%zu got=%.9g expected=%.9g\n", name,
                     call + 1, i, actual[prefix + i], expected[i]);
        passed = false;
        break;
      }
    }
    passed &= input.unchanged(stored_input);
  }
  std::printf("%s %-22s rows=%d cols=%d max_abs_error=%.3g\n",
              passed ? "PASS" : "FAIL", name, rows, cols, max_error);
  return passed;
}

int main() {
  bool passed = true;
  passed &= run_case("example-4x4", 4, 4);
  passed &= run_case("single-cell", 1, 1);
  passed &= run_case("single-row", 1, 33);
  passed &= run_case("single-column", 33, 1);
  passed &= run_case("small-rectangular", 3, 5, 1, 2);
  passed &= run_case("partial-blocks", 17, 33, 2, 1);
  passed &= run_case("many-blocks", 65, 49);
  return test::finish(passed);
}
