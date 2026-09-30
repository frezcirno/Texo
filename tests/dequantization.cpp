#include "test_utils.h"
#include <string>

extern "C" void solve(const float*, const float*, float*, int, int, int);

static bool run_case(const char* name, int rows, int cols, int tile_size) {
  const int scale_rows = (rows + tile_size - 1) / tile_size;
  const int scale_cols = (cols + tile_size - 1) / tile_size;
  std::vector<float> input(size_t(rows) * cols);
  std::vector<float> scales(size_t(scale_rows) * scale_cols);
  for (int row = 0; row < rows; ++row)
    for (int col = 0; col < cols; ++col)
      input[size_t(row) * cols + col] =
          float((row * 11 + col * 5) % 23 - 11) * 0.25f;
  for (int row = 0; row < scale_rows; ++row)
    for (int col = 0; col < scale_cols; ++col)
      scales[size_t(row) * scale_cols + col] =
          float((row * 7 + col * 13) % 19 - 9) * 0.125f;

  std::vector<double> expected(input.size());
  for (int row = 0; row < rows; ++row)
    for (int col = 0; col < cols; ++col)
      expected[size_t(row) * cols + col] =
          double(input[size_t(row) * cols + col]) *
          scales[size_t(row / tile_size) * scale_cols + col / tile_size];

  test::DeviceArray<float> device_input(input);
  test::DeviceArray<float> device_scales(scales);
  bool passed = test::check_output<float>(name, expected, [&](float* output) {
    solve(device_input.data(), device_scales.data(), output, rows, cols, tile_size);
  }, 1e-6, 1e-6);
  passed &= device_input.unchanged(input);
  passed &= device_scales.unchanged(scales);
  return passed;
}

int main() {
  bool passed = true;
  passed &= run_case("scalar", 1, 1, 16);
  passed &= run_case("small-partial-tile", 17, 31, 16);
  passed &= run_case("rectangular-t32", 65, 33, 32);
  passed &= run_case("partial-t64", 129, 193, 64);
  passed &= run_case("partial-t128", 257, 130, 128);
  return test::finish(passed);
}
