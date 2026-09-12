#include "test_utils.h"
#include <utility>

extern "C" void solve(const float*, float*, int, int);

static bool run_case(int width, int height, int pattern) {
  size_t pixels = size_t(width) * height;
  auto input = test::random_values(3 * pixels);
  std::vector<double> reference(pixels);
  for (size_t i = 0; i < pixels; ++i) {
    for (int c = 0; c < 3; ++c) {
      if (pattern == 0) input[3 * i + c] = (input[3 * i + c] + 8) / 16;
      if (pattern >= 1 && pattern <= 3) input[3 * i + c] = c == pattern - 1 ? 1 : 0;
      if (pattern == 4) input[3 * i + c] = 1;
      if (pattern == 5) input[3 * i + c] = 0;
    }
    // Interleaved RGB pixels, with the same coefficients as the exercise.
    reference[i] = 0.299 * double(input[3 * i]) +
                   0.587 * double(input[3 * i + 1]) +
                   0.114 * double(input[3 * i + 2]);
  }
  test::DeviceArray<float> di(input);
  const char* names[] = {"random-RGB", "red", "green", "blue", "white", "black"};
  bool passed = test::check_output<float>(names[pattern], reference, [&](float* output) {
    solve(di.data(), output, width, height);
  }, 1e-7, 1e-6);
  passed &= di.unchanged(input);
  return passed;
}

int main() {
  bool passed = true;
  const std::pair<int, int> shapes[] = {
    {1, 1}, {1, 257}, {257, 1}, {7, 9}, {16, 16}, {17, 17}, {35, 17}, {257, 255}
  };
  for (auto shape : shapes) passed &= run_case(shape.first, shape.second, 0);
  for (int pattern = 1; pattern <= 5; ++pattern) passed &= run_case(35, 17, pattern);
  return test::finish(passed);
}
