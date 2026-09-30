#include "test_utils.h"

extern "C" void solve(float*, float*, float*, float*, int, int);

static bool run_case(const char* name, int rows, int dims) {
  const size_t count = size_t(rows) * dims;
  std::vector<float> q(count), cosine(count), sine(count);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < dims / 2; ++col) {
      q[size_t(row) * dims + col] = float((row * 11 + col * 3) % 17 - 8) * 0.25f;
      q[size_t(row) * dims + dims / 2 + col] =
          float((row * 7 + col * 5) % 19 - 9) * 0.125f;
      const float c = float((row * 5 + col * 7) % 13 - 6) * 0.125f;
      const float s = float((row * 3 + col * 11) % 17 - 8) * 0.125f;
      cosine[size_t(row) * dims + col] = cosine[size_t(row) * dims + dims / 2 + col] = c;
      sine[size_t(row) * dims + col] = sine[size_t(row) * dims + dims / 2 + col] = s;
    }
  }
  std::vector<double> expected(count);
  for (int row = 0; row < rows; ++row)
    for (int col = 0; col < dims; ++col) {
      const int other = col < dims / 2 ? col + dims / 2 : col - dims / 2;
      const double rotated = col < dims / 2 ? -double(q[size_t(row) * dims + other])
                                             : double(q[size_t(row) * dims + other]);
      expected[size_t(row) * dims + col] =
          double(q[size_t(row) * dims + col]) * cosine[size_t(row) * dims + col] +
          rotated * sine[size_t(row) * dims + col];
    }

  test::DeviceArray<float> device_q(q), device_cosine(cosine), device_sine(sine);
  bool passed = test::check_output<float>(name, expected, [&](float* output) {
    solve(device_q.data(), device_cosine.data(), device_sine.data(), output, rows, dims);
  }, 1e-6, 1e-6);
  passed &= device_q.unchanged(q);
  passed &= device_cosine.unchanged(cosine);
  passed &= device_sine.unchanged(sine);
  return passed;
}

int main() {
  bool passed = true;
  passed &= run_case("minimal-d2", 1, 2);
  passed &= run_case("example-shape", 2, 4);
  passed &= run_case("partial-warp", 17, 18);
  passed &= run_case("wide-head", 33, 130);
  passed &= run_case("many-rows", 257, 128);
  return test::finish(passed);
}
