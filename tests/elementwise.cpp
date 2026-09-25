#include "test_utils.h"
#include <limits>

// Built separately for each operator to keep standalone solve entry points.
#if defined(TEST_CLIP)
extern "C" void solve(const float*, float*, float, float, int);
#else
extern "C" void solve(const float*, float*, int);
#endif

static double reference_value(double x, double gate, float lo, float hi) {
#if defined(TEST_RELU)
  return std::max(x, 0.0);
#elif defined(TEST_LEAKY_RELU)
  return x >= 0 ? x : 0.01 * x;
#elif defined(TEST_GEGLU)
  return x * gate * 0.5 * std::erfc(-gate / std::sqrt(2.0));
#elif defined(TEST_SILU) || defined(TEST_SWIGLU)
  // Stable CPU double reference, including large negative finite inputs.
  double sigmoid = x >= 0 ? 1.0 / (1.0 + std::exp(-x))
                         : std::exp(x) / (1.0 + std::exp(x));
#if defined(TEST_SWIGLU)
  return x * sigmoid * gate;
#else
  return x * sigmoid;
#endif
#elif defined(TEST_CLIP)
  return x < lo ? lo : (x > hi ? hi : x);
#else
#error Select an elementwise TEST_* operator in the Makefile
#endif
}

static bool run_case(int n, int pattern, float lo = -1.0f, float hi = 2.0f) {
#if defined(TEST_SWIGLU) || defined(TEST_GEGLU)
  const int input_size = 2 * n; // [x0..xN-1, gate0..gateN-1], not interleaved.
#else
  const int input_size = n;
#endif
  auto input = test::random_values(input_size);
#if defined(TEST_GEGLU)
  const int activation_start = n; // GEGLU applies GELU to the second half.
#else
  const int activation_start = 0;
#endif
  const float special[] = {-1000, -100, -20, -1, -1e-6f, -0.0f, 0, 1e-6f, 1, 20, 100, 1000};
  for (int i = 0; i < std::min(n, 12); ++i) input[activation_start + i] = special[i];
  for (int i = 0; i < n; ++i) {
    float& x = input[activation_start + i];
    if (pattern == 1) x = 0;
    if (pattern == 2) x = -std::abs(x) - 0.5f;
    if (pattern == 3) x = std::abs(x) + 0.5f;
  }
#if defined(TEST_CLIP)
  if (n >= 4) {
    input[0] = lo;
    input[1] = hi;
    input[2] = std::nextafter(lo, -std::numeric_limits<float>::infinity());
    input[3] = std::nextafter(hi, std::numeric_limits<float>::infinity());
  }
#endif
  std::vector<double> reference(n);
  for (int i = 0; i < n; ++i) {
    double gate = 1;
#if defined(TEST_SWIGLU) || defined(TEST_GEGLU)
    gate = input[n + i];
#endif
    reference[i] = reference_value(input[i], gate, lo, hi);
  }
  test::DeviceArray<float> device_input(input);
  const char* names[] = {"mixed", "zero", "negative", "positive"};
#if defined(TEST_RELU) || defined(TEST_CLIP)
  double atol = 0, rtol = 0;
#elif defined(TEST_GEGLU)
  // Float erf loses relative accuracy when 1+erf approaches zero.
  double atol = 5e-6, rtol = 3e-6;
#else
  double atol = 1e-6, rtol = 3e-6;
#endif
  bool passed = test::check_output<float>(names[pattern], reference, [&](float* output) {
#if defined(TEST_CLIP)
    solve(device_input.data(), output, lo, hi, input_size);
#else
    solve(device_input.data(), output, input_size);
#endif
  }, atol, rtol);
  passed &= device_input.unchanged(input);
  return passed;
}

int main() {
  bool passed = true;
  for (int n : {1, 31, 32, 33, 255, 256, 257, 1025, 65537})
    passed &= run_case(n, 0);
  for (int pattern : {1, 2, 3}) passed &= run_case(257, pattern);
#if defined(TEST_CLIP)
  passed &= run_case(257, 0, -5, -2);
  passed &= run_case(257, 0, 2, 5);
  passed &= run_case(257, 0, 3, 3);
#endif
  return test::finish(passed);
}
