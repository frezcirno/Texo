#include "test_utils.h"
#include <limits>

extern "C" void solve(const float*, float*, int);

// Use double and avoid exp(-x) overflow on large negative inputs.
static double reference_value(float value) {
  double x = value;
  if (x >= 0) return 1.0 / (1.0 + std::exp(-x));
  double e = std::exp(x);
  return e / (1.0 + e);
}

static bool run_case(const char* name, std::vector<float> input,
                     int input_offset = 0, int output_offset = 0,
                     double atol = 1e-6) {
  constexpr size_t GUARD = 4;
  constexpr float MARKER = 1234567.0f;
  constexpr double RTOL = 3e-6;
  const size_t n = input.size();
  const size_t prefix = GUARD + output_offset;
  // No input suffix padding: memcheck can catch reads past the final element.
  test::DeviceArray<float> device_input(n + input_offset);
  test::DeviceArray<float> device_output(prefix + n + GUARD);
  bool passed = true;
  double max_error = 0;
  for (int call = 0; call < 2; ++call) {
    // Reuse allocations with different inputs to catch stale results.
    if (call == 1) {
      for (float& value : input) value = -value;
    }
    std::vector<float> stored_input(n + input_offset, MARKER);
    std::copy(input.begin(), input.end(), stored_input.begin() + input_offset);
    device_input.upload(stored_input);
    std::vector<float> initial(prefix + n + GUARD, MARKER);
    // Both poisons are outside [0,1] and non-NaN, so even a missing NaN write fails.
    std::fill(initial.begin() + prefix, initial.begin() + prefix + n,
              call == 0 ? -1234.5f : 4567.0f);
    device_output.upload(initial);
    solve(device_input.data() + input_offset, device_output.data() + prefix, int(n));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto actual = device_output.download();
    for (size_t i = 0; i < actual.size(); ++i) {
      if ((i < prefix || i >= prefix + n) && actual[i] != MARKER) {
        std::fprintf(stderr, "%s call=%d output guard overwritten at %zu\n",
                     name, call + 1, i);
        passed = false;
        break;
      }
    }
    for (size_t i = 0; i < n; ++i) {
      const double expected = reference_value(input[i]);
      const double got = actual[prefix + i];
      const double error = std::abs(got - expected);
      bool matches;
      if (std::isnan(expected)) {
        matches = std::isnan(got);
      } else {
        max_error = std::max(max_error, error);
        matches = std::isfinite(got) && got >= 0 && got <= 1 &&
                  error <= atol + RTOL * std::abs(expected);
        // Signed zeros and infinities have exactly representable answers.
        if (input[i] == 0 || std::isinf(input[i])) matches &= got == expected;
      }
      if (!matches) {
        std::fprintf(stderr, "%s call=%d index=%zu x=%.9g got=%.12g expected=%.12g\n",
                     name, call + 1, i, input[i], got, expected);
        passed = false;
        break;
      }
    }
    passed &= device_input.unchanged(stored_input);
  }
  std::printf("%s %-20s N=%zu offsets=%d/%d max_abs_error=%.3g\n",
              passed ? "PASS" : "FAIL", name, n, input_offset, output_offset, max_error);
  return passed;
}

static std::vector<float> mixed_input(int n) {
  auto input = test::random_values(n, 42 + unsigned(n));
  const float special[] = {-1000, -100, -20, -1, -1e-6f, -0.0f,
                          0, 1e-6f, 1, 20, 100, 1000};
  for (int i = 0; i < std::min(n, 12); ++i) input[i] = special[i];
  return input;
}

int main() {
  bool passed = true;
  for (int n : {1, 2, 7, 31, 32, 33, 127, 128, 129, 255, 256, 257,
                511, 512, 513, 1023, 1024, 1025, 65537}) {
    passed &= run_case("mixed-boundary", mixed_input(n));
  }
  passed &= run_case("zero", std::vector<float>(257, 0.0f));
  auto negative = mixed_input(257);
  for (float& value : negative) value = -std::abs(value) - 0.5f;
  passed &= run_case("negative", negative);
  for (float& value : negative) value = -value;
  passed &= run_case("positive", negative);

  // Relative-only comparison catches loss of small, normal positive results
  // that the usual absolute tolerance would hide. expf remains finite here.
  std::vector<float> precision(8193);
  for (size_t i = 0; i < precision.size(); ++i)
    precision[i] = float(-80.0 + 160.0 * double(i) / double(precision.size() - 1));
  passed &= run_case("relative-precision", precision, 0, 0, 0);

  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float tiny = std::numeric_limits<float>::denorm_min();
  const float small = std::numeric_limits<float>::min();
  const float large = std::numeric_limits<float>::max();
  const float special[] = {-inf, -large, -1000, -100, -89, -88.7f, -80, -20,
                          -small, -tiny, -0.0f, 0.0f, tiny, small,
                          20, 80, 88.7f, 89, 100, 1000, large, inf, nan};
  std::vector<float> extremes(257);
  for (size_t i = 0; i < extremes.size(); ++i)
    extremes[i] = special[i % (sizeof(special) / sizeof(special[0]))];
  passed &= run_case("special-values", extremes);
  passed &= run_case("unaligned-input", mixed_input(257), 1, 0);
  passed &= run_case("unaligned-output", mixed_input(257), 0, 1);
  passed &= run_case("unaligned-both", mixed_input(259), 3, 1);
  passed &= run_case("large-tail", mixed_input(1048579));
  return test::finish(passed);
}
