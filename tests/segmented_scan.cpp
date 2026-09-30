#include "test_utils.h"
#include <string>

extern "C" void solve(const float*, const int*, float*, int);

struct Case {
  std::string name;
  std::vector<float> values;
  std::vector<int> flags;
  int input_offset = 0;
  int output_offset = 0;
};

static std::vector<float> reference(const std::vector<float>& values,
                                    const std::vector<int>& flags) {
  std::vector<float> output(values.size());
  float sum = 0.0f;
  for (size_t i = 0; i < values.size(); ++i) {
    if (flags[i] != 0) sum = 0.0f;
    output[i] = sum;
    sum += values[i];
  }
  return output;
}

template <typename IsHead>
static Case make_case(const std::string& name, int n, IsHead is_head) {
  Case c;
  c.name = name;
  c.values.resize(n);
  c.flags.resize(n);
  for (int i = 0; i < n; ++i) {
    // Binary fractions keep every tested prefix exactly representable in FP32.
    c.values[i] = float((i * 29 + n * 7 + 11) % 31 - 15) * 0.125f;
    c.flags[i] = i == 0 || is_head(i);
  }
  return c;
}

static bool run_case(const Case& c) {
  constexpr size_t GUARD = 4;
  constexpr float FLOAT_MARKER = 1234567.0f;
  constexpr int INT_MARKER = 0x13579bdf;
  const size_t n = c.values.size();
  const size_t input_prefix = GUARD + size_t(c.input_offset);
  const size_t output_prefix = GUARD + size_t(c.output_offset);

  if (c.flags.size() != n || c.input_offset < 0 || c.output_offset < 0 ||
      (n != 0 && c.flags[0] == 0)) {
    std::fprintf(stderr, "%s: invalid test case\n", c.name.c_str());
    return false;
  }

  const std::vector<float> expected = reference(c.values, c.flags);
  std::vector<float> stored_values(input_prefix + n + GUARD, FLOAT_MARKER);
  std::vector<int> stored_flags(input_prefix + n + GUARD, INT_MARKER);
  std::copy(c.values.begin(), c.values.end(), stored_values.begin() + input_prefix);
  std::copy(c.flags.begin(), c.flags.end(), stored_flags.begin() + input_prefix);
  std::vector<float> initial_output(output_prefix + n + GUARD, FLOAT_MARKER);

  test::DeviceArray<float> values(stored_values);
  test::DeviceArray<int> flags(stored_flags);
  test::DeviceArray<float> output(initial_output);
  bool passed = true;

  // Repeated calls catch unintended state retained between launches.
  for (int call = 0; call < 2; ++call) {
    output.upload(initial_output);
    solve(values.data() + input_prefix, flags.data() + input_prefix,
          output.data() + output_prefix, int(n));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto actual = output.download();
    for (size_t i = 0; i < actual.size(); ++i) {
      if ((i < output_prefix || i >= output_prefix + n) &&
          actual[i] != FLOAT_MARKER) {
        std::fprintf(stderr, "%s call=%d output guard overwritten at %zu\n",
                     c.name.c_str(), call + 1, i);
        passed = false;
        break;
      }
    }
    for (size_t i = 0; i < n; ++i) {
      if (actual[output_prefix + i] != expected[i]) {
        std::fprintf(stderr,
                     "%s call=%d index=%zu got=%.9g expected=%.9g flag=%d\n",
                     c.name.c_str(), call + 1, i, actual[output_prefix + i],
                     expected[i], c.flags[i]);
        passed = false;
        break;
      }
    }
    passed &= values.unchanged(stored_values);
    passed &= flags.unchanged(stored_flags);
  }

  std::printf("%s %-28s N=%zu offsets=%d/%d\n", passed ? "PASS" : "FAIL",
              c.name.c_str(), n, c.input_offset, c.output_offset);
  return passed;
}

int main() {
  bool passed = true;
  passed &= run_case({"empty", {}, {}});
  passed &= run_case({"problem-example", {1, 2, 3, 4, 5, 6}, {1, 0, 0, 1, 0, 1}});

  for (int n : {1, 31, 32, 33, 255, 256, 257, 511, 512, 513, 1025})
    passed &= run_case(make_case("single-segment-" + std::to_string(n), n,
                                 [](int) { return false; }));

  passed &= run_case(make_case("every-element-head", 513,
                               [](int) { return true; }));
  passed &= run_case(make_case("periodic-heads", 1025,
                               [](int i) { return (i * 17 + 5) % 23 == 0; }));

  Case boundaries = make_case("heads-near-boundaries", 1025, [](int) { return false; });
  for (int i : {31, 32, 33, 255, 256, 257, 511, 512, 513, 767, 768, 769})
    boundaries.flags[i] = 1;
  passed &= run_case(boundaries);

  Case unaligned = make_case("unaligned-input-output", 513,
                             [](int i) { return i % 47 == 0; });
  unaligned.input_offset = 3;
  unaligned.output_offset = 1;
  passed &= run_case(unaligned);

  // 257 first-level blocks force a recursive scan of block summaries.
  passed &= run_case(make_case("recursive-single-segment", 256 * 256 + 7,
                               [](int) { return false; }));
  passed &= run_case(make_case("recursive-sparse-heads", 256 * 256 + 7,
                               [](int i) { return i % 1009 == 0; }));
  return test::finish(passed);
}
