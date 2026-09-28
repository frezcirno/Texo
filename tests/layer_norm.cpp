#include "test_utils.h"
#include <array>
#include <limits>
#include <string>
#include <utility>

extern "C" void solve(const float*, const float*, const float*, float*,
                     int, int, float);

struct Case {
  std::string name;
  int n, c;
  std::vector<float> input, weight, bias;
  std::array<int, 4> offsets{{0, 0, 0, 0}}; // input, weight, bias, output
  bool repeat_identical = false;
};

static std::vector<double> reference(const Case& c, float eps) {
  std::vector<double> expected(c.input.size());
  for (int row = 0; row < c.n; ++row) {
    const size_t base = size_t(row) * c.c;
    // Independent FP64 Welford statistics per row, with population variance / C.
    double mean = 0, m2 = 0;
    for (int col = 0; col < c.c; ++col) {
      const double value = c.input[base + col], delta = value - mean;
      mean += delta / double(col + 1);
      m2 += delta * (value - mean);
    }
    const double inv_std = 1.0 / std::sqrt(m2 / double(c.c) + double(eps));
    for (int col = 0; col < c.c; ++col)
      expected[base + col] = double(c.weight[col]) *
                            (double(c.input[base + col]) - mean) * inv_std +
                            double(c.bias[col]);
  }
  return expected;
}

static bool run_case(Case c) {
  constexpr float EPS = 1e-5f;
  constexpr double ATOL = 1e-4, RTOL = 1e-4;
  constexpr size_t GUARD = 4;
  constexpr float MARKER = 1234567.0f;
  const size_t count = c.input.size(), prefix = GUARD + c.offsets[3];
  // No suffix padding on inputs or parameters: catch overreads with memcheck.
  test::DeviceArray<float> input(count + c.offsets[0]);
  test::DeviceArray<float> weight(c.c + c.offsets[1]);
  test::DeviceArray<float> bias(c.c + c.offsets[2]);
  test::DeviceArray<float> output(prefix + count + GUARD);
  std::printf("RUN  %-27s N=%d C=%d offsets=%d/%d/%d/%d\n",
              c.name.c_str(), c.n, c.c,
              c.offsets[0], c.offsets[1], c.offsets[2], c.offsets[3]);
  bool passed = true;
  double max_error = 0;
  for (int call = 0; call < 2; ++call) {
    if (call == 1 && !c.repeat_identical) {
      // Change each row's mean/variance and the per-feature affine parameters.
      // Stay within the input, weight and bias ranges in the challenge.
      for (size_t i = 0; i < count; ++i)
        c.input[i] = -0.5f * c.input[i] + float(int(i / c.c % 7) - 3) * 0.25f;
      std::rotate(c.weight.begin(), c.weight.begin() + 1, c.weight.end());
      for (float& value : c.bias) value = -value;
    }
    const auto expected = reference(c, EPS);
    std::vector<float> stored_input(count + c.offsets[0], MARKER);
    std::vector<float> stored_weight(c.c + c.offsets[1], MARKER);
    std::vector<float> stored_bias(c.c + c.offsets[2], MARKER);
    std::copy(c.input.begin(), c.input.end(), stored_input.begin() + c.offsets[0]);
    std::copy(c.weight.begin(), c.weight.end(), stored_weight.begin() + c.offsets[1]);
    std::copy(c.bias.begin(), c.bias.end(), stored_bias.begin() + c.offsets[2]);
    input.upload(stored_input);
    weight.upload(stored_weight);
    bias.upload(stored_bias);
    std::vector<float> initial(prefix + count + GUARD, MARKER);
    std::fill(initial.begin() + prefix, initial.begin() + prefix + count,
              call == 0 ? std::numeric_limits<float>::quiet_NaN() : -4321.5f);
    output.upload(initial);

    solve(input.data() + c.offsets[0], weight.data() + c.offsets[1],
          bias.data() + c.offsets[2], output.data() + prefix, c.n, c.c, EPS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto actual = output.download();
    for (size_t i = 0; i < actual.size(); ++i) {
      if ((i < prefix || i >= prefix + count) && actual[i] != MARKER) {
        std::fprintf(stderr, "%s call=%d output guard overwritten at %zu\n",
                     c.name.c_str(), call + 1, i);
        passed = false;
        break;
      }
    }
    size_t mismatches = 0;
    for (size_t i = 0; i < count; ++i) {
      const double got = actual[prefix + i], error = std::abs(got - expected[i]);
      if (std::isfinite(error)) max_error = std::max(max_error, error);
      else max_error = std::numeric_limits<double>::infinity();
      if (!std::isfinite(got) || error > ATOL + RTOL * std::abs(expected[i])) {
        if (mismatches == 0)
          std::fprintf(stderr, "%s call=%d (row,col)=(%zu,%zu) got=%.12g expected=%.12g\n",
                       c.name.c_str(), call + 1, i / c.c, i % c.c, got, expected[i]);
        ++mismatches;
      }
    }
    if (mismatches) {
      std::fprintf(stderr, "%s call=%d mismatches=%zu/%zu\n",
                   c.name.c_str(), call + 1, mismatches, count);
      passed = false;
    }
    passed &= input.unchanged(stored_input);
    passed &= weight.unchanged(stored_weight);
    passed &= bias.unchanged(stored_bias);
  }
  std::printf("%s %-27s max_abs_error=%.3g\n",
              passed ? "PASS" : "FAIL", c.name.c_str(), max_error);
  return passed;
}

static Case mixed_case(const std::string& name, int n, int cols) {
  Case c{name, n, cols, std::vector<float>(size_t(n) * cols),
         std::vector<float>(cols), std::vector<float>(cols)};
  std::mt19937 rng(42 + unsigned(n) * 1031 + unsigned(cols) * 97);
  std::uniform_int_distribution<int> dist(-1600, 1600);
  for (float& value : c.input) value = float(dist(rng)) / 16.0f;
  const float scales[] = {0.1f, 0.5f, 1, 2, 10};
  const float shifts[] = {-10, -1, 0, 1, 10};
  for (int col = 0; col < cols; ++col) {
    c.weight[col] = scales[col % 5];
    c.bias[col] = shifts[(col * 3) % 5];
  }
  return c;
}

static std::vector<Case> quick_cases() {
  std::vector<Case> cases;
  // The published output is rounded; compare against the formula including eps.
  cases.push_back({"example", 2, 4, {1, 2, 3, 4, -1, 0, 0, 1},
                   {1, 1, 1, 1}, {0, 0, 0, 0}});
  cases.push_back({"scalar", 1, 1, {-7}, {2}, {1.5f}});
  for (int cols : {1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65,
                   127, 128, 129, 255, 256, 257, 511, 512, 513,
                   1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096})
    cases.push_back(mixed_case("features-" + std::to_string(cols), 3, cols));
  for (int rows : {1, 7, 31, 32, 33, 255, 256, 257})
    cases.push_back(mixed_case("rows-" + std::to_string(rows), rows, 37));
  cases.push_back(mixed_case("max-rows-single-feature", 65536, 1));

  auto zero = mixed_case("all-zero", 3, 257);
  std::fill(zero.input.begin(), zero.input.end(), 0);
  cases.push_back(std::move(zero));
  auto constant = mixed_case("constant-rows", 7, 513);
  for (size_t i = 0; i < constant.input.size(); ++i)
    constant.input[i] = float(int(i / constant.c) - 3) * 32;
  cases.push_back(std::move(constant));
  for (int cols : {256, 4096}) {
    auto decimal = mixed_case("constant-decimal-" + std::to_string(cols), 3, cols);
    const float values[] = {0.1f, 99.9f, -99.9f};
    for (size_t i = 0; i < decimal.input.size(); ++i)
      decimal.input[i] = values[i / cols];
    cases.push_back(std::move(decimal));
  }
  auto rows = mixed_case("independent-rows", 3, 35);
  for (size_t i = 0; i < rows.input.size(); ++i) {
    const int row = int(i / rows.c);
    rows.input[i] = float(row - 1) * 64 +
                    float(int(i % rows.c) - 17) * float(row + 1) * 0.125f;
  }
  cases.push_back(std::move(rows));
  auto repeated_rows = mixed_case("identical-rows", 5, 129);
  for (int row = 1; row < repeated_rows.n; ++row)
    std::copy_n(repeated_rows.input.begin(), repeated_rows.c,
                repeated_rows.input.begin() + size_t(row) * repeated_rows.c);
  cases.push_back(std::move(repeated_rows));
  auto affine = mixed_case("per-feature-affine", 3, 10);
  for (size_t i = 0; i < affine.input.size(); ++i)
    affine.input[i] = float(int(i % 10) - 5);
  cases.push_back(std::move(affine));
  cases.push_back({"population-variance", 2, 2, {-1, 1, -2, 2}, {1, 1}, {0, 0}});
  cases.push_back({"subtract-mean", 2, 3, {1, 2, 3, 9, 10, 11}, {1, 1, 1}, {0, 0, 0}});
  auto tiny = mixed_case("epsilon-dominated", 3, 32);
  for (size_t i = 0; i < tiny.input.size(); ++i)
    tiny.input[i] = float(int(i % 8) - 4) / 4096.0f;
  cases.push_back(std::move(tiny));
  auto low_variance = mixed_case("large-offset-low-variance", 3, 1024);
  const float centers[] = {64, -64, 99};
  for (size_t i = 0; i < low_variance.input.size(); ++i)
    low_variance.input[i] = centers[i / low_variance.c] +
                            (i % 2 ? 1.0f : -1.0f) / 1024;
  cases.push_back(std::move(low_variance));
  auto last_row = mixed_case("active-last-row", 33, 65);
  std::fill(last_row.input.begin(), last_row.input.end(), 0);
  for (size_t i = last_row.input.size() - last_row.c; i < last_row.input.size(); ++i)
    last_row.input[i] = float(int(i % 7) - 3);
  cases.push_back(std::move(last_row));
  auto last_col = mixed_case("active-last-column", 3, 4095);
  std::fill(last_col.input.begin(), last_col.input.end(), 0);
  for (int row = 0; row < last_col.n; ++row)
    last_col.input[size_t(row + 1) * last_col.c - 1] = float(row - 1) * 100;
  cases.push_back(std::move(last_col));
  auto extrema = mixed_case("input-extrema", 3, 512);
  for (size_t i = 0; i < extrema.input.size(); ++i)
    extrema.input[i] = i % 2 ? 100 : -100;
  cases.push_back(std::move(extrema));
  auto repeat = mixed_case("repeat-identical", 7, 513);
  repeat.repeat_identical = true;
  cases.push_back(std::move(repeat));
  for (int which = 0; which < 4; ++which) {
    auto c = mixed_case("unaligned-" + std::to_string(which), 3, 257);
    c.offsets[which] = 1;
    cases.push_back(std::move(c));
  }
  auto all = mixed_case("unaligned-all", 5, 513);
  all.offsets = {{1, 2, 3, 1}};
  cases.push_back(std::move(all));
  return cases;
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  bool large = false, list = false;
  std::string selected;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--large") == 0 && !large) large = true;
    else if (std::strcmp(argv[i], "--list-cases") == 0 && !list) list = true;
    else if (std::strcmp(argv[i], "--case") == 0 && selected.empty() && i + 1 < argc)
      selected = argv[++i];
    else {
      std::fprintf(stderr, "Usage: %s [--large] [--case NAME | --list-cases]\n", argv[0]);
      return 2;
    }
  }
  if (list && !selected.empty()) {
    std::fprintf(stderr, "--case and --list-cases cannot be combined\n");
    return 2;
  }
  auto cases = large ? std::vector<Case>{
      mixed_case("large-challenge", 65536, 512),
      mixed_case("large-wide", 1024, 4096)} : quick_cases();
  if (list) {
    for (const auto& c : cases) std::puts(c.name.c_str());
    return 0;
  }
  if (!selected.empty()) {
    auto found = std::find_if(cases.begin(), cases.end(),
                              [&](const Case& c) { return c.name == selected; });
    if (found == cases.end()) {
      std::fprintf(stderr, "Unknown case: %s (see --list-cases%s)\n",
                   selected.c_str(), large ? " --large" : "");
      return 2;
    }
    Case c = std::move(*found);
    cases.clear();
    cases.push_back(std::move(c));
  }
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s\nLayer normalization: per-row CPU FP64 Welford reference; "
              "eps=1e-5; atol=1e-4 rtol=1e-4; 2 calls per case\n", prop.name);
  int failed = 0;
  for (auto& c : cases)
    if (!run_case(std::move(c))) ++failed;
  std::printf("Layer normalization: %zu/%zu cases passed\n", cases.size() - failed, cases.size());
  return test::finish(failed == 0);
}
