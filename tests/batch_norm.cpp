#include "test_utils.h"
#include <array>
#include <limits>
#include <string>
#include <utility>

extern "C" void solve(const float*, const float*, const float*, float*, int, int, float);

struct Case {
  std::string name;
  int n, c;
  std::vector<float> input, gamma, beta;
  std::array<int, 4> offsets{{0, 0, 0, 0}}; // input, gamma, beta, output
};

static std::vector<double> reference(const Case& c, float eps) {
  // Independent FP64 Welford statistics: population variance, not N-1.
  std::vector<double> mean(c.c, 0), m2(c.c, 0);
  for (int row = 0; row < c.n; ++row) {
    const double inv_count = 1.0 / (row + 1);
    for (int col = 0; col < c.c; ++col) {
      const double value = c.input[size_t(row) * c.c + col];
      const double delta = value - mean[col];
      mean[col] += delta * inv_count;
      m2[col] += delta * (value - mean[col]);
    }
  }
  std::vector<double> inv_std(c.c);
  for (int col = 0; col < c.c; ++col)
    inv_std[col] = 1.0 / std::sqrt(m2[col] / c.n + double(eps));
  std::vector<double> expected(c.input.size());
  for (int row = 0; row < c.n; ++row)
    for (int col = 0; col < c.c; ++col) {
      const size_t i = size_t(row) * c.c + col;
      expected[i] = double(c.gamma[col]) * (double(c.input[i]) - mean[col]) *
                    inv_std[col] + double(c.beta[col]);
    }
  return expected;
}

static bool run_case(Case c) {
  constexpr float EPS = 1e-5f;
  constexpr double ATOL = 1e-4, RTOL = 1e-4;
  constexpr size_t GUARD = 4;
  constexpr float MARKER = 1234567.0f;
  const size_t count = c.input.size(), prefix = GUARD + c.offsets[3];
  // No suffix padding on any input: memcheck can detect row/channel overreads.
  test::DeviceArray<float> input(count + c.offsets[0]);
  test::DeviceArray<float> gamma(c.c + c.offsets[1]);
  test::DeviceArray<float> beta(c.c + c.offsets[2]);
  test::DeviceArray<float> output(prefix + count + GUARD);
  bool passed = true;
  double max_error = 0;
  for (int call = 0; call < 2; ++call) {
    if (call == 1) {
      // Change mean, variance and affine parameters on the same allocations.
      // Transformed values remain within the challenge's input/parameter ranges.
      for (int row = 0; row < c.n; ++row)
        for (int col = 0; col < c.c; ++col) {
          const size_t i = size_t(row) * c.c + col;
          c.input[i] = -0.5f * c.input[i] + float(col % 5 - 2) * 0.25f;
        }
      std::rotate(c.gamma.begin(), c.gamma.begin() + 1, c.gamma.end());
      for (float& value : c.beta) value = -value;
    }
    const auto expected = reference(c, EPS);
    std::vector<float> stored_input(count + c.offsets[0], MARKER);
    std::vector<float> stored_gamma(c.c + c.offsets[1], MARKER);
    std::vector<float> stored_beta(c.c + c.offsets[2], MARKER);
    std::copy(c.input.begin(), c.input.end(), stored_input.begin() + c.offsets[0]);
    std::copy(c.gamma.begin(), c.gamma.end(), stored_gamma.begin() + c.offsets[1]);
    std::copy(c.beta.begin(), c.beta.end(), stored_beta.begin() + c.offsets[2]);
    input.upload(stored_input);
    gamma.upload(stored_gamma);
    beta.upload(stored_beta);
    std::vector<float> initial(prefix + count + GUARD, MARKER);
    std::fill(initial.begin() + prefix, initial.begin() + prefix + count,
              call == 0 ? std::numeric_limits<float>::quiet_NaN() : -4321.5f);
    output.upload(initial);

    solve(input.data() + c.offsets[0], gamma.data() + c.offsets[1],
          beta.data() + c.offsets[2], output.data() + prefix, c.n, c.c, EPS);
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
      const double got = actual[prefix + i];
      const double error = std::abs(got - expected[i]);
      max_error = std::max(max_error, error);
      if (!std::isfinite(got) || !std::isfinite(expected[i]) ||
          error > ATOL + RTOL * std::abs(expected[i])) {
        if (mismatches == 0)
          std::fprintf(stderr,
                       "%s call=%d row=%zu col=%zu got=%.12g expected=%.12g\n",
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
    passed &= gamma.unchanged(stored_gamma);
    passed &= beta.unchanged(stored_beta);
  }
  std::printf("%s %-26s N=%d C=%d offsets=%d/%d/%d/%d max_abs_error=%.3g\n",
              passed ? "PASS" : "FAIL", c.name.c_str(), c.n, c.c,
              c.offsets[0], c.offsets[1], c.offsets[2], c.offsets[3], max_error);
  return passed;
}

static Case mixed_case(const std::string& name, int n, int channels) {
  Case c{name, n, channels, std::vector<float>(size_t(n) * channels),
         std::vector<float>(channels), std::vector<float>(channels)};
  std::mt19937 rng(42 + unsigned(n) * 1031 + unsigned(channels) * 97);
  std::uniform_int_distribution<int> dist(-1600, 1600);
  for (float& value : c.input) value = float(dist(rng)) / 16.0f;
  const float scales[] = {0.1f, 0.5f, 1.0f, 2.0f, 10.0f};
  const float shifts[] = {-10.0f, -1.0f, 0.0f, 1.0f, 10.0f};
  for (int col = 0; col < channels; ++col) {
    c.gamma[col] = scales[col % 5];
    c.beta[col] = shifts[(col * 3) % 5];
  }
  return c;
}

static std::vector<Case> quick_cases() {
  std::vector<Case> cases;
  // The displayed example outputs are rounded: use the mathematical reference.
  cases.push_back({"example-1", 3, 2, {1, 2, 3, 4, 5, 6}, {1, 1}, {0, 0}});
  cases.push_back({"example-2", 2, 2, {0, 1, 2, 3}, {2, 0.5f}, {1, -1}});
  for (int channels : {1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 127, 128, 129,
                        255, 256, 257, 511, 512, 513, 1023, 1024})
    cases.push_back(mixed_case("channels-" + std::to_string(channels), 7, channels));
  for (int n : {1, 2, 3, 7, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025})
    cases.push_back(mixed_case("batch-" + std::to_string(n), n, 5));
  cases.push_back(mixed_case("single-row-wide", 1, 257));
  cases.push_back(mixed_case("both-tails", 257, 257));

  auto zero = mixed_case("all-zero", 33, 17);
  std::fill(zero.input.begin(), zero.input.end(), 0.0f);
  cases.push_back(std::move(zero));
  auto constant = mixed_case("constant-channels", 257, 9);
  const float values[] = {-100, -12.5f, -0.125f, 0, 0.125f, 1, 12.5f, 50, 100};
  for (int row = 0; row < constant.n; ++row)
    for (int col = 0; col < constant.c; ++col)
      constant.input[size_t(row) * constant.c + col] = values[col];
  cases.push_back(std::move(constant));

  cases.push_back({"population-variance", 2, 5,
                   {-1, -2, -4, -8, -16, 1, 2, 4, 8, 16},
                   {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0}});
  auto independent = mixed_case("independent-channels", 8, 7);
  for (int row = 0; row < independent.n; ++row)
    for (int col = 0; col < independent.c; ++col)
      independent.input[size_t(row) * independent.c + col] =
          float(-60 + 20 * col) + (row - 3.5f) * (col + 1) * 0.125f;
  cases.push_back(std::move(independent));
  auto tiny = mixed_case("epsilon-dominated", 3, 7);
  for (int row = 0; row < tiny.n; ++row)
    for (int col = 0; col < tiny.c; ++col)
      tiny.input[size_t(row) * tiny.c + col] =
          float((row - 1) * (col + 1)) / 4096.0f;
  cases.push_back(std::move(tiny));
  auto low_variance = mixed_case("large-offset-low-variance", 256, 3);
  const float centers[] = {64, -64, 99};
  for (int row = 0; row < low_variance.n; ++row)
    for (int col = 0; col < low_variance.c; ++col)
      low_variance.input[size_t(row) * low_variance.c + col] =
          centers[col] + (row % 2 ? 1.0f : -1.0f) / 1024.0f;
  cases.push_back(std::move(low_variance));

  auto last_row = mixed_case("last-row-active", 257, 17);
  std::fill(last_row.input.begin(), last_row.input.end(), 0.0f);
  for (int col = 0; col < last_row.c; ++col)
    last_row.input[size_t(last_row.n - 1) * last_row.c + col] = float(20 + col);
  cases.push_back(std::move(last_row));
  auto last_col = mixed_case("last-channel-active", 257, 257);
  std::fill(last_col.input.begin(), last_col.input.end(), 0.0f);
  for (int row = 0; row < last_col.n; ++row)
    last_col.input[size_t(row) * last_col.c + last_col.c - 1] = float(row % 17 - 8);
  cases.push_back(std::move(last_col));
  for (int which = 0; which < 4; ++which) {
    auto c = mixed_case("unaligned-" + std::to_string(which), 17, 257);
    c.offsets[which] = 1;
    cases.push_back(std::move(c));
  }
  auto unaligned = mixed_case("unaligned-all", 33, 259);
  unaligned.offsets = {{1, 2, 3, 1}};
  cases.push_back(std::move(unaligned));

  // A constant channel must produce beta, including decimal FP32 inputs.
  // This detects drift in long FP32 mean accumulation, independently of variance.
  auto decimal = mixed_case("constant-decimal-long", 10000, 3);
  const float decimals[] = {0.1f, 99.9f, -99.9f};
  std::fill(decimal.gamma.begin(), decimal.gamma.end(), 10.0f);
  decimal.beta = {0, 0.25f, -0.5f};
  for (int row = 0; row < decimal.n; ++row)
    for (int col = 0; col < decimal.c; ++col)
      decimal.input[size_t(row) * decimal.c + col] = decimals[col];
  cases.push_back(std::move(decimal));
  return cases;
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  const bool large = argc == 2 && std::strcmp(argv[1], "--large") == 0;
  const bool list = argc == 2 && std::strcmp(argv[1], "--list-cases") == 0;
  const bool selected = argc == 3 && std::strcmp(argv[1], "--case") == 0;
  if (argc != 1 && !large && !list && !selected) {
    std::fprintf(stderr, "Usage: %s [--large | --case NAME | --list-cases]\n", argv[0]);
    return 2;
  }
  auto cases = large ? std::vector<Case>{mixed_case("large-5000", 5000, 1024),
                                        mixed_case("large-10000", 10000, 1024)} : quick_cases();
  if (list) {
    for (const auto& c : cases) std::puts(c.name.c_str());
    return 0;
  }
  if (selected) {
    auto found = std::find_if(cases.begin(), cases.end(),
                              [&](const Case& c) { return c.name == argv[2]; });
    if (found == cases.end()) {
      std::fprintf(stderr, "Unknown case: %s (see --list-cases)\n", argv[2]);
      return 2;
    }
    Case c = std::move(*found);
    cases.clear();
    cases.push_back(std::move(c));
  }
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s\nBatch normalization: FP64 Welford reference; eps=1e-5; "
              "atol=1e-4 rtol=1e-4; 2 calls per case\n", prop.name);
  int failed = 0;
  for (auto& c : cases)
    if (!run_case(std::move(c))) ++failed;
  std::printf("Batch normalization: %zu/%zu cases passed\n", cases.size() - failed, cases.size());
  return test::finish(failed == 0);
}
