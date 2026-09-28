#include "test_utils.h"
#include <array>
#include <limits>
#include <string>
#include <utility>

extern "C" void solve(const float*, const float*, const float*, float*,
                     int, int, int, int, int, float);

struct Case {
  std::string name;
  int n, c, h, w, groups;
  std::vector<float> input, gamma, beta;
  std::array<int, 4> offsets{{0, 0, 0, 0}}; // input, gamma, beta, output
};

static std::vector<double> reference(const Case& c, float eps) {
  // Each (batch, group) occupies one contiguous span in NCHW storage.
  const size_t spatial = size_t(c.h) * c.w;
  const size_t group_size = size_t(c.c / c.groups) * spatial;
  std::vector<double> expected(c.input.size());
  for (size_t group = 0; group < size_t(c.n) * c.groups; ++group) {
    const size_t base = group * group_size;
    double mean = 0, m2 = 0;
    // Independent FP64 Welford statistics, with population variance (divide by M).
    for (size_t i = 0; i < group_size; ++i) {
      const double value = c.input[base + i], delta = value - mean;
      mean += delta / double(i + 1);
      m2 += delta * (value - mean);
    }
    const double inv_std = 1.0 / std::sqrt(m2 / double(group_size) + double(eps));
    for (size_t i = base; i < base + group_size; ++i) {
      const size_t channel = i / spatial % c.c;
      expected[i] = double(c.gamma[channel]) * (double(c.input[i]) - mean) *
                    inv_std + double(c.beta[channel]);
    }
  }
  return expected;
}

static bool run_case(Case c) {
  constexpr float EPS = 1e-5f;
  constexpr double ATOL = 1e-4, RTOL = 1e-4;
  constexpr size_t GUARD = 4;
  constexpr float MARKER = 1234567.0f;
  const size_t count = c.input.size(), prefix = GUARD + c.offsets[3];
  const size_t group_size = size_t(c.c / c.groups) * c.h * c.w;
  // No suffix padding on inputs or parameters: catch overreads with memcheck.
  test::DeviceArray<float> input(count + c.offsets[0]);
  test::DeviceArray<float> gamma(c.c + c.offsets[1]);
  test::DeviceArray<float> beta(c.c + c.offsets[2]);
  test::DeviceArray<float> output(prefix + count + GUARD);
  std::printf("RUN  %-25s NCHW=%d/%d/%d/%d G=%d M=%zu offsets=%d/%d/%d/%d\n",
              c.name.c_str(), c.n, c.c, c.h, c.w, c.groups, group_size,
              c.offsets[0], c.offsets[1], c.offsets[2], c.offsets[3]);
  bool passed = true;
  double max_error = 0;
  for (int call = 0; call < 2; ++call) {
    if (call == 1) {
      // Change each group's mean/variance and per-channel affine parameters.
      // Inputs stay in [-100,100], gamma in [0.1,10], and beta in [-10,10].
      for (size_t i = 0; i < count; ++i)
        c.input[i] = -0.5f * c.input[i] + float(int(i / group_size % 7) - 3) * 0.25f;
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
          beta.data() + c.offsets[2], output.data() + prefix,
          c.n, c.c, c.h, c.w, c.groups, EPS);
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
          std::fprintf(stderr,
                       "%s call=%d (n,c,h,w)=(%zu,%zu,%zu,%zu) got=%.12g expected=%.12g\n",
                       c.name.c_str(), call + 1, i / (size_t(c.c) * c.h * c.w),
                       i / (size_t(c.h) * c.w) % c.c, i / c.w % c.h, i % c.w,
                       got, expected[i]);
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
  std::printf("%s %-25s max_abs_error=%.3g\n",
              passed ? "PASS" : "FAIL", c.name.c_str(), max_error);
  return passed;
}

static Case mixed_case(const std::string& name, int n, int channels, int h, int w, int groups) {
  Case c{name, n, channels, h, w, groups,
         std::vector<float>(size_t(n) * channels * h * w),
         std::vector<float>(channels), std::vector<float>(channels)};
  std::mt19937 rng(42 + unsigned(n) * 1031 + unsigned(channels) * 97 + unsigned(h * w));
  std::uniform_int_distribution<int> dist(-1600, 1600);
  for (float& value : c.input) value = float(dist(rng)) / 16.0f;
  const float scales[] = {0.1f, 0.5f, 1, 2, 10};
  const float shifts[] = {-10, -1, 0, 1, 10};
  for (int channel = 0; channel < channels; ++channel) {
    c.gamma[channel] = scales[channel % 5];
    c.beta[channel] = shifts[(channel * 3) % 5];
  }
  return c;
}

static std::vector<Case> quick_cases() {
  std::vector<Case> cases;
  // The displayed answers omit the small eps effect; use the exact formula.
  cases.push_back({"example-1", 1, 4, 2, 2, 2,
                   {1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 6, 6, 6, 6},
                   {1, 1, 1, 1}, {0, 0, 0, 0}});
  cases.push_back({"example-2", 1, 2, 1, 2, 2, {1, 3, 2, 6}, {2, 1}, {0, 0}});
  cases.push_back({"scalar", 1, 1, 1, 1, 1, {-7}, {2}, {1.5f}});
  for (int m : {1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65,
                127, 128, 129, 255, 256, 257, 511, 512})
    cases.push_back(mixed_case("group-size-" + std::to_string(m), 2, 2 * m, 1, 1, 2));
  cases.push_back(mixed_case("group-size-1023", 2, 3, 31, 33, 3));
  cases.push_back(mixed_case("group-size-1024", 2, 3, 32, 32, 3));
  cases.push_back(mixed_case("group-size-1025", 2, 3, 25, 41, 3));
  for (int groups : {1, 2, 3, 4, 6, 12})
    cases.push_back(mixed_case("groups12-" + std::to_string(groups), 3, 12, 3, 7, groups));
  for (int groups : {1, 2, 4, 8, 16, 32})
    cases.push_back(mixed_case("groups32-" + std::to_string(groups), 2, 32, 5, 9, groups));
  cases.push_back(mixed_case("max-batches-and-groups", 32, 1024, 1, 1, 1024));
  cases.push_back(mixed_case("rectangular-tails", 3, 12, 17, 19, 3));
  cases.push_back(mixed_case("single-column", 2, 6, 33, 1, 3));
  cases.push_back(mixed_case("single-row", 2, 6, 1, 33, 3));

  auto zero = mixed_case("all-zero", 2, 6, 3, 7, 3);
  std::fill(zero.input.begin(), zero.input.end(), 0);
  cases.push_back(std::move(zero));
  auto constant = mixed_case("constant-groups", 3, 8, 5, 7, 4);
  for (size_t i = 0; i < constant.input.size(); ++i)
    constant.input[i] = float(int(i / 70) - 6) * 8;
  cases.push_back(std::move(constant));
  auto decimal = mixed_case("constant-decimal", 2, 6, 4, 8, 3);
  const float decimals[] = {0.1f, 99.9f, -99.9f};
  for (size_t i = 0; i < decimal.input.size(); ++i)
    decimal.input[i] = decimals[i / 64 % 3];
  cases.push_back(std::move(decimal));
  auto channels = mixed_case("constant-channels", 2, 12, 3, 5, 3);
  for (size_t i = 0; i < channels.input.size(); ++i)
    channels.input[i] = float(int(i / 15 % 12) - 6) * 8;
  cases.push_back(std::move(channels));
  auto batches = mixed_case("independent-batches", 3, 8, 2, 3, 4);
  for (size_t i = 0; i < batches.input.size(); ++i) {
    const int n = int(i / 48);
    batches.input[i] = float(n - 1) * 48 +
                       float(int(i % 12) - 5) * float(n + 1) * 0.125f;
  }
  cases.push_back(std::move(batches));
  auto groups = mixed_case("independent-groups", 2, 12, 2, 3, 4);
  for (size_t i = 0; i < groups.input.size(); ++i) {
    const int g = int(i / 18 % 4);
    groups.input[i] = float(-60 + g * 40) +
                      float(int(i % 18) - 8) * float(g + 1) * 0.125f;
  }
  cases.push_back(std::move(groups));
  auto affine = mixed_case("per-channel-affine", 2, 10, 2, 2, 5);
  for (size_t i = 0; i < affine.input.size(); ++i)
    affine.input[i] = float(int(i % 4) - 2);
  cases.push_back(std::move(affine));
  cases.push_back({"population-variance", 1, 2, 1, 2, 2, {-1, 1, -2, 2}, {1, 1}, {0, 0}});
  auto tiny = mixed_case("epsilon-dominated", 2, 6, 2, 2, 3);
  for (size_t i = 0; i < tiny.input.size(); ++i)
    tiny.input[i] = float(int(i % 8) - 4) / 4096.0f;
  cases.push_back(std::move(tiny));
  auto low_variance = mixed_case("large-offset-low-variance", 2, 6, 4, 8, 3);
  const float centers[] = {64, -64, 99};
  for (size_t i = 0; i < low_variance.input.size(); ++i)
    low_variance.input[i] = centers[i / 64 % 3] + (i % 2 ? 1.0f : -1.0f) / 1024;
  cases.push_back(std::move(low_variance));
  auto last_group = mixed_case("last-batch-group", 3, 8, 3, 5, 4);
  std::fill(last_group.input.begin(), last_group.input.end(), 0);
  for (size_t i = last_group.input.size() - 30; i < last_group.input.size(); ++i)
    last_group.input[i] = float(int(i % 7) - 3);
  cases.push_back(std::move(last_group));
  auto last_pixel = mixed_case("last-spatial-element", 2, 6, 17, 19, 3);
  std::fill(last_pixel.input.begin(), last_pixel.input.end(), 0);
  last_pixel.input.back() = 100;
  cases.push_back(std::move(last_pixel));
  for (int which = 0; which < 4; ++which) {
    auto c = mixed_case("unaligned-" + std::to_string(which), 2, 12, 3, 7, 3);
    c.offsets[which] = 1;
    cases.push_back(std::move(c));
  }
  auto all = mixed_case("unaligned-all", 3, 15, 5, 7, 5);
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
      mixed_case("large-challenge", 8, 512, 64, 64, 32),
      mixed_case("large-spatial", 2, 32, 128, 128, 8)} : quick_cases();
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
  std::printf("GPU: %s\nGroup normalization: NCHW; per-(batch,group) CPU FP64 Welford "
              "reference; eps=1e-5; atol=1e-4 rtol=1e-4; 2 calls per case\n", prop.name);
  int failed = 0;
  for (auto& c : cases)
    if (!run_case(std::move(c))) ++failed;
  std::printf("Group normalization: %zu/%zu cases passed\n", cases.size() - failed, cases.size());
  return test::finish(failed == 0);
}
