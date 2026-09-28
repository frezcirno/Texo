#include "test_utils.h"
#include <limits>
#include <string>
#include <utility>

extern "C" void solve(const float*, float, float, float*, int, float);

struct Case {
  std::string name;
  std::vector<float> input;
  float gamma = 1.0f, beta = 0.0f;
  int input_offset = 0, output_offset = 0;
  bool change_input = true;
};

static double reference_rms(const std::vector<float>& input, float eps) {
  double sum = 0;
  for (float value : input) sum += double(value) * double(value);
  return std::sqrt(sum / double(input.size()) + double(eps));
}

static bool run_case(Case c) {
  constexpr float EPS = 1e-5f;
  constexpr double ATOL = 1e-5, RTOL = 1e-5;
  constexpr size_t GUARD = 4;
  constexpr float MARKER = 1234567.0f;
  const size_t n = c.input.size(), prefix = GUARD + c.output_offset;
  // Input offsets are multiples of four floats, preserving float4 alignment.
  // Leave the input tail unpadded so memcheck can detect vector overreads.
  test::DeviceArray<float> input(n + c.input_offset);
  test::DeviceArray<float> output(prefix + n + GUARD);
  bool passed = true;
  double max_error = 0;
  std::printf("RUN  %-25s N=%zu gamma=%.4g beta=%.4g offsets=%d/%d\n",
              c.name.c_str(), n, c.gamma, c.beta, c.input_offset, c.output_offset);
  for (int call = 0; call < 2; ++call) {
    if (call == 1 && c.change_input) {
      // Reuse allocations with different values and normalization parameters.
      // All transformed inputs/parameters remain within the challenge bounds.
      std::reverse(c.input.begin(), c.input.end());
      for (float& value : c.input) value *= -0.5f;
      c.gamma = c.gamma <= 1.0f ? 2.0f : 0.5f;
      c.beta = -c.beta;
    }
    const double rms = reference_rms(c.input, EPS);
    std::vector<float> stored_input(n + c.input_offset, MARKER);
    std::copy(c.input.begin(), c.input.end(), stored_input.begin() + c.input_offset);
    input.upload(stored_input);
    std::vector<float> initial(prefix + n + GUARD, MARKER);
    std::fill(initial.begin() + prefix, initial.begin() + prefix + n,
              call == 0 ? std::numeric_limits<float>::quiet_NaN() : -4321.5f);
    output.upload(initial);

    solve(input.data() + c.input_offset, c.gamma, c.beta,
          output.data() + prefix, int(n), EPS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto actual = output.download();
    for (size_t i = 0; i < actual.size(); ++i) {
      if ((i < prefix || i >= prefix + n) && actual[i] != MARKER) {
        std::fprintf(stderr, "%s call=%d output guard overwritten at %zu\n",
                     c.name.c_str(), call + 1, i);
        passed = false;
        break;
      }
    }
    size_t mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
      const double expected = double(c.gamma) * double(c.input[i]) / rms + double(c.beta);
      const double got = actual[prefix + i], error = std::abs(got - expected);
      if (std::isfinite(error)) max_error = std::max(max_error, error);
      else max_error = std::numeric_limits<double>::infinity();
      if (!std::isfinite(got) || error > ATOL + RTOL * std::abs(expected)) {
        if (mismatches == 0)
          std::fprintf(stderr,
                       "%s call=%d index=%zu got=%.12g expected=%.12g rms=%.12g\n",
                       c.name.c_str(), call + 1, i, got, expected, rms);
        ++mismatches;
      }
    }
    if (mismatches) {
      std::fprintf(stderr, "%s call=%d mismatches=%zu/%zu\n",
                   c.name.c_str(), call + 1, mismatches, n);
      passed = false;
    }
    passed &= input.unchanged(stored_input);
  }
  std::printf("%s %-25s max_abs_error=%.3g\n",
              passed ? "PASS" : "FAIL", c.name.c_str(), max_error);
  return passed;
}

static Case mixed_case(const std::string& name, int n) {
  Case c{name, std::vector<float>(n)};
  std::mt19937 rng(42 + unsigned(n) * 97);
  std::uniform_int_distribution<int> dist(-1600, 1600);
  for (float& value : c.input) value = float(dist(rng)) / 16.0f;
  return c;
}

static std::vector<Case> quick_cases() {
  std::vector<Case> cases;
  // Published outputs are rounded; use the formula with the supplied eps.
  cases.push_back({"example-1", {1, 2, 3, 4}});
  cases.push_back({"example-2", {1, 2, 3}});
  for (int n : {1, 2, 3, 4, 5, 7, 8, 9, 31, 32, 33, 255, 256, 257,
                511, 512, 513, 1023, 1024, 1025, 4095, 4096, 4097})
    cases.push_back(mixed_case("boundary-" + std::to_string(n), n));
  const float constants[] = {0, 0.1f, -0.1f, 4, -4, 100, -100};
  for (int i = 0; i < 7; ++i)
    cases.push_back({"constant-" + std::to_string(i),
                     std::vector<float>(257, constants[i]), 10, -0.5f});
  cases.push_back({"epsilon-inside-sqrt", {1e-4f}, 10, 1});
  cases.push_back({"epsilon-dominated", {-3e-5f, -2e-5f, -1e-5f, 0, 1e-5f, 2e-5f, 3e-5f},
                   10, 0});
  for (int sign : {-1, 1}) {
    auto c = mixed_case(sign < 0 ? "all-negative" : "all-positive", 1025);
    for (float& value : c.input) value = sign * (std::abs(value) * 0.5f + 0.25f);
    cases.push_back(std::move(c));
  }
  auto alternating = mixed_case("alternating-sign", 4097);
  for (size_t i = 0; i < alternating.input.size(); ++i)
    alternating.input[i] = i % 2 ? -100.0f : 100.0f;
  cases.push_back(std::move(alternating));
  for (int lane = 0; lane < 4; ++lane) {
    Case c{"vector-lane-" + std::to_string(lane), std::vector<float>(1024, 0)};
    c.input[lane] = 8;
    c.input[1020 + lane] = -16;
    cases.push_back(std::move(c));
  }
  for (int tail = 1; tail <= 3; ++tail) {
    Case c{"tail-only-" + std::to_string(tail), std::vector<float>(1024 + tail, 0)};
    for (int i = 0; i < tail; ++i) c.input[1024 + i] = float(i + 1);
    cases.push_back(std::move(c));
  }
  Case last{"last-element", std::vector<float>(4097, 0)};
  last.input.back() = -99;
  cases.push_back(std::move(last));
  Case global{"global-rms", std::vector<float>(2048, 1), 3, -1};
  std::fill(global.input.begin() + 1024, global.input.end(), 20);
  cases.push_back(std::move(global));
  auto offset = mixed_case("aligned-input-offset", 1025);
  offset.input_offset = 4;
  cases.push_back(std::move(offset));
  for (int offset = 1; offset <= 3; ++offset) {
    auto c = mixed_case("output-offset-" + std::to_string(offset), 259);
    c.output_offset = offset;
    cases.push_back(std::move(c));
  }
  auto both = mixed_case("offset-both", 1027);
  both.input_offset = 4;
  both.output_offset = 1;
  cases.push_back(std::move(both));
  auto low = mixed_case("affine-low", 257);
  low.gamma = 0.1f;
  low.beta = -10;
  cases.push_back(std::move(low));
  auto high = mixed_case("affine-high", 257);
  high.gamma = 10;
  high.beta = 10;
  cases.push_back(std::move(high));
  auto repeat = mixed_case("repeat-identical", 1025);
  repeat.change_input = false;
  cases.push_back(std::move(repeat));
  cases.push_back(mixed_case("many-blocks-tail", 65537));
  return cases;
}

static std::vector<Case> large_cases() {
  std::vector<Case> cases;
  cases.push_back(mixed_case("large-99999", 99999));
  cases.push_back(mixed_case("large-100000", 100000));
  Case dynamic{"large-mixed-magnitudes", std::vector<float>(100000), 10, -0.5f};
  for (size_t i = 0; i < dynamic.input.size(); ++i) {
    const float magnitude = i % 1024 == 0 ? 100.0f : 1.0f / 1024;
    dynamic.input[i] = i % 3 ? -magnitude : magnitude;
  }
  cases.push_back(std::move(dynamic));
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
  auto cases = large ? large_cases() : quick_cases();
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
  std::printf("GPU: %s\nRMS normalization: CPU FP64 reference; eps=1e-5; "
              "atol=1e-5 rtol=1e-5; 2 calls per case\n", prop.name);
  int failed = 0;
  for (auto& c : cases)
    if (!run_case(std::move(c))) ++failed;
  std::printf("RMS normalization: %zu/%zu cases passed\n", cases.size() - failed, cases.size());
  return test::finish(failed == 0);
}
