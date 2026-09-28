#include "test_utils.h"
#include <limits>
#include <string>
#include <utility>

extern "C" void solve(const float*, float*, int, int, int, int, int, int, int);

struct Case {
  std::string name;
  int n, c, h, w, kernel, stride, padding;
  std::vector<float> input;
  int input_offset = 0, output_offset = 0;
};

static int output_size(int input, const Case& c) {
  // All test cases have nonnegative numerators and positive output dimensions.
  return (input + 2 * c.padding - c.kernel) / c.stride + 1;
}

static std::vector<float> reference(const Case& c) {
  const int oh = output_size(c.h, c), ow = output_size(c.w, c);
  std::vector<float> expected;
  expected.reserve(size_t(c.n) * c.c * oh * ow);
  for (int n = 0; n < c.n; ++n)
    for (int channel = 0; channel < c.c; ++channel)
      for (int y = 0; y < oh; ++y)
        for (int x = 0; x < ow; ++x) {
          const int top = y * c.stride - c.padding;
          const int left = x * c.stride - c.padding;
          float value = -std::numeric_limits<float>::infinity();
          // Clip the window to real input coordinates: padding cannot win a max.
          for (int row = std::max(0, top); row < std::min(c.h, top + c.kernel); ++row)
            for (int col = std::max(0, left); col < std::min(c.w, left + c.kernel); ++col)
              value = std::max(value, c.input[((size_t(n) * c.c + channel) * c.h + row) * c.w + col]);
          expected.push_back(value);
        }
  return expected;
}

static bool run_case(Case c) {
  constexpr size_t GUARD = 4;
  constexpr float MARKER = 1234567.0f;
  const int oh = output_size(c.h, c), ow = output_size(c.w, c);
  const size_t count = size_t(c.n) * c.c * oh * ow;
  const size_t prefix = GUARD + c.output_offset;
  // No input suffix padding, so memcheck can detect reads past the final plane.
  test::DeviceArray<float> input(c.input.size() + c.input_offset);
  test::DeviceArray<float> output(prefix + count + GUARD);
  std::printf("RUN  %-24s NCHW=%d/%d/%d/%d k/s/p=%d/%d/%d out=%dx%d offsets=%d/%d\n",
              c.name.c_str(), c.n, c.c, c.h, c.w, c.kernel, c.stride, c.padding,
              oh, ow, c.input_offset, c.output_offset);
  bool passed = true;
  for (int call = 0; call < 2; ++call) {
    if (call == 1) {
      // Change both locations and values of maxima on the same device buffers.
      std::reverse(c.input.begin(), c.input.end());
      for (float& value : c.input) value = -0.5f * value - 0.25f;
    }
    const auto expected = reference(c);
    std::vector<float> stored_input(c.input.size() + c.input_offset, MARKER);
    std::copy(c.input.begin(), c.input.end(), stored_input.begin() + c.input_offset);
    input.upload(stored_input);
    std::vector<float> initial(prefix + count + GUARD, MARKER);
    std::fill(initial.begin() + prefix, initial.begin() + prefix + count,
              call == 0 ? std::numeric_limits<float>::quiet_NaN() : -4321.5f);
    output.upload(initial);

    solve(input.data() + c.input_offset, output.data() + prefix,
          c.n, c.c, c.h, c.w, c.kernel, c.stride, c.padding);
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
      // Max selects an input value; finite inputs need no rounding tolerance.
      if (actual[prefix + i] != expected[i]) {
        if (mismatches == 0) {
          const size_t plane = i / (size_t(oh) * ow);
          std::fprintf(stderr,
                       "%s call=%d (n,c,h,w)=(%zu,%zu,%zu,%zu) got=%.9g expected=%.9g\n",
                       c.name.c_str(), call + 1, plane / c.c, plane % c.c,
                       i / ow % oh, i % ow, actual[prefix + i], expected[i]);
        }
        ++mismatches;
      }
    }
    if (mismatches) {
      std::fprintf(stderr, "%s call=%d mismatches=%zu/%zu\n",
                   c.name.c_str(), call + 1, mismatches, count);
      passed = false;
    }
    passed &= input.unchanged(stored_input);
  }
  std::printf("%s %s\n", passed ? "PASS" : "FAIL", c.name.c_str());
  return passed;
}

static Case mixed_case(const std::string& name, int n, int channels, int h, int w,
                       int kernel, int stride, int padding) {
  Case c{name, n, channels, h, w, kernel, stride, padding,
         std::vector<float>(size_t(n) * channels * h * w)};
  std::mt19937 rng(42 + unsigned(h) * 1031 + unsigned(w) * 97 + unsigned(channels));
  std::uniform_int_distribution<int> dist(-1600, 1600);
  for (float& value : c.input) value = float(dist(rng)) / 16.0f;
  return c;
}

static std::vector<Case> quick_cases() {
  std::vector<Case> cases;
  cases.push_back({"example-1", 1, 1, 3, 3, 2, 1, 0, {1, 2, 3, 4, 5, 6, 7, 8, 9}});
  auto example2 = mixed_case("example-2", 1, 1, 5, 5, 3, 1, 1);
  for (size_t i = 0; i < example2.input.size(); ++i) example2.input[i] = float(i + 1);
  cases.push_back(std::move(example2));
  // Pin both published answers independently to sanity-check the CPU oracle.
  const std::vector<float> answer1{5, 6, 8, 9};
  const std::vector<float> answer2{7, 8, 9, 10, 10, 12, 13, 14, 15, 15,
                                   17, 18, 19, 20, 20, 22, 23, 24, 25, 25,
                                   22, 23, 24, 25, 25};
  if (reference(cases[0]) != answer1 || reference(cases[1]) != answer2) {
    std::fprintf(stderr, "CPU reference disagrees with the published examples\n");
    std::exit(EXIT_FAILURE);
  }

  cases.push_back({"scalar-negative", 1, 1, 1, 1, 1, 1, 0, {-7}});
  cases.push_back(mixed_case("identity-rectangular", 2, 3, 17, 19, 1, 1, 0));
  cases.push_back(mixed_case("single-row", 2, 3, 1, 33, 3, 1, 1));
  cases.push_back(mixed_case("single-column", 2, 3, 33, 1, 3, 1, 1));
  cases.push_back(mixed_case("downsample-2x2", 1, 1, 4, 4, 2, 2, 0));
  cases.push_back(mixed_case("floor-tail", 2, 3, 8, 11, 3, 2, 0));
  cases.push_back(mixed_case("global-window", 2, 3, 7, 7, 7, 1, 0));
  for (int size : {15, 16, 17, 31, 32, 33}) {
    cases.push_back(mixed_case("height-" + std::to_string(size), 2, 3, size, 19, 3, 1, 1));
    cases.push_back(mixed_case("width-" + std::to_string(size), 2, 3, 19, size, 3, 1, 1));
  }
  for (int kernel = 1; kernel <= 16; ++kernel)
    cases.push_back(mixed_case("kernel-" + std::to_string(kernel), 2, 3,
                               2 * kernel + 1, 3 * kernel + 3,
                               kernel, 1 + kernel % 3, kernel / 2));
  for (int stride = 2; stride <= 16; ++stride)
    cases.push_back(mixed_case("stride-" + std::to_string(stride), 2, 3,
                               3 * stride + 3, 3 * stride + 7, 3, stride, 1));

  for (int padding : {0, 1}) {
    auto negative = mixed_case("all-negative-p" + std::to_string(padding),
                               2, 3, 5, 7, 3, 1, padding);
    for (size_t i = 0; i < negative.input.size(); ++i)
      negative.input[i] = -float(1 + i % 127) / 8.0f;
    cases.push_back(std::move(negative));
  }
  auto zeros = mixed_case("all-zero", 2, 3, 17, 19, 3, 1, 1);
  std::fill(zeros.input.begin(), zeros.input.end(), 0.0f);
  cases.push_back(std::move(zeros));
  auto planes = mixed_case("independent-planes", 3, 5, 7, 9, 3, 1, 0);
  for (size_t i = 0; i < planes.input.size(); ++i)
    planes.input[i] = float(int(i / (planes.h * planes.w)) - 7) * 4.0f;
  cases.push_back(std::move(planes));
  auto corners = mixed_case("last-plane-corner", 2, 3, 17, 19, 3, 1, 1);
  std::fill(corners.input.begin(), corners.input.end(), -8.0f);
  corners.input.back() = 99.0f;
  cases.push_back(std::move(corners));
  auto ties = mixed_case("repeated-maxima", 2, 3, 9, 11, 3, 2, 1);
  for (size_t i = 0; i < ties.input.size(); ++i) ties.input[i] = i % 3 ? -9.0f : 5.0f;
  cases.push_back(std::move(ties));
  cases.push_back(mixed_case("batch-100", 100, 3, 5, 7, 3, 1, 1));
  cases.push_back(mixed_case("channels-512", 2, 512, 5, 7, 3, 1, 1));
  cases.push_back(mixed_case("planes-51200", 100, 512, 1, 1, 1, 1, 0));
  for (int which = 0; which < 3; ++which) {
    auto c = mixed_case("unaligned-" + std::to_string(which), 2, 3, 17, 19, 3, 1, 1);
    c.input_offset = which == 1 ? 0 : 1;
    c.output_offset = which == 0 ? 0 : 1;
    cases.push_back(std::move(c));
  }
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
      mixed_case("large-spatial", 4, 8, 1024, 1023, 3, 2, 1),
      mixed_case("large-channels", 4, 512, 63, 65, 3, 2, 1)} : quick_cases();
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
  std::printf("GPU: %s\n2D max pooling: NCHW, floor output shape, negative-infinity "
              "padding; exact CPU comparison; 2 calls per case\n", prop.name);
  int failed = 0;
  for (auto& c : cases)
    if (!run_case(std::move(c))) ++failed;
  std::printf("2D max pooling: %zu/%zu cases passed\n", cases.size() - failed, cases.size());
  return test::finish(failed == 0);
}
