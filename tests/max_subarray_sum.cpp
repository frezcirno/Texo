#include "test_utils.h"
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

extern "C" void solve(const int*, int*, int, int);

enum class OutputMode { Lowest, Poisoned, Reused };
constexpr int64_t NO_KNOWN_RESULT = std::numeric_limits<int64_t>::lowest();

struct Case {
  std::string name;
  int n, window;
  std::vector<int> input;
  OutputMode mode = OutputMode::Lowest;
  int input_offset = 0, output_offset = 0;
  int next_window = 0;
  int fill = 0; // Nonzero: constant input. Otherwise generate deterministic values.
  int64_t known_first = NO_KNOWN_RESULT;
};

static void prepare_input(Case& c) {
  if (!c.input.empty()) return;
  c.input.resize(c.n);
  std::mt19937 rng(42 + unsigned(c.n) * 97 + unsigned(c.window) * 1031);
  std::uniform_int_distribution<int> dist(-10, 10);
  for (int& value : c.input) value = c.fill ? c.fill : dist(rng);
}

static int64_t reference(const std::vector<int>& input, int window) {
  // CPU sliding window, independent of the GPU prefix scans/reduction.
  // Windows are nonempty and have EXACTLY window elements, even if all negative.
  int64_t sum = 0;
  for (int i = 0; i < window; ++i) sum += input[i];
  int64_t best = sum;
  for (size_t end = window; end < input.size(); ++end) {
    sum += int64_t(input[end]) - int64_t(input[end - window]);
    best = std::max(best, sum);
  }
  return best;
}

static bool run_case(Case c) {
  constexpr size_t GUARD = 4;
  constexpr int MARKER = 123456789;
  const size_t prefix = GUARD + c.output_offset;
  const int calls = c.mode == OutputMode::Reused ? 3 : 2;
  prepare_input(c);
  std::printf("RUN  %-28s N=%d window=%d offsets=%d/%d\n", c.name.c_str(),
              c.n, c.window, c.input_offset, c.output_offset);
  if (c.n < 1 || c.n > 50000 || c.input.size() != size_t(c.n) ||
      c.window < 1 || c.window > c.n || c.next_window < 0 || c.next_window > c.n ||
      std::any_of(c.input.begin(), c.input.end(), [](int x) { return x < -10 || x > 10; })) {
    std::fprintf(stderr, "%s: invalid test input\n", c.name.c_str());
    return false;
  }
  // Input has no suffix padding: memcheck can detect tail overreads.
  test::DeviceArray<int> input(c.n + c.input_offset);
  test::DeviceArray<int> output(prefix + 1 + GUARD);
  bool passed = true;
  for (int call = 0; call < calls; ++call) {
    if (call == calls - 1) {
      std::reverse(c.input.begin(), c.input.end());
      for (int& value : c.input) value = -value;
      if (c.next_window) c.window = c.next_window;
    }
    const int64_t expected = reference(c.input, c.window);
    if (call == 0 && c.known_first != NO_KNOWN_RESULT && expected != c.known_first) {
      std::fprintf(stderr, "%s: CPU reference=%lld, known result=%lld\n",
                   c.name.c_str(), static_cast<long long>(expected),
                   static_cast<long long>(c.known_first));
      return false;
    }
    std::vector<int> stored_input(c.n + c.input_offset, MARKER);
    std::copy(c.input.begin(), c.input.end(), stored_input.begin() + c.input_offset);
    input.upload(stored_input);
    if (c.mode != OutputMode::Reused || call == 0) {
      std::vector<int> initial(prefix + 1 + GUARD, MARKER);
      initial[prefix] = c.mode == OutputMode::Poisoned && call == 0
                           ? std::numeric_limits<int>::max()
                           : std::numeric_limits<int>::lowest();
      output.upload(initial);
    }
    solve(input.data() + c.input_offset, output.data() + prefix, c.n, c.window);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto actual = output.download();
    if (int64_t(actual[prefix]) != expected) {
      std::fprintf(stderr, "%s call=%d N=%d window=%d got=%d expected=%lld\n",
                   c.name.c_str(), call + 1, c.n, c.window, actual[prefix],
                   static_cast<long long>(expected));
      passed = false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
      if (i != prefix && actual[i] != MARKER) {
        std::fprintf(stderr, "%s call=%d output guard overwritten at %zu\n",
                     c.name.c_str(), call + 1, i);
        passed = false;
        break;
      }
    }
    passed &= input.unchanged(stored_input);
  }
  const char* mode = c.mode == OutputMode::Lowest ? "INT_MIN" :
                     c.mode == OutputMode::Poisoned ? "poisoned" : "reused";
  std::printf("%s %-28s output=%s calls=%d\n",
              passed ? "PASS" : "FAIL", c.name.c_str(), mode, calls);
  return passed;
}

static Case known_case(const std::string& name, int window,
                       std::vector<int> values, int64_t expected) {
  Case c{name, int(values.size()), window, std::move(values)};
  c.known_first = expected;
  return c;
}

static std::vector<Case> quick_cases() {
  std::vector<Case> cases;
  cases.push_back(known_case("example-1", 2, {1, 2, 4, 2, 3}, 6));
  cases.push_back(known_case("example-2", 3, {-1, -4, -2, 1}, -5));
  cases.push_back(known_case("scalar-positive", 1, {7}, 7));
  cases.push_back(known_case("scalar-negative", 1, {-7}, -7));
  cases.push_back(known_case("scalar-zero", 1, {0}, 0));
  cases.push_back(known_case("all-negative", 2, {-6, -1, -5, -2}, -6));
  cases.push_back(known_case("all-zero", 33, std::vector<int>(257, 0), 0));
  cases.push_back(known_case("all-positive", 257, std::vector<int>(513, 10), 2570));
  cases.push_back(known_case("all-minus-ten", 255, std::vector<int>(513, -10), -2550));
  for (int window : {32, 33}) {
    Case c{"alternating-" + std::to_string(window), 513, window, std::vector<int>(513)};
    for (int i = 0; i < c.n; ++i) c.input[i] = i % 2 ? -10 : 10;
    c.known_first = window % 2 ? 10 : 0;
    cases.push_back(std::move(c));
  }
  // One array, four window sizes: catches ignoring window_size.
  const int expected_by_window[] = {6, 10, 1, 6};
  for (int window = 1; window <= 4; ++window)
    cases.push_back(known_case("fixed-window-" + std::to_string(window), window,
                               {6, -10, 5, 5}, expected_by_window[window - 1]));
  for (int n : {1, 2, 3, 7, 31, 32, 33, 127, 128, 129, 255, 256, 257,
                511, 512, 513, 1023, 1024, 1025, 4097})
    cases.push_back({"length-" + std::to_string(n), n, n / 3 + 1, {}});
  for (int window : {1, 2, 3, 7, 31, 32, 33, 127, 128, 129, 255, 256, 257,
                     511, 512, 513, 1023, 1024, 1025})
    cases.push_back({"window-" + std::to_string(window), window + 97, window, {}});
  for (int start : {0, 15, 30}) {
    Case c{"best-start-" + std::to_string(start), 37, 7, std::vector<int>(37, -10)};
    std::fill_n(c.input.begin() + start, c.window, 10);
    c.known_first = 70;
    cases.push_back(std::move(c));
  }
  for (int boundary : {32, 256, 512}) {
    for (int window : {3, 33}) {
      Case c{"cross-" + std::to_string(boundary) + "-w" + std::to_string(window),
             boundary + window + 3, window, std::vector<int>(boundary + window + 3, -10)};
      std::fill_n(c.input.begin() + boundary - window / 2, window, 10);
      c.known_first = int64_t(window) * 10;
      cases.push_back(std::move(c));
    }
  }
  for (int seed = 0; seed < 5; ++seed) {
    Case c{"random-" + std::to_string(seed), 73 + 37 * seed, 2 + 11 * seed, {}};
    prepare_input(c);
    cases.push_back(std::move(c));
  }
  for (int offset : {1, 2, 3}) {
    Case input{"input-offset-" + std::to_string(offset), 259, 33, {}};
    input.input_offset = offset;
    cases.push_back(std::move(input));
    Case output{"output-offset-" + std::to_string(offset), 259, 33, {}};
    output.output_offset = offset;
    cases.push_back(std::move(output));
  }
  Case both{"both-offsets", 513, 257, {}};
  both.input_offset = 1;
  both.output_offset = 3;
  cases.push_back(std::move(both));
  auto change = known_case("window-change", 1, {6, -10, 5, 5}, 6);
  change.next_window = 4;
  cases.push_back(std::move(change));
  auto positive = known_case("overwrite-positive", 1, {7}, 7);
  positive.mode = OutputMode::Poisoned;
  cases.push_back(std::move(positive));
  auto negative = known_case("overwrite-negative", 2, {-6, -1, -5, -2}, -6);
  negative.mode = OutputMode::Poisoned;
  cases.push_back(std::move(negative));
  auto repeat = known_case("consecutive-calls", 1, {7}, 7);
  repeat.mode = OutputMode::Reused;
  cases.push_back(std::move(repeat));
  Case blocks{"consecutive-blocks", 1025, 257, {}, OutputMode::Reused};
  blocks.fill = 10;
  blocks.next_window = 513;
  blocks.known_first = 2570;
  cases.push_back(std::move(blocks));
  return cases;
}

static std::vector<Case> large_cases() {
  Case single{"large-single", 50000, 1, {}};
  Case interior{"large-window-257", 50000, 257, {}};
  Case almost_full{"large-negative-49999", 50000, 49999, {}};
  almost_full.fill = -10;
  almost_full.known_first = -499990;
  Case full{"large-full", 50000, 50000, {}};
  full.fill = 10;
  full.known_first = 500000;
  return {std::move(single), std::move(interior), std::move(almost_full), std::move(full)};
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
  std::printf("GPU: %s\nMax subarray sum: exactly window_size elements; "
              "exact INT64 CPU reference; 2/3 calls per case\n", prop.name);
  int failed = 0;
  for (auto& c : cases)
    if (!run_case(std::move(c))) ++failed;
  std::printf("Max subarray sum: %zu/%zu cases passed\n", cases.size() - failed, cases.size());
  return test::finish(failed == 0);
}
