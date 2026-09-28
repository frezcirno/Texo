#include "test_utils.h"
#include <cstdint>
#include <string>
#include <utility>

extern "C" void solve(const int*, int*, int, int);

enum class OutputMode { Cleared, Poisoned, Reused };

struct Case {
  std::string name;
  std::vector<int> input;
  int k;
  OutputMode mode = OutputMode::Cleared;
  int input_offset = 0, output_offset = 0;
};

static bool run_case(Case c) {
  constexpr size_t GUARD = 4;
  constexpr int MARKER = 123456789;
  const size_t prefix = GUARD + c.output_offset;
  const int calls = c.mode == OutputMode::Reused ? 3 : 2;
  // Input starts at a 16-byte boundary for int4, with no padded suffix.
  test::DeviceArray<int> input(c.input.size() + c.input_offset);
  test::DeviceArray<int> output(prefix + 1 + GUARD);
  bool passed = true;
  for (int call = 0; call < calls; ++call) {
    if (call == calls - 1) {
      // Change K and complement the matching positions, within [1,100000].
      const int next_k = c.k == 100000 ? 1 : c.k + 1;
      const int other = next_k == 1 ? 2 : 1;
      for (int& value : c.input) value = value == c.k ? other : next_k;
      c.k = next_k;
    }
    const int64_t expected = std::count(c.input.begin(), c.input.end(), c.k);
    std::vector<int> stored_input(c.input.size() + c.input_offset, MARKER);
    std::copy(c.input.begin(), c.input.end(), stored_input.begin() + c.input_offset);
    input.upload(stored_input);
    // Baseline cases isolate counting; separate regressions check overwrite
    // semantics with nonzero output and consecutive calls without a reset.
    if (c.mode != OutputMode::Reused || call == 0) {
      std::vector<int> initial(prefix + 1 + GUARD, MARKER);
      initial[prefix] = c.mode == OutputMode::Poisoned ? (call == 0 ? 17 : -23) : 0;
      output.upload(initial);
    }
    solve(input.data() + c.input_offset, output.data() + prefix,
          int(c.input.size()), c.k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto actual = output.download();
    if (int64_t(actual[prefix]) != expected) {
      std::fprintf(stderr, "%s call=%d N=%zu K=%d got=%d expected=%lld\n",
                   c.name.c_str(), call + 1, c.input.size(), c.k, actual[prefix],
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
  const char* mode = c.mode == OutputMode::Cleared ? "cleared" :
                     c.mode == OutputMode::Poisoned ? "poisoned" : "reused";
  std::printf("%s %-25s N=%zu output=%s offsets=%d/%d\n",
              passed ? "PASS" : "FAIL", c.name.c_str(), c.input.size(), mode,
              c.input_offset, c.output_offset);
  return passed;
}

static Case mixed_case(const std::string& name, int n, int k = 97) {
  Case c{name, std::vector<int>(n), k};
  std::mt19937 rng(42 + unsigned(n));
  std::uniform_int_distribution<int> dist(1, 100000);
  for (int& value : c.input) {
    value = dist(rng);
    if (value % 5 == 0) value = k;
  }
  return c;
}

static std::vector<Case> quick_cases() {
  std::vector<Case> cases;
  cases.push_back({"example-1", {1, 2, 3, 4, 1}, 1}); // 2 matches
  cases.push_back({"example-2", {5, 10, 5, 2}, 11}); // 0 matches
  for (int n : {1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33,
                127, 128, 129, 255, 256, 257, 258, 259, 511, 512, 513,
                1023, 1024, 1025, 4095, 4096, 4097, 65535, 65536, 65537})
    cases.push_back(mixed_case("boundary-" + std::to_string(n), n));
  cases.push_back({"all-match-max-value", std::vector<int>(4097, 100000), 100000});
  cases.push_back({"no-matches", std::vector<int>(4097, 1), 100000});
  cases.push_back(mixed_case("target-min-value", 1031, 1));
  auto clustered = mixed_case("clustered-matches", 8197, 99);
  std::fill(clustered.input.begin(), clustered.input.end(), 1);
  std::fill(clustered.input.end() - 1029, clustered.input.end(), clustered.k);
  cases.push_back(std::move(clustered));
  for (int component = 0; component < 4; ++component) {
    Case c{"vector-component-" + std::to_string(component), std::vector<int>(1024, 1), 100000};
    for (size_t i = component; i < c.input.size(); i += 4) c.input[i] = c.k;
    cases.push_back(std::move(c));
  }
  for (int tail = 1; tail <= 3; ++tail)
    for (int pos = 0; pos < tail; ++pos) {
      Case c{"tail-" + std::to_string(tail) + "-position-" + std::to_string(pos),
             std::vector<int>(1024 + tail, 1), 7};
      c.input[1024 + pos] = c.k;
      cases.push_back(std::move(c));
    }
  Case last{"last-element-only", std::vector<int>(1048579, 1), 100000};
  last.input.back() = last.k;
  cases.push_back(std::move(last));
  for (int offset : {1, 2, 3}) {
    auto c = mixed_case("output-offset-" + std::to_string(offset), 257);
    c.output_offset = offset;
    cases.push_back(std::move(c));
  }
  auto offset = mixed_case("aligned-input-offset", 1027);
  offset.input_offset = 4; // Still 16-byte aligned for the vectorized input.
  cases.push_back(std::move(offset));
  cases.push_back({"overwrite-nonzero", {1, 2, 3, 4, 1}, 1, OutputMode::Poisoned});
  cases.push_back({"overwrite-zero-count", std::vector<int>(257, 1), 2, OutputMode::Poisoned});
  cases.push_back({"consecutive-calls", {1, 2, 3, 4, 1}, 1, OutputMode::Reused});
  auto reused = mixed_case("consecutive-blocks", 4097);
  reused.mode = OutputMode::Reused;
  cases.push_back(std::move(reused));
  return cases;
}

static std::vector<Case> large_cases() {
  std::vector<Case> cases;
  // The odd count above 2^24 detects accidental floating-point count storage.
  cases.push_back({"large-exact-integer", std::vector<int>(16777217, 100000), 100000});
  Case c{"large-100-million", std::vector<int>(100000000), 100000};
  const int pattern[] = {1, 100000, 7, 100000, 9, 100000, 22};
  for (size_t i = 0; i < c.input.size(); ++i) c.input[i] = pattern[i % 7];
  cases.push_back(std::move(c));
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
  std::printf("GPU: %s\nCount array element: exact CPU integer reference, "
              "aligned inputs, 2/3 calls per case\n", prop.name);
  int failed = 0;
  for (auto& c : cases)
    if (!run_case(std::move(c))) ++failed;
  std::printf("Count array element: %zu/%zu cases passed\n", cases.size() - failed, cases.size());
  return test::finish(failed == 0);
}
