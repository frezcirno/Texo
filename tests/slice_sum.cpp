#include "test_utils.h"
#include <cstdint>
#include <numeric>
#include <string>
#include <utility>

extern "C" void solve(const int*, int*, int, int, int);

enum class OutputMode { Cleared, Poisoned, Reused };

struct Case {
  std::string name;
  int n, s, e;
  std::vector<int> input;
  OutputMode mode = OutputMode::Cleared;
  int output_offset = 0;
  int next_s = -1, next_e = -1;
  int fill = 0; // 0: deterministic mixed values; otherwise a constant in [1,10].
};

static bool run_case(Case c) {
  constexpr size_t GUARD = 4;
  constexpr int MARKER = 123456789;
  const size_t prefix = GUARD + c.output_offset;
  const int calls = c.mode == OutputMode::Reused ? 3 : 2;
  std::printf("RUN  %-24s N=%d slice=[%d,%d] length=%d\n",
              c.name.c_str(), c.n, c.s, c.e, c.e - c.s + 1);
  if (c.input.empty()) {
    c.input.resize(c.n);
    for (int i = 0; i < c.n; ++i) c.input[i] = c.fill ? c.fill : 1 + (i * 7LL + i / 11) % 10;
  }
  if (c.input.size() != size_t(c.n) || c.s < 0 || c.e < c.s || c.e >= c.n) {
    std::fprintf(stderr, "%s: invalid test shape\n", c.name.c_str());
    return false;
  }
  // The allocation is aligned, but arbitrary legal S values need not be.
  // Do not pad the allocation tail: memcheck should catch reads after N.
  test::DeviceArray<int> input(c.n);
  test::DeviceArray<int> output(prefix + 1 + GUARD);
  bool passed = true;
  for (int call = 0; call < calls; ++call) {
    if (call == calls - 1) {
      for (int& value : c.input) value = 11 - value;
      if (c.next_s >= 0) {
        c.s = c.next_s;
        c.e = c.next_e;
      }
    }
    // Inclusive E, independently accumulated in INT64; expected sums fit INT32.
    const int64_t expected = std::accumulate(c.input.begin() + c.s,
                                            c.input.begin() + c.e + 1, int64_t(0));
    input.upload(c.input);
    if (c.mode != OutputMode::Reused || call == 0) {
      std::vector<int> initial(prefix + 1 + GUARD, MARKER);
      initial[prefix] = c.mode == OutputMode::Poisoned ? (call == 0 ? 17 : -23) : 0;
      output.upload(initial);
    }
    solve(input.data(), output.data() + prefix, c.n, c.s, c.e);
    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
      // A zero-grid launch for a valid singleton should fail this case without
      // hiding later cases. Fatal execution errors still stop the process.
      std::fprintf(stderr, "%s call=%d launch error: %s\n", c.name.c_str(),
                   call + 1, cudaGetErrorString(launch_error));
      passed = false;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto actual = output.download();
    if (int64_t(actual[prefix]) != expected) {
      std::fprintf(stderr, "%s call=%d N=%d S=%d E=%d got=%d expected=%lld\n",
                   c.name.c_str(), call + 1, c.n, c.s, c.e, actual[prefix],
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
    passed &= input.unchanged(c.input);
  }
  const char* mode = c.mode == OutputMode::Cleared ? "cleared" :
                     c.mode == OutputMode::Poisoned ? "poisoned" : "reused";
  std::printf("%s %-24s output=%s output_offset=%d\n", passed ? "PASS" : "FAIL",
              c.name.c_str(), mode, c.output_offset);
  return passed;
}

static std::vector<Case> quick_cases() {
  std::vector<Case> cases;
  cases.push_back({"example-1", 5, 1, 3, {1, 2, 1, 3, 4}}); // 6
  cases.push_back({"example-2", 4, 0, 3, {1, 2, 3, 4}}); // 10
  for (int length : {1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33,
                     127, 128, 129, 255, 256, 257, 258, 259, 511, 512, 513,
                     1023, 1024, 1025, 4095, 4096, 4097})
    cases.push_back({"full-length-" + std::to_string(length), length, 0, length - 1, {}});
  for (int start = 0; start < 4; ++start)
    for (int length : {1, 3, 4, 5, 16, 17})
      cases.push_back({"start-" + std::to_string(start) + "-length-" + std::to_string(length),
                       start + length + 3, start, start + length - 1, {}});
  for (int endpoint : {0, 1}) {
    Case c{endpoint == 0 ? "start-inclusive" : "end-inclusive", 37, 4, 20,
           std::vector<int>(37, 1)};
    c.input[endpoint == 0 ? c.s : c.e] = 10;
    cases.push_back(std::move(c));
  }
  Case outside{"outside-sentinels", 37, 4, 20, std::vector<int>(37, 10)};
  std::fill(outside.input.begin() + outside.s, outside.input.begin() + outside.e + 1, 1);
  cases.push_back(std::move(outside));
  cases.push_back({"last-element", 17, 16, 16, {}});
  for (int tail = 1; tail <= 3; ++tail)
    for (int pos = 0; pos < tail; ++pos) {
      const int start = 4, length = 1024 + tail, n = start + length;
      Case c{"tail-" + std::to_string(tail) + "-position-" + std::to_string(pos),
             n, start, n - 1, std::vector<int>(n, 1)};
      c.input[start + 1024 + pos] = 10;
      cases.push_back(std::move(c));
    }
  for (int offset : {1, 2, 3}) {
    Case c{"output-offset-" + std::to_string(offset), 259, 0, 258, {}};
    c.output_offset = offset;
    cases.push_back(std::move(c));
  }
  Case range{"range-change", 37, 0, 11, {}};
  range.next_s = 4;
  range.next_e = 24;
  cases.push_back(std::move(range));
  // Singletons isolate output reset from multi-element indexing mistakes.
  cases.push_back({"overwrite-nonzero", 1, 0, 0, {7}, OutputMode::Poisoned});
  cases.push_back({"overwrite-middle", 37, 4, 20, {}, OutputMode::Poisoned});
  cases.push_back({"consecutive-calls", 1, 0, 0, {7}, OutputMode::Reused});
  Case reuse{"consecutive-blocks", 8192, 4, 4100, {}, OutputMode::Reused};
  reuse.next_s = 4104;
  reuse.next_e = 8191;
  cases.push_back(std::move(reuse));
  return cases;
}

static std::vector<Case> large_cases() {
  Case full{"large-100-million-full", 100000000, 0, 99999999, {}};
  full.fill = 10; // The largest legal sum is 1,000,000,000, within INT32.
  Case interior{"large-100-million-interior", 100000000, 4, 99999998, {}};
  interior.fill = 1; // Odd length 99,999,995: not exactly representable in FP32.
  return {std::move(full), std::move(interior)};
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
  std::printf("GPU: %s\nSubarray sum: inclusive [S,E], exact INT64 CPU reference; "
              "2/3 calls per case\n", prop.name);
  int failed = 0;
  for (auto& c : cases)
    if (!run_case(std::move(c))) ++failed;
  std::printf("Subarray sum: %zu/%zu cases passed\n", cases.size() - failed, cases.size());
  return test::finish(failed == 0);
}
