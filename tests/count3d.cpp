#include "test_utils.h"
#include <cstdint>
#include <string>
#include <utility>

extern "C" void solve(const int*, int*, int, int, int, int);

enum class OutputMode { Cleared, Poisoned, Reused };

struct Case {
  std::string name;
  int n, m, k, p;
  std::vector<int> input;
  OutputMode mode = OutputMode::Cleared;
  bool change_target = true, almost_all = false;
  int input_offset = 0, output_offset = 0;
};

static size_t elements(const Case& c) {
  return size_t(c.n) * c.m * c.k;
}

static void fill_input(Case& c) {
  if (!c.input.empty()) return;
  c.input.resize(elements(c));
  if (c.almost_all) {
    std::fill(c.input.begin(), c.input.end(), c.p);
    c.input.back() = c.p == 1 ? 2 : 1;
  } else {
    // Deterministic scattered matches, identical for equal-sized reshapes.
    const int pattern[] = {1, 100, 3, 5, 7, 31, 99};
    for (size_t i = 0; i < c.input.size(); ++i)
      c.input[i] = i % 5 == 0 ? c.p : pattern[i % 7];
  }
}

static bool run_case(Case c) {
  constexpr size_t GUARD = 4;
  constexpr int MARKER = 123456789;
  const size_t count = elements(c), prefix = GUARD + c.output_offset;
  const int calls = c.mode == OutputMode::Reused ? 3 : 2;
  std::printf("RUN  %-24s N/M/K=%d/%d/%d P=%d elements=%zu\n",
              c.name.c_str(), c.n, c.m, c.k, c.p, count);
  fill_input(c);
  if (c.input.size() != count) {
    std::fprintf(stderr, "%s: invalid test input size\n", c.name.c_str());
    return false;
  }
  // Keep input 16-byte aligned for int4, and leave the allocation tail unpadded.
  test::DeviceArray<int> input(count + c.input_offset);
  test::DeviceArray<int> output(prefix + 1 + GUARD);
  bool passed = true;
  for (int call = 0; call < calls; ++call) {
    if (call == calls - 1) {
      const int next_p = c.change_target ? (c.p == 100 ? 1 : c.p + 1) : c.p;
      const int other = next_p == 1 ? 2 : 1;
      for (int& value : c.input) value = value == c.p ? other : next_p;
      c.p = next_p;
    }
    const int64_t expected = std::count(c.input.begin(), c.input.end(), c.p);
    // Avoid an extra full host copy for the 1000^3 regression.
    std::vector<int> shifted;
    if (c.input_offset) {
      shifted.assign(count + c.input_offset, MARKER);
      std::copy(c.input.begin(), c.input.end(), shifted.begin() + c.input_offset);
    }
    const auto& stored_input = c.input_offset ? shifted : c.input;
    input.upload(stored_input);
    if (c.mode != OutputMode::Reused || call == 0) {
      std::vector<int> initial(prefix + 1 + GUARD, MARKER);
      initial[prefix] = c.mode == OutputMode::Poisoned ? (call == 0 ? 17 : -23) : 0;
      output.upload(initial);
    }
    solve(input.data() + c.input_offset, output.data() + prefix, c.n, c.m, c.k, c.p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto actual = output.download();
    if (int64_t(actual[prefix]) != expected) {
      std::fprintf(stderr, "%s call=%d N/M/K=%d/%d/%d P=%d got=%d expected=%lld\n",
                   c.name.c_str(), call + 1, c.n, c.m, c.k, c.p, actual[prefix],
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
  std::printf("%s %-24s output=%s offsets=%d/%d\n", passed ? "PASS" : "FAIL",
              c.name.c_str(), mode, c.input_offset, c.output_offset);
  return passed;
}

static std::vector<Case> quick_cases() {
  std::vector<Case> cases;
  cases.push_back({"example-1", 2, 2, 3, 1, {1, 2, 3, 4, 5, 1, 1, 1, 1, 2, 2, 2}}); // 5
  cases.push_back({"example-2", 1, 3, 2, 1, {5, 10, 5, 2, 2, 2}}); // 0
  for (int length : {1, 2, 3, 4, 5, 7, 8, 15, 16, 17})
    cases.push_back({"short-" + std::to_string(length), 1, 1, length, 97, {}});
  for (int dim : {1, 3, 31, 32, 33, 255, 256, 257, 999, 1000}) {
    cases.push_back({"axis-n-" + std::to_string(dim), dim, 3, 5, 97, {}});
    cases.push_back({"axis-m-" + std::to_string(dim), 3, dim, 5, 97, {}});
    cases.push_back({"axis-k-" + std::to_string(dim), 3, 5, dim, 97, {}});
  }
  int dims[] = {2, 3, 5};
  do {
    cases.push_back({"reshape-" + std::to_string(dims[0]) + "-" +
                     std::to_string(dims[1]) + "-" + std::to_string(dims[2]),
                     dims[0], dims[1], dims[2], 7, {}});
  } while (std::next_permutation(dims, dims + 3));
  cases.push_back({"all-match-p100", 3, 5, 7, 100, std::vector<int>(105, 100)});
  cases.push_back({"none-match-p1", 3, 5, 7, 1, std::vector<int>(105, 100)});
  for (int component = 0; component < 4; ++component) {
    Case c{"vector-component-" + std::to_string(component), 4, 8, 32, 99,
           std::vector<int>(1024, 1)};
    for (size_t i = component; i < c.input.size(); i += 4) c.input[i] = c.p;
    cases.push_back(std::move(c));
  }
  // Factorizations of 1025/1026/1027; every dimension respects the <=1000 bound.
  const int tail_shapes[][3] = {{5, 5, 41}, {2, 9, 57}, {13, 79, 1}};
  for (int tail = 1; tail <= 3; ++tail)
    for (int pos = 0; pos < tail; ++pos) {
      const int* d = tail_shapes[tail - 1];
      Case c{"tail-" + std::to_string(tail) + "-position-" + std::to_string(pos),
             d[0], d[1], d[2], 99, std::vector<int>(1024 + tail, 1)};
      c.input[1024 + pos] = c.p;
      cases.push_back(std::move(c));
    }
  for (int axis = 0; axis < 3; ++axis) {
    Case c{"last-slice-axis-" + std::to_string(axis), 7, 9, 11, 100,
           std::vector<int>(7 * 9 * 11, 1)};
    for (int n = 0; n < c.n; ++n)
      for (int m = 0; m < c.m; ++m)
        for (int k = 0; k < c.k; ++k)
          if ((axis == 0 && n == c.n - 1) || (axis == 1 && m == c.m - 1) ||
              (axis == 2 && k == c.k - 1))
            c.input[(size_t(n) * c.m + m) * c.k + k] = c.p;
    cases.push_back(std::move(c));
  }
  Case last{"last-element-only", 101, 103, 107, 100,
             std::vector<int>(101 * 103 * 107, 1)};
  last.input.back() = last.p;
  cases.push_back(std::move(last));
  for (int offset : {1, 2, 3}) {
    Case c{"output-offset-" + std::to_string(offset), 3, 5, 17, 99, {}};
    c.output_offset = offset;
    cases.push_back(std::move(c));
  }
  Case offset{"aligned-input-offset", 13, 79, 1, 99, {}};
  offset.input_offset = 4;
  cases.push_back(std::move(offset));

  // Keep P == K throughout these cases to isolate counting/output reset from
  // accidentally passing the third dimension as the target value.
  Case control{"p-equals-k", 3, 5, 7, 7, {}};
  control.change_target = false;
  cases.push_back(std::move(control));
  for (int which = 0; which < 4; ++which) {
    const char* names[] = {"overwrite-nonzero", "overwrite-zero-count",
                           "consecutive-calls", "consecutive-blocks"};
    Case c{names[which], which == 3 ? 311 : 2, 3, 5, 5, {}};
    c.change_target = false;
    c.mode = which < 2 ? OutputMode::Poisoned : OutputMode::Reused;
    if (which == 1) c.input.assign(elements(c), 1);
    cases.push_back(std::move(c));
  }
  return cases;
}

static std::vector<Case> large_cases() {
  // Allocate one case at a time in run_case, not when listing/selecting cases.
  std::vector<Case> cases{{"large-500-cubed", 500, 500, 500, 97, {}}};
  Case limit{"large-1000-cubed", 1000, 1000, 1000, 100, {}};
  limit.almost_all = true; // 999,999,999: a large odd count, not exactly FP32.
  cases.push_back(std::move(limit));
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
  std::printf("GPU: %s\n3D count: exact CPU integer reference over N*M*K, "
              "target P, aligned inputs, 2/3 calls per case\n", prop.name);
  int failed = 0;
  for (auto& c : cases)
    if (!run_case(std::move(c))) ++failed;
  std::printf("Count 3D array element: %zu/%zu cases passed\n", cases.size() - failed, cases.size());
  return test::finish(failed == 0);
}
