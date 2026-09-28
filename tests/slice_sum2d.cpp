#include "test_utils.h"
#include <cstdint>
#include <string>
#include <utility>

extern "C" void solve(const int*, int*, int, int, int, int, int, int);

enum class OutputMode { Cleared, Poisoned, Reused };

struct Region {
  int sr, er, sc, ec;
};

struct Case {
  std::string name;
  int n, m;
  Region region;
  std::vector<int> input;
  OutputMode mode = OutputMode::Cleared;
  Region next{-1, -1, -1, -1};
  int output_offset = 0, fill = 0;
};

static bool valid_region(const Case& c) {
  const auto& r = c.region;
  return r.sr >= 0 && r.er >= r.sr && r.er < c.n &&
         r.sc >= 0 && r.ec >= r.sc && r.ec < c.m;
}

static int64_t reference(const Case& c) {
  // Independent nested row/column loops, with both endpoints inclusive.
  int64_t sum = 0;
  for (int row = c.region.sr; row <= c.region.er; ++row)
    for (int col = c.region.sc; col <= c.region.ec; ++col)
      sum += c.input[size_t(row) * c.m + col];
  return sum;
}

static bool run_case(Case c) {
  constexpr size_t GUARD = 4;
  constexpr int MARKER = 123456789;
  const size_t count = size_t(c.n) * c.m, prefix = GUARD + c.output_offset;
  const int calls = c.mode == OutputMode::Reused ? 3 : 2;
  std::printf("RUN  %-25s N/M=%d/%d rows=[%d,%d] cols=[%d,%d]\n",
              c.name.c_str(), c.n, c.m, c.region.sr, c.region.er, c.region.sc, c.region.ec);
  if (c.input.empty()) {
    c.input.resize(count);
    for (int row = 0; row < c.n; ++row)
      for (int col = 0; col < c.m; ++col)
        c.input[size_t(row) * c.m + col] =
            c.fill ? c.fill : 1 + (row * 3 + col * 7 + row / 5 + col / 11) % 10;
  }
  if (c.input.size() != count || !valid_region(c)) {
    std::fprintf(stderr, "%s: invalid test shape\n", c.name.c_str());
    return false;
  }
  // No allocation suffix padding: final-row/column overreads remain detectable.
  test::DeviceArray<int> input(count);
  test::DeviceArray<int> output(prefix + 1 + GUARD);
  bool passed = true;
  for (int call = 0; call < calls; ++call) {
    if (call == calls - 1) {
      for (int& value : c.input) value = 11 - value;
      if (c.next.sr >= 0) c.region = c.next;
    }
    if (!valid_region(c)) {
      std::fprintf(stderr, "%s: invalid changed region\n", c.name.c_str());
      return false;
    }
    const int64_t expected = reference(c);
    input.upload(c.input);
    if (c.mode != OutputMode::Reused || call == 0) {
      std::vector<int> initial(prefix + 1 + GUARD, MARKER);
      initial[prefix] = c.mode == OutputMode::Poisoned ? (call == 0 ? 17 : -23) : 0;
      output.upload(initial);
    }
    const auto& r = c.region;
    solve(input.data(), output.data() + prefix, c.n, c.m, r.sr, r.er, r.sc, r.ec);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto actual = output.download();
    if (int64_t(actual[prefix]) != expected) {
      std::fprintf(stderr,
                   "%s call=%d rows=[%d,%d] cols=[%d,%d] got=%d expected=%lld\n",
                   c.name.c_str(), call + 1, r.sr, r.er, r.sc, r.ec, actual[prefix],
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
  std::printf("%s %-25s output=%s output_offset=%d\n", passed ? "PASS" : "FAIL",
              c.name.c_str(), mode, c.output_offset);
  return passed;
}

static std::vector<Case> quick_cases() {
  std::vector<Case> cases;
  cases.push_back({"example-1", 2, 3, {0, 1, 1, 2}, {1, 2, 3, 4, 5, 1}}); // 11
  cases.push_back({"example-2", 2, 2, {0, 0, 1, 1}, {5, 10, 5, 2}}); // 10
  if (reference(cases[0]) != 11 || reference(cases[1]) != 10) {
    std::fprintf(stderr, "CPU reference disagrees with the published examples\n");
    std::exit(EXIT_FAILURE);
  }
  for (int length : {1, 2, 3, 4, 7, 15, 16, 17, 31, 32, 33,
                     127, 128, 129, 255, 256, 257}) {
    cases.push_back({"height-" + std::to_string(length), length + 4, 11,
                     {1, length, 2, 8}, {}});
    cases.push_back({"width-" + std::to_string(length), 9, length + 5,
                     {2, 6, 1, length}, {}});
  }
  const int shapes[][2] = {{15, 17}, {16, 16}, {1, 257}, {7, 73}, {16, 32}, {19, 27},
                           {31, 33}, {32, 32}, {25, 41}, {63, 65}, {64, 64}, {17, 241}};
  for (const auto& shape : shapes) {
    const int h = shape[0], w = shape[1];
    cases.push_back({"area-" + std::to_string(h * w), h + 3, w + 7,
                     {2, h + 1, 3, w + 2}, {}});
  }
  cases.push_back({"single-row-matrix", 1, 37, {0, 0, 0, 36}, {}});
  cases.push_back({"single-column-matrix", 37, 1, {0, 36, 0, 0}, {}});
  cases.push_back({"scalar", 1, 1, {0, 0, 0, 0}, {7}});
  for (int row : {0, 6})
    for (int col : {0, 10})
      cases.push_back({"corner-" + std::to_string(row) + "-" + std::to_string(col),
                       7, 11, {row, row, col, col}, {}});
  Case outside{"outside-sentinels", 7, 11, {2, 4, 3, 7}, std::vector<int>(77, 10)};
  for (int row = 2; row <= 4; ++row)
    for (int col = 3; col <= 7; ++col) outside.input[row * 11 + col] = 1;
  cases.push_back(std::move(outside));
  Case stride{"original-row-stride", 4, 7, {1, 3, 2, 3}, std::vector<int>(28)};
  for (int row = 0; row < 4; ++row)
    for (int col = 0; col < 7; ++col) stride.input[row * 7 + col] = 1 + 3 * row;
  cases.push_back(std::move(stride));
  for (int which = 0; which < 4; ++which) {
    Case c{"inclusive-corner-" + std::to_string(which), 7, 11, {1, 5, 2, 8},
           std::vector<int>(77, 1)};
    c.input[(which / 2 ? 5 : 1) * 11 + (which % 2 ? 8 : 2)] = 10;
    cases.push_back(std::move(c));
  }
  cases.push_back({"last-row", 17, 19, {16, 16, 0, 18}, {}});
  cases.push_back({"last-column", 17, 19, {0, 16, 18, 18}, {}});
  for (int offset : {1, 2, 3}) {
    Case c{"output-offset-" + std::to_string(offset), 17, 19, {1, 15, 2, 18}, {}};
    c.output_offset = offset;
    cases.push_back(std::move(c));
  }
  Case range{"region-change", 37, 41, {1, 3, 2, 9}, {}};
  range.next = {5, 35, 7, 39};
  cases.push_back(std::move(range));
  cases.push_back({"overwrite-singleton", 1, 1, {0, 0, 0, 0}, {7}, OutputMode::Poisoned});
  cases.push_back({"overwrite-interior", 37, 41, {1, 35, 3, 39}, {}, OutputMode::Poisoned});
  cases.push_back({"consecutive-calls", 1, 1, {0, 0, 0, 0}, {7}, OutputMode::Reused});
  Case reuse{"consecutive-blocks", 67, 71, {2, 65, 3, 68}, {}, OutputMode::Reused};
  reuse.next = {1, 63, 5, 69};
  cases.push_back(std::move(reuse));
  return cases;
}

static std::vector<Case> large_cases() {
  // Allocate arrays only when executing a case, not when listing/selecting it.
  Case full{"large-10000-square-full", 10000, 10000, {0, 9999, 0, 9999}, {}};
  full.fill = 10; // Maximum legal sum: 1,000,000,000, within INT32.
  Case interior{"large-10000-square-interior", 10000, 10000, {1, 9999, 3, 9997}, {}};
  interior.fill = 1; // Odd area 99,940,005: not exactly representable in FP32.
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
  std::printf("GPU: %s\n2D subarray sum: inclusive row/column endpoints, "
              "exact INT64 CPU reference; 2/3 calls per case\n", prop.name);
  int failed = 0;
  for (auto& c : cases)
    if (!run_case(std::move(c))) ++failed;
  std::printf("2D subarray sum: %zu/%zu cases passed\n", cases.size() - failed, cases.size());
  return test::finish(failed == 0);
}
