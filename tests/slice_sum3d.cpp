#include "test_utils.h"
#include <cstdint>
#include <string>
#include <utility>

extern "C" void solve(const int*, int*, int, int, int, int, int, int, int, int, int);

enum class OutputMode { Cleared, Poisoned, Reused };

struct Region {
  int sd, ed, sr, er, sc, ec;
};

struct Case {
  std::string name;
  int n, m, k;
  Region region;
  std::vector<int> input;
  OutputMode mode = OutputMode::Cleared;
  Region next{-1, -1, -1, -1, -1, -1};
  int output_offset = 0, fill = 0;
};

static bool valid_region(const Case& c) {
  const auto& r = c.region;
  return r.sd >= 0 && r.ed >= r.sd && r.ed < c.n &&
         r.sr >= 0 && r.er >= r.sr && r.er < c.m &&
         r.sc >= 0 && r.ec >= r.sc && r.ec < c.k;
}

static size_t index(const Case& c, int dep, int row, int col) {
  return (size_t(dep) * c.m + row) * c.k + col;
}

static int64_t reference(const Case& c) {
  // Independent nested loops: retain original M/K strides and all endpoints,
  // without mirroring the kernel's flattened-index decoding.
  int64_t sum = 0;
  for (int dep = c.region.sd; dep <= c.region.ed; ++dep)
    for (int row = c.region.sr; row <= c.region.er; ++row)
      for (int col = c.region.sc; col <= c.region.ec; ++col)
        sum += c.input[index(c, dep, row, col)];
  return sum;
}

static bool run_case(Case c) {
  constexpr size_t GUARD = 4;
  constexpr int MARKER = 123456789;
  const size_t count = size_t(c.n) * c.m * c.k, prefix = GUARD + c.output_offset;
  const int calls = c.mode == OutputMode::Reused ? 3 : 2;
  const auto& initial_region = c.region;
  std::printf("RUN  %-25s N/M/K=%d/%d/%d dep=[%d,%d] rows=[%d,%d] cols=[%d,%d]\n",
              c.name.c_str(), c.n, c.m, c.k, initial_region.sd, initial_region.ed,
              initial_region.sr, initial_region.er, initial_region.sc, initial_region.ec);
  if (c.input.empty()) {
    c.input.resize(count);
    for (int dep = 0; dep < c.n; ++dep)
      for (int row = 0; row < c.m; ++row)
        for (int col = 0; col < c.k; ++col)
          c.input[index(c, dep, row, col)] =
              c.fill ? c.fill : 1 + (dep * 3 + row * 5 + col * 7 + row / 3 + col / 11) % 10;
  }
  if (c.input.size() != count || !valid_region(c)) {
    std::fprintf(stderr, "%s: invalid test shape\n", c.name.c_str());
    return false;
  }
  // No suffix padding hides overreads from the last depth/row/column.
  test::DeviceArray<int> input(count);
  test::DeviceArray<int> output(prefix + 1 + GUARD);
  bool passed = true;
  for (int call = 0; call < calls; ++call) {
    if (call == calls - 1) {
      for (int& value : c.input) value = 11 - value;
      if (c.next.sd >= 0) c.region = c.next;
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
    solve(input.data(), output.data() + prefix, c.n, c.m, c.k,
          r.sd, r.ed, r.sr, r.er, r.sc, r.ec);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto actual = output.download();
    if (int64_t(actual[prefix]) != expected) {
      std::fprintf(stderr,
                   "%s call=%d dep=[%d,%d] rows=[%d,%d] cols=[%d,%d] got=%d expected=%lld\n",
                   c.name.c_str(), call + 1, r.sd, r.ed, r.sr, r.er, r.sc, r.ec,
                   actual[prefix], static_cast<long long>(expected));
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
  cases.push_back({"example-1", 2, 2, 3, {0, 1, 0, 0, 1, 2},
                   {1, 2, 3, 4, 5, 1, 1, 1, 1, 2, 2, 2}}); // 7
  cases.push_back({"example-2", 1, 3, 2, {0, 0, 0, 2, 1, 1}, {5, 10, 5, 2, 2, 2}}); // 14
  if (reference(cases[0]) != 7 || reference(cases[1]) != 14) {
    std::fprintf(stderr, "CPU reference disagrees with the published examples\n");
    std::exit(EXIT_FAILURE);
  }
  cases.push_back({"scalar", 1, 1, 1, {0, 0, 0, 0, 0, 0}, {7}});
  for (int length : {1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 255, 256, 257}) {
    cases.push_back({"depth-" + std::to_string(length), length + 3, 7, 11,
                     {1, length, 2, 4, 3, 7}, {}});
    cases.push_back({"height-" + std::to_string(length), 5, length + 3, 11,
                     {1, 3, 1, length, 3, 7}, {}});
    cases.push_back({"width-" + std::to_string(length), 5, 7, length + 3,
                     {1, 3, 2, 4, 1, length}, {}});
  }
  const int shapes[][3] = {{3, 5, 17}, {2, 8, 16}, {1, 1, 257}, {1, 7, 73},
                           {4, 8, 16}, {3, 9, 19}, {1, 31, 33}, {4, 16, 16},
                           {1, 25, 41}, {3, 21, 65}, {4, 32, 32}, {1, 17, 241}};
  for (const auto& shape : shapes) {
    const int d = shape[0], h = shape[1], w = shape[2];
    cases.push_back({"volume-" + std::to_string(d * h * w), d + 3, h + 5, w + 7,
                     {1, d, 2, h + 1, 3, w + 2}, {}});
  }
  cases.push_back({"depth-line", 37, 1, 1, {0, 36, 0, 0, 0, 0}, {}});
  cases.push_back({"row-line", 1, 37, 1, {0, 0, 0, 36, 0, 0}, {}});
  cases.push_back({"column-line", 1, 1, 37, {0, 0, 0, 0, 0, 36}, {}});
  Case planes{"distinct-depths", 3, 4, 5, {0, 1, 1, 2, 1, 3}, std::vector<int>(60)};
  const int values[] = {1, 4, 9};
  for (int dep = 0; dep < 3; ++dep)
    std::fill(planes.input.begin() + dep * 20, planes.input.begin() + (dep + 1) * 20, values[dep]);
  cases.push_back(std::move(planes));
  cases.push_back({"original-strides", 4, 7, 11, {1, 2, 2, 4, 3, 7}, {}});
  Case outside{"outside-sentinels", 5, 7, 11, {1, 3, 2, 4, 3, 7}, std::vector<int>(385, 10)};
  for (int dep = 1; dep <= 3; ++dep)
    for (int row = 2; row <= 4; ++row)
      for (int col = 3; col <= 7; ++col) outside.input[index(outside, dep, row, col)] = 1;
  cases.push_back(std::move(outside));
  for (int which = 0; which < 8; ++which) {
    const int dep = which & 4 ? 4 : 0, row = which & 2 ? 6 : 0, col = which & 1 ? 10 : 0;
    cases.push_back({"tensor-corner-" + std::to_string(which), 5, 7, 11,
                     {dep, dep, row, row, col, col}, {}});
    Case c{"inclusive-corner-" + std::to_string(which), 5, 7, 11,
           {1, 3, 1, 5, 2, 8}, std::vector<int>(385, 1)};
    c.input[index(c, which & 4 ? 3 : 1, which & 2 ? 5 : 1, which & 1 ? 8 : 2)] = 10;
    cases.push_back(std::move(c));
  }
  cases.push_back({"last-depth", 7, 9, 11, {6, 6, 0, 8, 0, 10}, {}});
  cases.push_back({"last-row", 7, 9, 11, {0, 6, 8, 8, 0, 10}, {}});
  cases.push_back({"last-column", 7, 9, 11, {0, 6, 0, 8, 10, 10}, {}});
  for (int offset : {1, 2, 3}) {
    Case c{"output-offset-" + std::to_string(offset), 5, 7, 11, {1, 3, 2, 4, 3, 7}, {}};
    c.output_offset = offset;
    cases.push_back(std::move(c));
  }
  Case region{"region-change", 17, 19, 23, {1, 2, 3, 5, 4, 9}, {}};
  region.next = {3, 15, 1, 17, 2, 21};
  cases.push_back(std::move(region));
  cases.push_back({"overwrite-singleton", 1, 1, 1, {0, 0, 0, 0, 0, 0}, {7}, OutputMode::Poisoned});
  cases.push_back({"overwrite-interior", 5, 7, 11, {1, 3, 2, 4, 3, 7}, {}, OutputMode::Poisoned});
  cases.push_back({"consecutive-calls", 1, 1, 1, {0, 0, 0, 0, 0, 0}, {7}, OutputMode::Reused});
  Case reuse{"consecutive-blocks", 17, 19, 23, {1, 15, 2, 17, 3, 21}, {}, OutputMode::Reused};
  reuse.next = {2, 15, 3, 18, 1, 20};
  cases.push_back(std::move(reuse));
  return cases;
}

static std::vector<Case> large_cases() {
  // At most 500^3 * 10 = 1,250,000,000, so every legal sum fits INT32.
  Case full{"large-500-cubed-full", 500, 500, 500, {0, 499, 0, 499, 0, 499}, {}};
  full.fill = 10;
  Case interior{"large-500-cubed-interior", 500, 500, 500, {1, 499, 2, 498, 3, 497}, {}};
  interior.fill = 1; // 499*497*495 = 122,761,485, an odd sum not exactly FP32.
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
  std::printf("GPU: %s\n3D subarray sum: six inclusive endpoints, exact INT64 CPU reference; "
              "2/3 calls per case\n", prop.name);
  int failed = 0;
  for (auto& c : cases)
    if (!run_case(std::move(c))) ++failed;
  std::printf("3D subarray sum: %zu/%zu cases passed\n", cases.size() - failed, cases.size());
  return test::finish(failed == 0);
}
