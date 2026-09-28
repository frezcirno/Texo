#include "test_utils.h"
#include <cuda_fp16.h>
#include <limits>
#include <string>
#include <utility>

extern "C" void solve(const half*, const half*, half*, int, int, int, int);

enum class Pattern {
  Binary, Decimal, ZeroA, ZeroB, LeftIdentity, RightIdentity,
  BatchConstants, LastBatch, TailOnly, Negative, Separable, Explicit
};
enum class OutputMode { Cleared, Finite, NaN, Infinity, Reused };

struct Case {
  std::string name;
  int batches, m, n, k;
  Pattern pattern = Pattern::Binary;
  OutputMode mode = OutputMode::Cleared;
  int a_offset = 0, b_offset = 0, c_offset = 0;
  std::vector<half> a, b;
  std::vector<double> known;
};

static std::vector<half> halves(std::initializer_list<float> values) {
  std::vector<half> result;
  for (float value : values) result.push_back(__float2half_rn(value));
  return result;
}

static void prepare(Case& c) {
  if (c.pattern == Pattern::Explicit) return;
  c.a.resize(size_t(c.batches) * c.m * c.k);
  c.b.resize(size_t(c.batches) * c.k * c.n);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> binary(-32, 32);
  std::uniform_real_distribution<float> decimal(-1.0f, 1.0f);
  for (int batch = 0; batch < c.batches; ++batch) {
    for (int row = 0; row < c.m; ++row) {
      for (int inner = 0; inner < c.k; ++inner) {
        float value = binary(rng) / 32.0f;
        switch (c.pattern) {
          case Pattern::Decimal: value = decimal(rng); break;
          case Pattern::ZeroA: value = 0; break;
          case Pattern::LeftIdentity: value = row == inner ? 1 : 0; break;
          case Pattern::BatchConstants: value = (batch + 1) / 16.0f; break;
          case Pattern::LastBatch: value = batch == c.batches - 1 ? (row + 1) / 8.0f : 0; break;
          case Pattern::TailOnly: value = inner == c.k - 1 ? (row + batch + 1) / 8.0f : 0; break;
          case Pattern::Negative: value = -(row % 7 + 1) / 8.0f; break;
          case Pattern::Separable:
            value = ((batch + row) % 7 + 1) * (inner % 2 ? -1.0f : 1.0f) / 64.0f;
            break;
          default: break;
        }
        c.a[(size_t(batch) * c.m + row) * c.k + inner] = __float2half_rn(value);
      }
    }
    for (int inner = 0; inner < c.k; ++inner) {
      for (int col = 0; col < c.n; ++col) {
        float value = binary(rng) / 32.0f;
        switch (c.pattern) {
          case Pattern::Decimal: value = decimal(rng); break;
          case Pattern::ZeroB: value = 0; break;
          case Pattern::RightIdentity: value = inner == col ? 1 : 0; break;
          case Pattern::BatchConstants: value = -(batch + 2) / 32.0f; break;
          case Pattern::LastBatch: value = batch == c.batches - 1 ? (col + 1) / 8.0f : 0; break;
          case Pattern::Negative: value = (col % 5 + 1) / 8.0f; break;
          case Pattern::Separable:
            value = ((batch + col) % 5 + 1) * (inner % 2 ? -1.0f : 1.0f) / 64.0f;
            break;
          default: break;
        }
        c.b[(size_t(batch) * c.k + inner) * c.n + col] = __float2half_rn(value);
      }
    }
  }
}

static std::vector<double> reference(const Case& c) {
  std::vector<double> result(size_t(c.batches) * c.m * c.n);
  for (int batch = 0; batch < c.batches; ++batch) {
    for (int row = 0; row < c.m; ++row) {
      for (int col = 0; col < c.n; ++col) {
        double sum = 0;
        if (c.pattern == Pattern::Separable) {
          // A=r*s/64 and B=t*s/64, with s=+/-1: dot product = K*r*t/4096.
          sum = double(c.k) * ((batch + row) % 7 + 1) * ((batch + col) % 5 + 1) / 4096.0;
        } else {
          // Reference reads the actual FP16-quantized inputs, not their generators.
          for (int inner = 0; inner < c.k; ++inner)
            sum += double(__half2float(c.a[(size_t(batch) * c.m + row) * c.k + inner])) *
                   double(__half2float(c.b[(size_t(batch) * c.k + inner) * c.n + col]));
        }
        result[(size_t(batch) * c.m + row) * c.n + col] = __half2float(__double2half(sum));
      }
    }
  }
  return result;
}

static bool run_case(Case c) {
  prepare(c);
  const size_t count = size_t(c.batches) * c.m * c.n;
  if (c.batches < 1 || c.batches > 128 || c.m < 1 || c.m > 1024 ||
      c.n < 1 || c.n > 1024 || c.k < 1 || c.k > 1024 ||
      c.a.size() != size_t(c.batches) * c.m * c.k ||
      c.b.size() != size_t(c.batches) * c.k * c.n) {
    std::fprintf(stderr, "%s: invalid test input\n", c.name.c_str());
    return false;
  }
  auto expected = reference(c);
  if (!c.known.empty() && expected != c.known) {
    std::fprintf(stderr, "%s: CPU reference disagrees with known result\n", c.name.c_str());
    return false;
  }
  constexpr size_t GUARD = 8; // Preserve 16-byte alignment before optional half offset.
  const half marker = __float2half_rn(1234.0f);
  const size_t prefix = GUARD + c.c_offset;
  // No input suffix padding, so memcheck can detect tail overreads.
  std::vector<half> stored_a(c.a_offset + c.a.size(), marker);
  std::vector<half> stored_b(c.b_offset + c.b.size(), marker);
  std::copy(c.a.begin(), c.a.end(), stored_a.begin() + c.a_offset);
  std::copy(c.b.begin(), c.b.end(), stored_b.begin() + c.b_offset);
  test::DeviceArray<half> a(stored_a), b(stored_b), output(prefix + count + GUARD);
  std::vector<half> initial(prefix + count + GUARD, marker);
  bool passed = true;
  double max_error = 0;
  const int calls = c.mode == OutputMode::Reused ? 3 : 2;
  for (int call = 0; call < calls; ++call) {
    if (call == calls - 1) {
      // Change the product on the final call to catch stale/cached output.
      for (size_t i = c.a_offset; i < stored_a.size(); ++i)
        stored_a[i] = __float2half_rn(-__half2float(stored_a[i]));
      for (double& value : expected) value = -value;
      a.upload(stored_a);
    }
    if (c.mode != OutputMode::Reused || call == 0) {
      for (size_t i = 0; i < count; ++i) {
        float value = 0;
        if (c.mode == OutputMode::Finite) value = (i % 2 ? -1 : 1) * (call + 1) * 8.0f;
        if (c.mode == OutputMode::NaN) value = std::numeric_limits<float>::quiet_NaN();
        if (c.mode == OutputMode::Infinity) value = (i % 2 ? -1 : 1) * std::numeric_limits<float>::infinity();
        initial[prefix + i] = __float2half_rn(value);
      }
      output.upload(initial);
    }
    solve(a.data() + c.a_offset, b.data() + c.b_offset, output.data() + prefix,
          c.batches, c.m, c.n, c.k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto actual = output.download();
    bool reported = false;
    for (size_t i = 0; i < count; ++i) {
      const double got = __half2float(actual[prefix + i]);
      const double error = std::isfinite(got) ? std::abs(got - expected[i])
                                            : std::numeric_limits<double>::infinity();
      max_error = std::max(max_error, error);
      // Binary/analytic cases have exactly representable FP32 sums. Decimal
      // inputs allow FP32 reduction order differences followed by FP16 rounding.
      const double tolerance = c.pattern == Pattern::Decimal
                                   ? 2e-4 + 1e-3 * std::abs(expected[i]) : 0;
      if (!std::isfinite(got) || error > tolerance) {
        if (!reported)
          std::fprintf(stderr, "%s call=%d batch=%zu row=%zu col=%zu got=%.9g expected=%.9g\n",
                       c.name.c_str(), call + 1, i / (size_t(c.m) * c.n),
                       (i / c.n) % c.m, i % c.n, got, expected[i]);
        reported = true;
        passed = false;
      }
    }
    if (std::memcmp(actual.data(), initial.data(), prefix * sizeof(half)) != 0 ||
        std::memcmp(actual.data() + prefix + count, initial.data() + prefix + count,
                    GUARD * sizeof(half)) != 0) {
      std::fprintf(stderr, "%s call=%d output guard overwritten\n", c.name.c_str(), call + 1);
      passed = false;
    }
    passed &= a.unchanged(stored_a);
    passed &= b.unchanged(stored_b);
  }
  std::printf("%s %-28s B/M/N/K=%d/%d/%d/%d calls=%d max_abs_error=%.3g\n",
              passed ? "PASS" : "FAIL", c.name.c_str(), c.batches, c.m, c.n, c.k, calls, max_error);
  return passed;
}

static std::vector<Case> quick_cases() {
  std::vector<Case> cases;
  Case example{"example", 2, 2, 2, 3, Pattern::Explicit};
  example.a = halves({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  example.b = halves({1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1});
  example.known = {22, 28, 49, 64, 92, 68, 128, 95};
  cases.push_back(std::move(example));
  const int shapes[][4] = {
      {1, 1, 1, 1}, {1, 1, 17, 7}, {1, 17, 1, 7}, {2, 3, 5, 7},
      {7, 8, 4, 9}, {8, 7, 4, 9}, {8, 8, 3, 9}, {8, 8, 4, 8},
      {9, 8, 4, 9}, {8, 9, 4, 9}, {8, 8, 5, 9}, {9, 9, 5, 9},
      {16, 16, 8, 16}, {17, 5, 7, 33}, {3, 17, 33, 19},
      {2, 33, 65, 1}, {2, 3, 9, 257}, {2, 3, 5, 1023}, {2, 3, 5, 1024},
      {4, 65, 67, 31}, {127, 3, 5, 7}, {128, 3, 5, 7},
      {1, 1024, 3, 17}, {1, 3, 1024, 17}};
  for (const auto& s : shapes)
    cases.push_back({"shape-" + std::to_string(s[0]) + "-" + std::to_string(s[1]) +
                     "-" + std::to_string(s[2]) + "-" + std::to_string(s[3]),
                     s[0], s[1], s[2], s[3]});
  cases.push_back({"zero-a", 9, 9, 5, 17, Pattern::ZeroA});
  cases.push_back({"zero-b", 9, 9, 5, 17, Pattern::ZeroB});
  cases.push_back({"left-identity", 3, 17, 19, 17, Pattern::LeftIdentity});
  cases.push_back({"right-identity", 3, 19, 17, 17, Pattern::RightIdentity});
  cases.push_back({"batch-constants", 17, 9, 5, 33, Pattern::BatchConstants});
  cases.push_back({"last-batch", 9, 9, 5, 17, Pattern::LastBatch});
  cases.push_back({"tail-only", 9, 9, 5, 257, Pattern::TailOnly});
  cases.push_back({"negative-products", 9, 9, 5, 17, Pattern::Negative});
  for (int k : {17, 257, 1024})
    cases.push_back({"decimal-k" + std::to_string(k), 3, 9, 5, k, Pattern::Decimal});
  for (int offsets = 0; offsets < 4; ++offsets) {
    Case c{"offset-" + std::to_string(offsets), 9, 9, 5, 17};
    c.a_offset = offsets == 0 || offsets == 3;
    c.b_offset = offsets == 1 || offsets == 3;
    c.c_offset = offsets == 2 || offsets == 3;
    cases.push_back(std::move(c));
  }
  cases.push_back({"overwrite-finite", 9, 9, 5, 17, Pattern::Binary, OutputMode::Finite});
  cases.push_back({"overwrite-zero-product", 9, 9, 5, 17, Pattern::ZeroA, OutputMode::Finite});
  cases.push_back({"overwrite-nan", 2, 3, 5, 7, Pattern::Binary, OutputMode::NaN});
  cases.push_back({"overwrite-infinity", 2, 3, 5, 7, Pattern::Binary, OutputMode::Infinity});
  cases.push_back({"consecutive-calls", 1, 1, 1, 1, Pattern::BatchConstants, OutputMode::Reused});
  cases.push_back({"consecutive-blocks", 9, 9, 5, 17, Pattern::Binary, OutputMode::Reused});

  Case small{"fp32-small-increments", 1, 1, 1, 1024, Pattern::Explicit};
  small.a.assign(1024, __float2half_rn(1));
  small.b.assign(1024, __float2half_rn(1.0f / 4096));
  small.b[0] = __float2half_rn(1);
  small.known = {1.25};
  cases.push_back(std::move(small));
  Case cancel{"fp32-product-cancellation", 1, 1, 1, 3, Pattern::Explicit};
  cancel.a = halves({32768, -32768, 0.5f});
  cancel.b = halves({2, 2, 1});
  cancel.known = {0.5};
  cases.push_back(std::move(cancel));
  Case wide{"fp32-sum-cancellation", 1, 1, 1, 5, Pattern::Explicit};
  wide.a = halves({32768, 32768, -32768, -32768, 1});
  wide.b = halves({1, 1, 1, 1, 1});
  wide.known = {1};
  cases.push_back(std::move(wide));
  Case ties{"half-round-ties-even", 1, 1, 4, 2, Pattern::Explicit};
  ties.a = halves({1, 1});
  ties.b = halves({1, 1, -1, -1, 1.0f / 2048, 3.0f / 2048, -1.0f / 2048, -3.0f / 2048});
  ties.known = {1, 1.001953125, -1, -1.001953125};
  cases.push_back(std::move(ties));
  Case tiny{"half-subnormals", 1, 1, 4, 1, Pattern::Explicit};
  tiny.a = halves({0.5f});
  tiny.b = halves({1.0f / 8388608, -1.0f / 8388608, 1.0f / 16384, -1.0f / 16384});
  tiny.known = {1.0 / 16777216, -1.0 / 16777216, 1.0 / 32768, -1.0 / 32768};
  cases.push_back(std::move(tiny));
  return cases;
}

static std::vector<Case> large_cases() {
  // Analytic references avoid CPU cubic work while checking every output.
  return {{"max-dimensions", 1, 1024, 1024, 1024, Pattern::Separable},
          {"large-tails", 3, 1023, 1023, 1023, Pattern::Separable},
          {"performance-shape-b128", 128, 256, 256, 256, Pattern::Separable}};
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
  std::printf("GPU: %s\nFP16 batched MM: C=A*B; FP32 accumulation, FP16 output\n", prop.name);
  int failed = 0;
  for (auto& c : cases)
    if (!run_case(std::move(c))) ++failed;
  std::printf("FP16 batched MM: %zu/%zu cases passed\n", cases.size() - failed, cases.size());
  return test::finish(failed == 0);
}
