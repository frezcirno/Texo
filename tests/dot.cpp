#include "test_utils.h"
#include <cuda_fp16.h>
#include <limits>
#include <string>
#include <utility>

// Compile this harness twice, linking exactly one standalone solve each time.
#if defined(DOT_FP16)
using Scalar = half;
constexpr const char* TYPE_NAME = "FP16";
static Scalar scalar(double value) { return __double2half(value); }
static double number(Scalar value) { return __half2float(value); }
constexpr double DECIMAL_ATOL = 2e-5, DECIMAL_RTOL = 1e-3;
#elif defined(DOT_FP32)
using Scalar = float;
constexpr const char* TYPE_NAME = "FP32";
static Scalar scalar(double value) { return static_cast<float>(value); }
static double number(Scalar value) { return value; }
constexpr double DECIMAL_ATOL = 1e-5, DECIMAL_RTOL = 2e-5;
#else
#error "Build with DOT_FP16 or DOT_FP32"
#endif

extern "C" void solve(const Scalar*, const Scalar*, Scalar*, int);

enum class Pattern { Binary, Decimal, ZeroA, ZeroB, Negative, Alternating,
                     Squares, Tail, Constant, Explicit };
enum class OutputMode { Cleared, Finite, NaN, Infinity, Reused };

struct Case {
  std::string name;
  int n;
  Pattern pattern = Pattern::Binary;
  OutputMode mode = OutputMode::Cleared;
  int a_offset = 0, b_offset = 0, output_offset = 0;
  double fill_a = 0, fill_b = 1;
  std::vector<double> a, b;
  double known = std::numeric_limits<double>::quiet_NaN();
  bool approximate = false;
};

static void prepare(const Case& c, std::vector<Scalar>& a, std::vector<Scalar>& b) {
  if (c.pattern == Pattern::Constant || c.pattern == Pattern::Tail) {
    std::fill(a.begin() + c.a_offset, a.end(), scalar(c.fill_a));
    std::fill(b.begin() + c.b_offset, b.end(), scalar(c.fill_b));
    if (c.pattern == Pattern::Tail) a.back() = scalar(123);
    return;
  }
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> binary(-8, 8);
  std::uniform_real_distribution<float> decimal(-1.0f, 1.0f);
  for (int i = 0; i < c.n; ++i) {
    double av = binary(rng) / 16.0, bv = binary(rng) / 16.0;
    switch (c.pattern) {
      case Pattern::Decimal: av = decimal(rng); bv = decimal(rng); break;
      case Pattern::ZeroA: av = 0; break;
      case Pattern::ZeroB: bv = 0; break;
      case Pattern::Negative: av = -0.5; bv = 0.5; break;
      case Pattern::Alternating: av = i % 2 ? -1 : 1; bv = 1; break;
      case Pattern::Squares: bv = av; break;
      case Pattern::Explicit: av = c.a[i]; bv = c.b[i]; break;
      default: break;
    }
    a[c.a_offset + i] = scalar(av);
    b[c.b_offset + i] = scalar(bv);
  }
}

static double full_reference(const Case& c, const std::vector<Scalar>& a,
                             const std::vector<Scalar>& b) {
  // Read the actual stored inputs, including FP16 quantization.
  double sum = 0;
  for (int i = 0; i < c.n; ++i)
    sum += number(a[c.a_offset + i]) * number(b[c.b_offset + i]);
  return sum;
}

static double reference(const Case& c, const std::vector<Scalar>& a,
                        const std::vector<Scalar>& b) {
  if (c.pattern == Pattern::Constant)
    return double(c.n) * number(a[c.a_offset]) * number(b[c.b_offset]);
  if (c.pattern == Pattern::Tail)
    return double(c.n - 1) * number(a[c.a_offset]) * number(b[c.b_offset]) +
           number(a.back()) * number(b.back());
  return full_reference(c, a, b);
}

static bool run_case(const Case& c) {
  if (c.n < 1 || c.n > 100000000 ||
      (c.pattern == Pattern::Explicit && (c.a.size() != size_t(c.n) || c.b.size() != size_t(c.n)))) {
    std::fprintf(stderr, "%s: invalid test input\n", c.name.c_str());
    return false;
  }
  constexpr size_t GUARD = 16 / sizeof(Scalar);
  const Scalar marker = scalar(1234);
  const size_t prefix = GUARD + c.output_offset;
  // Input offsets preserve 16-byte alignment; tails have no padding for memcheck.
  std::vector<Scalar> a(c.a_offset + c.n, marker), b(c.b_offset + c.n, marker);
  prepare(c, a, b);
  double ref = reference(c, a, b);
  if ((c.pattern == Pattern::Constant || c.pattern == Pattern::Tail) && c.n <= 65537 &&
      std::abs(ref - full_reference(c, a, b)) > 1e-12) {
    std::fprintf(stderr, "%s: analytic reference disagrees with full CPU dot\n", c.name.c_str());
    return false;
  }
  if (std::isfinite(c.known) && number(scalar(ref)) != c.known) {
    std::fprintf(stderr, "%s: reference disagrees with known rounded result\n", c.name.c_str());
    return false;
  }
  test::DeviceArray<Scalar> da(a), db(b), output(prefix + 1 + GUARD);
  std::vector<Scalar> initial(prefix + 1 + GUARD, marker);
  bool passed = true;
  double max_error = 0;
  const int calls = c.mode == OutputMode::Reused ? 3 : 2;
  const bool decimal = c.pattern == Pattern::Decimal || c.approximate;
  for (int call = 0; call < calls; ++call) {
    if (call == calls - 1) {
      for (size_t i = c.a_offset; i < a.size(); ++i) a[i] = scalar(-number(a[i]));
      da.upload(a);
      ref = -ref;
    }
    if (c.mode != OutputMode::Reused || call == 0) {
      double value = 0;
      if (c.mode == OutputMode::Finite) value = call == 0 ? 8 : -8;
      if (c.mode == OutputMode::NaN) value = std::numeric_limits<double>::quiet_NaN();
      if (c.mode == OutputMode::Infinity) value = std::numeric_limits<double>::infinity();
      initial[prefix] = scalar(value);
      output.upload(initial);
    }
    solve(da.data() + c.a_offset, db.data() + c.b_offset, output.data() + prefix, c.n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto actual = output.download();
    const double got = number(actual[prefix]), expected = number(scalar(ref));
    const double error = std::isfinite(got) ? std::abs(got - expected)
                                          : std::numeric_limits<double>::infinity();
    const double tolerance = decimal ? DECIMAL_ATOL + DECIMAL_RTOL * std::abs(expected) : 0;
    max_error = std::max(max_error, error);
    if (!std::isfinite(expected) || error > tolerance) {
      std::fprintf(stderr, "%s call=%d got=%.12g expected=%.12g double_ref=%.12g tolerance=%.6g\n",
                   c.name.c_str(), call + 1, got, expected, ref, tolerance);
      passed = false;
    }
    if (std::memcmp(actual.data(), initial.data(), prefix * sizeof(Scalar)) != 0 ||
        std::memcmp(actual.data() + prefix + 1, initial.data() + prefix + 1,
                    GUARD * sizeof(Scalar)) != 0) {
      std::fprintf(stderr, "%s call=%d output guard overwritten\n", c.name.c_str(), call + 1);
      passed = false;
    }
    passed &= da.unchanged(a);
    passed &= db.unchanged(b);
  }
  std::printf("%s %-28s N=%d calls=%d max_abs_error=%.6g\n",
              passed ? "PASS" : "FAIL", c.name.c_str(), c.n, calls, max_error);
  return passed;
}

static Case explicit_case(const std::string& name, std::vector<double> a,
                          std::vector<double> b, double known) {
  Case c{name, int(a.size()), Pattern::Explicit};
  c.a = std::move(a); c.b = std::move(b); c.known = known;
  return c;
}

static Case constant_case(const std::string& name, int n, double a, double b,
                          bool approximate = false) {
  Case c{name, n, Pattern::Constant};
  c.fill_a = a; c.fill_b = b;
  c.approximate = approximate;
  return c;
}

static std::vector<Case> quick_cases() {
  std::vector<Case> cases;
  cases.push_back(explicit_case("example-1", {1,2,3,4}, {5,6,7,8}, 70));
  cases.push_back(explicit_case("example-2", {0.5,1.5,2.5}, {2,3,4}, 15.5));
  for (int n : {1,2,3,4,5,6,7,8,9,31,32,33,63,64,65,127,128,129,
                255,256,257,511,512,513,1023,1024,1025,4095,4096,4097,65535,65536,65537})
    cases.push_back({"length-" + std::to_string(n), n});
  cases.push_back({"zero-a", 513, Pattern::ZeroA});
  cases.push_back({"zero-b", 513, Pattern::ZeroB});
  cases.push_back({"negative", 257, Pattern::Negative});
  cases.push_back({"alternating-even", 1024, Pattern::Alternating});
  cases.push_back({"alternating-odd", 1025, Pattern::Alternating});
  cases.push_back({"sum-of-squares", 1025, Pattern::Squares});
  for (int n : {5,6,7,257,513,1025,2051})
    cases.push_back({"last-element-" + std::to_string(n), n, Pattern::Tail});
  for (int n : {7,257,4097})
    cases.push_back({"decimal-" + std::to_string(n), n, Pattern::Decimal});
  cases.push_back(constant_case("binary-constant", 2051, 1.0/1024, 1.0/1024));
  cases.push_back(constant_case("decimal-constant", 2051, 1.0/1024, 0.1, true));
  for (int offset = 0; offset < 4; ++offset) {
    Case c{"offset-" + std::to_string(offset), 1027};
    c.a_offset = offset == 0 || offset == 3 ? 16 / sizeof(Scalar) : 0;
    c.b_offset = offset == 1 || offset == 3 ? 16 / sizeof(Scalar) : 0;
    c.output_offset = offset == 2 || offset == 3 ? 1 : 0;
    cases.push_back(std::move(c));
  }
  cases.push_back({"overwrite-finite", 257, Pattern::Binary, OutputMode::Finite});
  cases.push_back({"overwrite-zero", 513, Pattern::ZeroA, OutputMode::Finite});
  cases.push_back({"overwrite-nan", 257, Pattern::Binary, OutputMode::NaN});
  cases.push_back({"overwrite-infinity", 257, Pattern::Binary, OutputMode::Infinity});
  cases.push_back({"consecutive-scalar", 1, Pattern::Negative, OutputMode::Reused});
  cases.push_back({"consecutive-blocks", 1025, Pattern::Negative, OutputMode::Reused});
  std::vector<double> small_a(1024,1), small_b(1024,1.0/4096);
  small_b[0] = 1;
  cases.push_back(explicit_case("fp32-small-increments", std::move(small_a), std::move(small_b),
                                number(scalar(1 + 1023.0/4096))));
  cases.push_back(explicit_case("fp32-product-cancellation", {32768,-32768,0.5}, {2,2,1}, 0.5));
  std::vector<double> cancel(2050,0), ones(2050,1);
  std::fill_n(cancel.begin(),1024,128);
  std::fill_n(cancel.begin()+1024,1024,-128);
  cancel[2048] = 1;
  cases.push_back(explicit_case("fp32-block-cancellation", std::move(cancel), std::move(ones), 1));
#if defined(DOT_FP16)
  cases.push_back(explicit_case("half-tie-down", {1,1}, {1,1.0/2048}, 1));
  cases.push_back(explicit_case("half-tie-up", {1,1}, {1,3.0/2048}, 1.001953125));
  cases.push_back(explicit_case("half-smallest-subnormal", {0.5}, {1.0/8388608}, 1.0/16777216));
#endif
  return cases;
}

static std::vector<Case> large_cases() {
  return {constant_case("maximum-binary", 100000000, 1.0/1024, 1.0/1024),
          {"maximum-odd-tail", 99999999, Pattern::Tail},
          constant_case("maximum-decimal", 100000000, 1.0/1024, 0.1, true)};
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
  std::printf("GPU: %s\n%s dot: overwrite result; CPU FP64 reference rounded to output type\n", prop.name, TYPE_NAME);
  int failed = 0;
  for (const auto& c : cases) if (!run_case(c)) ++failed;
  std::printf("%s dot: %zu/%zu cases passed\n", TYPE_NAME, cases.size() - failed, cases.size());
  return test::finish(failed == 0);
}
