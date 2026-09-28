#include "test_utils.h"
#include <limits>
#include <string>
#include <utility>

extern "C" void solve(const float*, const float*, const float*, float*, int, int,
                      int, float);

enum class Pattern {
  Random, ZeroQ, ZeroK, ZeroV, ConstantV, IdenticalQ, TailFeature, TailKey,
  IdentityV, Separable, Explicit
};

struct Case {
  std::string name;
  int m, n, d;
  float alpha;
  Pattern pattern = Pattern::Random;
  bool reuse_output = false;
  int q_offset = 0, k_offset = 0, v_offset = 0, output_offset = 0;
  std::vector<float> q, k, v;
  std::vector<double> known;
};

static void prepare(Case& c) {
  if (c.pattern == Pattern::Explicit) return;
  c.q.resize(size_t(c.m) * c.d);
  c.k.resize(size_t(c.n) * c.d);
  c.v.resize(size_t(c.n) * c.d);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(-16, 16);
  for (auto* input : {&c.q, &c.k, &c.v})
    for (float& value : *input) value = dist(rng) / 32.0f;

  if (c.pattern == Pattern::ZeroQ) std::fill(c.q.begin(), c.q.end(), 0);
  if (c.pattern == Pattern::ZeroK) std::fill(c.k.begin(), c.k.end(), 0);
  if (c.pattern == Pattern::ZeroV) std::fill(c.v.begin(), c.v.end(), 0);
  if (c.pattern == Pattern::IdenticalQ)
    for (int i = 1; i < c.m; ++i)
      std::copy_n(c.q.begin(), c.d, c.q.begin() + size_t(i) * c.d);
  if (c.pattern == Pattern::ConstantV || c.pattern == Pattern::IdentityV ||
      c.pattern == Pattern::TailKey) {
    for (int j = 0; j < c.n; ++j)
      for (int t = 0; t < c.d; ++t) {
        float value = 0;
        if (c.pattern == Pattern::ConstantV) value = (t % 9 - 4) / 4.0f;
        if (c.pattern == Pattern::IdentityV) value = j == t ? 1 : 0;
        if (c.pattern == Pattern::TailKey && j == c.n - 1) value = (t % 7 + 1) / 4.0f;
        c.v[size_t(j) * c.d + t] = value;
      }
  }
  if (c.pattern == Pattern::TailFeature) {
    std::fill(c.q.begin(), c.q.end(), 0);
    std::fill(c.k.begin(), c.k.end(), 0);
    for (int i = 0; i < c.m; ++i) c.q[size_t(i + 1) * c.d - 1] = (i % 7 - 3) / 2.0f;
    for (int j = 0; j < c.n; ++j) c.k[size_t(j + 1) * c.d - 1] = (j % 11 - 5) / 2.0f;
  }
  if (c.pattern == Pattern::Separable) {
    for (int i = 0; i < c.m; ++i)
      for (int t = 0; t < c.d; ++t)
        c.q[size_t(i) * c.d + t] = (i % 7 - 3) * (t % 2 ? -1.0f : 1.0f) / 32.0f;
    for (int j = 0; j < c.n; ++j)
      for (int t = 0; t < c.d; ++t) {
        c.k[size_t(j) * c.d + t] = (j % 11 - 5) * (t % 2 ? -1.0f : 1.0f) / 32.0f;
        c.v[size_t(j) * c.d + t] = (j % 13 - 6) * (t % 9 + 1) / 32.0f;
      }
  }
}

static std::vector<double> reference(const Case& c, bool analytic = true) {
  std::vector<double> output(size_t(c.m) * c.d, 0), scores(c.n);
  const bool separable = analytic && c.pattern == Pattern::Separable;
  for (int i = 0; i < c.m; ++i) {
    double maximum = -std::numeric_limits<double>::infinity();
    for (int j = 0; j < c.n; ++j) {
      double dot = 0;
      if (separable) {
        // Alternating signs match in Q and K, so all d products are equal.
        dot = double(c.d) * c.q[size_t(i) * c.d] * c.k[size_t(j) * c.d];
      } else {
        for (int t = 0; t < c.d; ++t)
          dot += double(c.q[size_t(i) * c.d + t]) * c.k[size_t(j) * c.d + t];
      }
      // Delta=i-j; no absolute distance, causal mask or scaling of the bias.
      scores[j] = dot / std::sqrt(double(c.d)) + double(c.alpha) * (i - j);
      maximum = std::max(maximum, scores[j]);
    }
    double total = 0;
    for (double& score : scores) {
      score = std::exp(score - maximum);
      total += score;
    }
    if (separable) {
      double value = 0;
      for (int j = 0; j < c.n; ++j) value += scores[j] / total * c.v[size_t(j) * c.d];
      // V[j,t] = V[j,0] * (t%9+1).
      for (int t = 0; t < c.d; ++t) output[size_t(i) * c.d + t] = value * (t % 9 + 1);
    } else {
      for (int j = 0; j < c.n; ++j) {
        const double weight = scores[j] / total;
        for (int t = 0; t < c.d; ++t)
          output[size_t(i) * c.d + t] += weight * c.v[size_t(j) * c.d + t];
      }
    }
  }
  return output;
}

static bool run_case(Case c) {
  prepare(c);
  if (c.m < 1 || c.m > 2048 || c.n < 1 || c.n > 2048 || c.d < 1 || c.d > 1024 ||
      !std::isfinite(c.alpha) || c.alpha < -1 || c.alpha > 1 ||
      c.q.size() != size_t(c.m) * c.d || c.k.size() != size_t(c.n) * c.d ||
      c.v.size() != size_t(c.n) * c.d) {
    std::fprintf(stderr, "%s: invalid test input\n", c.name.c_str());
    return false;
  }
  constexpr size_t GUARD = 4;
  constexpr float MARKER = 1234567.0f;
  const size_t count = size_t(c.m) * c.d, prefix = GUARD + c.output_offset;
  // Input tails are unpadded for memcheck; optional offsets are in floats.
  test::DeviceArray<float> q(c.q_offset + c.q.size()), k(c.k_offset + c.k.size());
  test::DeviceArray<float> v(c.v_offset + c.v.size()), output(prefix + count + GUARD);
  std::vector<float> initial(prefix + count + GUARD, MARKER);
  const int calls = c.reuse_output ? 3 : 2;
  bool passed = true;
  double max_error = 0;
  for (int call = 0; call < calls; ++call) {
    if (call == 1) {
      for (float& value : c.q) value = -value;
      for (float& value : c.v) value = -value;
    } else if (call == 2) {
      for (float& value : c.k) value = -value;
      c.alpha = -c.alpha;
    }
    const auto expected = reference(c);
    if (call == 0 && !c.known.empty()) {
      if (c.known.size() != expected.size()) return false;
      for (size_t i = 0; i < count; ++i)
        if (std::abs(expected[i] - c.known[i]) > 1e-12) {
          std::fprintf(stderr, "%s: CPU reference disagrees with known result at %zu\n", c.name.c_str(), i);
          return false;
        }
    }
    // A small separable case cross-checks the large-suite shortcut against full math.
    if (c.pattern == Pattern::Separable && c.m <= 17) {
      const auto full = reference(c, false);
      for (size_t i = 0; i < count; ++i)
        if (std::abs(expected[i] - full[i]) > 1e-12) {
          std::fprintf(stderr, "%s: analytic reference disagrees with full reference\n", c.name.c_str());
          return false;
        }
    }
    std::vector<float> stored_q(c.q_offset + c.q.size(), MARKER);
    std::vector<float> stored_k(c.k_offset + c.k.size(), MARKER);
    std::vector<float> stored_v(c.v_offset + c.v.size(), MARKER);
    std::copy(c.q.begin(), c.q.end(), stored_q.begin() + c.q_offset);
    std::copy(c.k.begin(), c.k.end(), stored_k.begin() + c.k_offset);
    std::copy(c.v.begin(), c.v.end(), stored_v.begin() + c.v_offset);
    q.upload(stored_q);
    k.upload(stored_k);
    v.upload(stored_v);
    if (!c.reuse_output || call == 0) {
      std::fill_n(initial.begin() + prefix, count,
                  call == 0 ? 8.0f : std::numeric_limits<float>::quiet_NaN());
      output.upload(initial);
    }
    solve(q.data() + c.q_offset, k.data() + c.k_offset, v.data() + c.v_offset,
          output.data() + prefix, c.m, c.n, c.d, c.alpha);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto actual = output.download();
    bool reported = false;
    for (size_t i = 0; i < count; ++i) {
      const double got = actual[prefix + i];
      const double error = std::isfinite(got) ? std::abs(got - expected[i])
                                            : std::numeric_limits<double>::infinity();
      max_error = std::max(max_error, error);
      const double tolerance = c.pattern == Pattern::ZeroV ? 0 : 1e-5 + 1e-5 * std::abs(expected[i]);
      if (!std::isfinite(expected[i]) || error > tolerance) {
        if (!reported)
          std::fprintf(stderr, "%s call=%d row=%zu col=%zu got=%.9g expected=%.9g\n",
                       c.name.c_str(), call + 1, i / c.d, i % c.d, got, expected[i]);
        reported = true;
        passed = false;
      }
    }
    if (std::memcmp(actual.data(), initial.data(), prefix * sizeof(float)) != 0 ||
        std::memcmp(actual.data() + prefix + count, initial.data() + prefix + count,
                    GUARD * sizeof(float)) != 0) {
      std::fprintf(stderr, "%s call=%d output guard overwritten\n", c.name.c_str(), call + 1);
      passed = false;
    }
    passed &= q.unchanged(stored_q);
    passed &= k.unchanged(stored_k);
    passed &= v.unchanged(stored_v);
  }
  std::printf("%s %-26s M/N/d=%d/%d/%d calls=%d max_abs_error=%.3g\n",
              passed ? "PASS" : "FAIL", c.name.c_str(), c.m, c.n, c.d, calls, max_error);
  return passed;
}

static std::vector<Case> quick_cases() {
  std::vector<Case> cases;
  Case first{"example-1-formula", 2, 3, 4, 0.5f, Pattern::Explicit};
  first.q = {1, 0, 0, 0, 0, 1, 0, 0};
  first.k = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  first.v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  // The statement's first output row has a typo in its last two columns.
  first.known = {3.046850655817304, 4.046850655817304, 5.046850655817304, 6.046850655817304,
                 3.9321744209817817, 4.932174420981782, 5.932174420981782, 6.932174420981782};
  cases.push_back(std::move(first));
  Case second{"example-2", 1, 2, 2, 0.8f, Pattern::Explicit};
  second.q = {1, 2}; second.k = {1, 0, 0, 1}; second.v = {3, 4, 5, 6};
  second.known = {3.953586755413498, 4.9535867554134985};
  cases.push_back(std::move(second));
  for (int m : {1, 15, 16, 17, 2047, 2048})
    cases.push_back({"rows-" + std::to_string(m), m, 3, 5, 1.0f / 4096});
  for (int n : {1, 15, 16, 17, 31, 32, 33, 255, 256, 257, 511, 513, 2047, 2048})
    cases.push_back({"keys-" + std::to_string(n), 3, n, 5, -1.0f / 4096});
  for (int d : {1, 3, 15, 16, 17, 31, 32, 33, 255, 256, 257, 1023, 1024})
    cases.push_back({"features-" + std::to_string(d), 3, 7, d, 0.125f});
  for (float alpha : {-1.0f, -0.5f, -0.125f, 0.0f, 0.125f, 0.5f, 1.0f})
    cases.push_back({"alpha-" + std::to_string(alpha), 3, 7, 5, alpha});
  cases.push_back({"rectangular-tails", 17, 33, 7, -0.5f});
  cases.push_back({"multiple-warps", 33, 513, 33, 1.0f / 128});
  cases.push_back({"zero-q-uniform", 17, 33, 7, 0, Pattern::ZeroQ});
  cases.push_back({"zero-k-bias", 17, 33, 7, -0.25f, Pattern::ZeroK});
  cases.push_back({"zero-v", 17, 33, 7, 0.25f, Pattern::ZeroV});
  cases.push_back({"constant-v", 17, 33, 7, 0.25f, Pattern::ConstantV});
  cases.push_back({"identical-query-rows", 17, 33, 7, 0.25f, Pattern::IdenticalQ});
  cases.push_back({"last-feature", 17, 33, 257, 0.125f, Pattern::TailFeature});
  cases.push_back({"last-key", 17, 257, 7, 0, Pattern::TailKey});
  cases.push_back({"identity-v-probabilities", 5, 33, 33, 0.125f, Pattern::IdentityV});
  for (int d : {1, 4, 16, 1024}) {
    Case c{"bias-independent-d" + std::to_string(d), 1, 2, d, 0.5f, Pattern::Explicit};
    c.q.assign(d, 0); c.k.assign(2 * d, 0); c.v.assign(2 * d, 0);
    std::fill(c.v.begin() + d, c.v.end(), 1);
    c.known.assign(d, 1 / (1 + std::exp(0.5)));
    cases.push_back(std::move(c));
  }
  for (int offset = 0; offset < 5; ++offset) {
    Case c{"offset-" + std::to_string(offset), 17, 33, 7, 0.125f};
    c.q_offset = offset == 0 || offset == 4;
    c.k_offset = offset == 1 || offset == 4;
    c.v_offset = offset == 2 || offset == 4;
    c.output_offset = offset == 3 || offset == 4;
    cases.push_back(std::move(c));
  }
  cases.push_back({"consecutive-calls", 2, 3, 4, 0.5f, Pattern::Random, true});
  cases.push_back({"consecutive-blocks", 17, 257, 33, 1.0f / 128, Pattern::Random, true});
  cases.push_back({"analytic-reference-check", 17, 33, 17, -0.125f, Pattern::Separable, true});
  return cases;
}

static std::vector<Case> large_cases() {
  // Extreme exponent stability tests remain deferred, as agreed in the review.
  // Small slopes exercise full dimensions without the deferred overflow cases.
  return {{"max-dimensions", 2048, 2048, 1024, 0, Pattern::Separable},
          {"large-positive-bias", 2047, 2048, 33, 1.0f / 4096, Pattern::Separable},
          {"large-negative-bias", 2048, 2047, 65, -1.0f / 4096, Pattern::Separable}};
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
  std::printf("GPU: %s\nALiBi: softmax(QK^T/sqrt(d) + alpha*(i-j))*V; CPU FP64 reference\n", prop.name);
  int failed = 0;
  for (auto& c : cases)
    if (!run_case(std::move(c))) ++failed;
  std::printf("ALiBi: %zu/%zu cases passed\n", cases.size() - failed, cases.size());
  return test::finish(failed == 0);
}
