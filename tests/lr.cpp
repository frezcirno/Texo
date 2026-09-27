#include "test_utils.h"
#include <chrono>

extern "C" void solve(const float *, const float *, float *, int, int);

// Independent FP64 Newton reference for SUM(BCE) + lambda/2 * ||beta||^2.
// Both CUDA optimizers are checked against this reference. Its linear solve
// uses pivoted Gaussian elimination rather than the CUDA Cholesky helpers.
constexpr double LAMBDA = 1e-6;

static double objective(const std::vector<float>& x, const std::vector<float>& y,
                        const std::vector<double>& beta) {
  const int n = beta.size();
  double loss = 0;
  for (size_t i = 0; i < y.size(); ++i) {
    double z = 0;
    for (int j = 0; j < n; ++j) z += double(x[i * n + j]) * beta[j];
    double margin = (2.0 * y[i] - 1.0) * z;
    loss += std::max(-margin, 0.0) + std::log1p(std::exp(-std::abs(margin)));
  }
  for (double b : beta) loss += 0.5 * LAMBDA * b * b;
  return loss;
}

static std::vector<double> reference(const std::vector<float>& x,
                                     const std::vector<float>& y, int n) {
  std::vector<double> beta(n, 0.0);
  for (int step = 0; step < 100; ++step) {
    std::vector<double> g(n), h(n * n, 0.0);
    for (int j = 0; j < n; ++j) {
      g[j] = LAMBDA * beta[j];
      h[j * n + j] = LAMBDA;
    }
    for (size_t i = 0; i < y.size(); ++i) {
      double z = 0;
      for (int j = 0; j < n; ++j) z += double(x[i * n + j]) * beta[j];
      double p = 1.0 / (1.0 + std::exp(-z));
      for (int j = 0; j < n; ++j) {
        g[j] += double(x[i * n + j]) * (p - y[i]);
        for (int k = 0; k < n; ++k)
          h[j * n + k] += double(x[i * n + j]) * x[i * n + k] * p * (1 - p);
      }
    }
    double norm = 0;
    for (double v : g) norm = std::max(norm, std::abs(v));
    if (norm < 1e-13) return beta;
    // Gaussian elimination with pivoting; intentionally independent of CUDA's
    // Cholesky implementation.
    std::vector<double> rhs = g, delta(n);
    for (int j = 0; j < n; ++j) {
      int pivot = j;
      for (int i = j + 1; i < n; ++i)
        if (std::abs(h[i * n + j]) > std::abs(h[pivot * n + j])) pivot = i;
      for (int k = 0; k < n; ++k) std::swap(h[j * n + k], h[pivot * n + k]);
      std::swap(rhs[j], rhs[pivot]);
      for (int i = j + 1; i < n; ++i) {
        const double f = h[i * n + j] / h[j * n + j];
        for (int k = j; k < n; ++k) h[i * n + k] -= f * h[j * n + k];
        rhs[i] -= f * rhs[j];
      }
    }
    for (int j = n - 1; j >= 0; --j) {
      double v = rhs[j];
      for (int k = j + 1; k < n; ++k) v -= h[j * n + k] * delta[k];
      delta[j] = v / h[j * n + j];
    }
    const double before = objective(x, y, beta);
    double directional = 0;
    for (int j = 0; j < n; ++j) directional += g[j] * delta[j];
    double rate = 1;
    std::vector<double> trial(n);
    for (int retry = 0; retry < 60; ++retry) {
      for (int j = 0; j < n; ++j) trial[j] = beta[j] - rate * delta[j];
      if (objective(x, y, trial) <= before - 1e-4 * rate * directional + 1e-14)
        break;
      rate *= 0.5;
    }
    beta = trial;
  }
  std::fprintf(stderr, "CPU LR reference did not converge\n");
  std::exit(EXIT_FAILURE);
}

static bool run_case(const char* name, const std::vector<float>& x,
                     std::vector<float> y, const std::vector<double>& expected,
                     const std::vector<double>& platform = {}) {
  const int m = y.size(), n = expected.size();
  constexpr int GUARD = 4;
  constexpr float MARKER = 1234567.f;
  test::DeviceArray<float> dx(x), dy(y), beta(n + 2 * GUARD);
  bool passed = true;
  double max_error = 0, elapsed_ms = 0;
  for (int call = 0; call < 2; ++call) {
    // Flipping every label must negate the L2-regularized solution. Reuse the
    // allocations and poison beta to exercise initialization on repeated calls.
    if (call) for (float& label : y) label = 1.f - label;
    dy.upload(y);
    std::vector<float> initial(n + 2 * GUARD, MARKER);
    std::fill(initial.begin() + GUARD, initial.begin() + GUARD + n, -123.f);
    beta.upload(initial);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::steady_clock::now();
    solve(dx.data(), dy.data(), beta.data() + GUARD, m, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    elapsed_ms += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - start).count();
    auto actual = beta.download();
    for (int j = 0; j < n; ++j) {
      double ref = call ? -expected[j] : expected[j];
      double got = actual[GUARD + j], error = std::abs(got - ref);
      max_error = std::max(max_error, error);
      if (!std::isfinite(got) || error > 2e-3 + 1e-5 * std::abs(ref)) {
        std::fprintf(stderr, "%s call=%d j=%d got=%.12g reference=%.12g\n",
                     name, call, j, got, ref);
        passed = false;
      }
      if (!platform.empty() && call == 0) {
        passed &= std::abs(got - platform[j]) <= 1e-2 + 1e-2 * std::abs(platform[j]);
        std::printf("  supplied case beta[%d]=%.9g expected=%.9g\n", j, got, platform[j]);
      }
    }
    for (int j = 0; j < GUARD; ++j)
      passed &= actual[j] == MARKER && actual[GUARD + n + j] == MARKER;
    passed &= dx.unchanged(x) && dy.unchanged(y);
  }
  std::printf("%s %-24s M=%d N=%d max_abs_error=%.3g solve_ms=%.3f\n",
              passed ? "PASS" : "FAIL", name, m, n, max_error, elapsed_ms / 2);
  return passed;
}

int main() {
#ifdef LR_OPTIMIZER
  std::printf("LR optimizer: %s\n", LR_OPTIMIZER);
#else
  std::puts("LR optimizer: source default");
#endif
  bool passed = true;
  const std::vector<float> separable = {.125f,.658f,.623f, -.802f,-.234f,-.858f,
                                       .929f,.044f,.474f};
  const std::vector<float> labels = {1,0,1};
  auto ref = reference(separable, labels, 3);
  passed &= run_case("separable regression", separable, labels, ref,
                    {7.599221706390381,6.970426082611084,9.579840660095215});
  const std::vector<float> example = {2,1, 1,2, 3,3, 1.5,2.5,
                                     -1,-2, -2,-1, -1.5,-2.5, -3,-3};
  const std::vector<float> example_y = {1,1,1,0,0,0,1,0};
  passed &= run_case("nonseparable example", example, example_y,
                     reference(example, example_y, 2));
  // Zero rows add only a constant to SUM(BCE), so the optimum must not change.
  // This also detects accidentally using the mean-loss definition of lambda.
  auto padded = separable;
  auto padded_y = labels;
  padded.resize(257 * 3, 0.f);
  padded_y.resize(257, 0.f);
  passed &= run_case("sample tail / loss sum", padded, padded_y, ref);
  for (float scale : {0.001f, 1.f, 10.f}) {
    std::vector<float> x(8, scale), y = {1,1,1,1,1,1,0,0};
    passed &= run_case("scaled scalar", x, y, reference(x, y, 1));
  }
  for (bool duplicate : {false, true}) {
    std::vector<float> x(16, 0.f), y = {1,1,1,1,1,1,0,0};
    for (int i = 0; i < 8; ++i) { x[2*i] = 1; x[2*i+1] = duplicate ? 1 : 0; }
    passed &= run_case(duplicate ? "duplicate columns" : "zero column", x, y,
                       reference(x, y, 2));
  }
  passed &= run_case("all zero", std::vector<float>(17 * 5, 0.f),
                     std::vector<float>(17, 1.f), std::vector<double>(5, 0));
  passed &= run_case("one sample", {1.f}, {1.f}, reference({1.f}, {1.f}, 1));
  // Independent feature groups provide scalar CPU references without a large
  // CPU matrix solve, and exercise >256 features and a partial final block.
  const int n = 257, m = 8 * n;
  std::vector<float> x(m * n, 0.f), y(m);
  std::vector<double> wide_ref(n);
  for (int j = 0; j < n; ++j) {
    const float scale = j % 3 == 0 ? 0.25f : (j % 3 == 1 ? 1.f : 4.f);
    std::vector<float> sy = {1,1,1,1,1,1,0,0};
    wide_ref[j] = reference(std::vector<float>(8, scale), sy, 1)[0];
    for (int i = 0; i < 8; ++i) { x[(8*j+i)*n+j] = scale; y[8*j+i] = sy[i]; }
  }
  passed &= run_case("feature tail", x, y, wide_ref);
  return test::finish(passed);
}
