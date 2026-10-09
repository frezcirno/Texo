#include "test_utils.h"

extern "C" void solve(const float*, const float*, const float*, float*, int,
                      int);

static double phi(double x) { return x > 0 ? x + 1 : std::exp(x); }

static bool run_case(const char* name, int m, int d, float low, float high,
                     std::vector<float> q = {}, std::vector<float> k = {},
                     std::vector<float> v = {}) {
  const size_t count = size_t(m) * d;
  if (q.empty()) {
    std::mt19937 rng(m * 131 + d * 7);
    std::uniform_real_distribution<float> dist(low, high);
    q.resize(count), k.resize(count), v.resize(count);
    for (float& x : q) x = dist(rng);
    for (float& x : k) x = dist(rng);
    for (float& x : v) x = dist(rng);
  }

  // S = phi(K)^T V and z = sum_j phi(K_j), then phi(Q_i) S / phi(Q_i) z.
  std::vector<double> state(size_t(d) * d, 0.0), z(d, 0.0);
  for (int j = 0; j < m; ++j)
    for (int a = 0; a < d; ++a) {
      const double f = phi(k[size_t(j) * d + a]);
      z[a] += f;
      for (int b = 0; b < d; ++b) state[size_t(a) * d + b] += f * v[size_t(j) * d + b];
    }
  std::vector<double> expected(count, 0.0), f(d);
  for (int i = 0; i < m; ++i) {
    double den = 0;
    for (int a = 0; a < d; ++a) den += (f[a] = phi(q[size_t(i) * d + a])) * z[a];
    for (int b = 0; b < d; ++b) {
      double num = 0;
      for (int a = 0; a < d; ++a) num += f[a] * state[size_t(a) * d + b];
      expected[size_t(i) * d + b] = num / den;
    }
  }

  test::DeviceArray<float> device_q(q), device_k(k), device_v(v);
  bool passed = test::check_output<float>(name, expected, [&](float* output) {
    solve(device_q.data(), device_k.data(), device_v.data(), output, m, d);
  }, 1e-4, 1e-4);
  passed &= device_q.unchanged(q);
  passed &= device_k.unchanged(k);
  passed &= device_v.unchanged(v);
  return passed;
}

int main() {
  bool passed = true;
  passed &= run_case("example-1", 2, 4, 0, 0, {1, 0, 0, 0, 0, 1, 0, 0},
                     {1, 0, 0, 0, 0, 1, 0, 0}, {1, 2, 3, 4, 5, 6, 7, 8});
  passed &= run_case("example-2", 2, 2, 0, 0, {0, 0, 1, 1}, {1, 0, 0, 1},
                     {3, 4, 5, 6});
  passed &= run_case("zero-inputs", 3, 5, 0, 0, std::vector<float>(15, 0.f),
                     std::vector<float>(15, 0.f), std::vector<float>(15, 0.f));
  passed &= run_case("mixed-d3", 4, 3, 0, 0,
                     {-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12},
                     {2, -1, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12},
                     {1, 0.5f, -0.5f, -1, 2, 3, 4, -2, 1, 0, 1, -1});
  passed &= run_case("single-row", 1, 1, -100, 100);
  passed &= run_case("m128-d32", 128, 32, -0.1f, 0.1f);
  passed &= run_case("odd-m97-d7", 97, 7, -100, 100);
  passed &= run_case("m33-d127", 33, 127, -100, 100);
  passed &= run_case("m1000-d64", 1000, 64, -100, 100);
  passed &= run_case("m3001-d100", 3001, 100, -100, 100);
  passed &= run_case("perf-shape", 10000, 128, -100, 100);
  return test::finish(passed);
}
