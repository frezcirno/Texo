#include "test_utils.h"

extern "C" void solve(const float*, const float*, const float*, float*, int,
                      int);

static bool run_case(const char* name, int m, int d, bool zero = false) {
  const size_t count = size_t(m) * d;
  std::vector<float> q(count, 0.f), k(count, 0.f), v(count, 0.f);
  if (!zero) {
    std::mt19937 rng(m * 131 + d * 7);
    std::normal_distribution<float> dist(0.f, 1.f);
    for (float& x : q) x = dist(rng);
    for (float& x : k) x = dist(rng);
    for (float& x : v) x = dist(rng);
  }

  // Query i attends to keys 0..i.
  const double scale = 1.0 / std::sqrt(double(d));
  std::vector<double> expected(count), score(m);
  for (int i = 0; i < m; ++i) {
    const float* qi = &q[size_t(i) * d];
    double max_score = -INFINITY;
    for (int j = 0; j <= i; ++j) {
      const float* kj = &k[size_t(j) * d];
      double dot = 0;
      for (int c = 0; c < d; ++c) dot += double(qi[c]) * kj[c];
      score[j] = dot * scale;
      max_score = std::max(max_score, score[j]);
    }
    double sum = 0;
    for (int j = 0; j <= i; ++j) sum += score[j] = std::exp(score[j] - max_score);
    double* out = &expected[size_t(i) * d];
    for (int c = 0; c < d; ++c) out[c] = 0;
    for (int j = 0; j <= i; ++j)
      for (int c = 0; c < d; ++c) out[c] += score[j] / sum * v[size_t(j) * d + c];
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
  passed &= run_case("single-row", 1, 4);
  passed &= run_case("d1", 7, 1);
  passed &= run_case("odd-d3", 11, 3);
  passed &= run_case("odd-d5-m65", 65, 5);
  passed &= run_case("zero-inputs", 6, 4, true);
  passed &= run_case("m63-d64", 63, 64);
  passed &= run_case("m64-d64", 64, 64);
  passed &= run_case("m129-d32", 129, 32);
  passed &= run_case("m300-d96", 300, 96);
  passed &= run_case("m200-d128", 200, 128);
  passed &= run_case("m130-d200", 130, 200);
  passed &= run_case("split-m1000-d64", 1000, 64);
  passed &= run_case("split-m4096-d16", 4096, 16);
  passed &= run_case("perf-shape", 2048, 128);
  return test::finish(passed);
}
