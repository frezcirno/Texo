#include "test_utils.h"

extern "C" void solve(const float*, const float*, const float*, float*, int, int,
                      int, int);

static bool run_case(const char* name, int m, int n, int heads, int d,
                     bool zero = false) {
  const int ld = heads * d;
  const size_t q_count = size_t(m) * ld, kv_count = size_t(n) * ld;
  std::vector<float> q(q_count, 0.f), k(kv_count, 0.f), v(kv_count, 0.f);
  if (!zero) {
    std::mt19937 rng(m * 131 + n * 17 + d * 7 + heads);
    std::normal_distribution<float> dist(0.f, 1.f);
    for (float& x : q) x = dist(rng);
    for (float& x : k) x = dist(rng);
    for (float& x : v) x = dist(rng);
  }

  const double scale = 1.0 / std::sqrt(double(d));
  std::vector<double> expected(q_count), score(n);
  for (int h = 0; h < heads; ++h) {
    for (int i = 0; i < m; ++i) {
      const float* qi = &q[size_t(i) * ld + h * d];
      double max_score = -INFINITY;
      for (int j = 0; j < n; ++j) {
        const float* kj = &k[size_t(j) * ld + h * d];
        double dot = 0;
        for (int c = 0; c < d; ++c) dot += double(qi[c]) * kj[c];
        score[j] = dot * scale;
        max_score = std::max(max_score, score[j]);
      }
      double sum = 0;
      for (int j = 0; j < n; ++j) sum += score[j] = std::exp(score[j] - max_score);
      double* out = &expected[size_t(i) * ld + h * d];
      for (int c = 0; c < d; ++c) out[c] = 0;
      for (int j = 0; j < n; ++j)
        for (int c = 0; c < d; ++c)
          out[c] += score[j] / sum * v[size_t(j) * ld + h * d + c];
    }
  }

  test::DeviceArray<float> device_q(q), device_k(k), device_v(v);
  bool passed = test::check_output<float>(name, expected, [&](float* output) {
    solve(device_q.data(), device_k.data(), device_v.data(), output, m, n,
          heads, d);
  }, 1e-4, 1e-4);
  passed &= device_q.unchanged(q);
  passed &= device_k.unchanged(k);
  passed &= device_v.unchanged(v);
  return passed;
}

int main() {
  bool passed = true;
  passed &= run_case("single-query", 1, 5, 2, 4);
  passed &= run_case("single-key", 9, 1, 3, 8);
  passed &= run_case("d1-heads", 5, 7, 4, 1);
  passed &= run_case("odd-d3", 7, 11, 2, 3);
  passed &= run_case("odd-d5-m33", 33, 70, 3, 5);
  passed &= run_case("zero-inputs", 4, 6, 2, 4, true);
  passed &= run_case("d64-m100-n37", 100, 37, 4, 64);
  passed &= run_case("d32-m257-n129", 257, 129, 8, 32);
  passed &= run_case("d96-m77-n300", 77, 300, 4, 96);
  passed &= run_case("d200-m65", 65, 90, 1, 200);
  passed &= run_case("d1024-m130", 130, 66, 1, 1024);
  passed &= run_case("many-heads", 129, 200, 64, 8);
  passed &= run_case("long-split", 16, 4096, 1, 64);
  passed &= run_case("perf-shape", 1024, 2048, 16, 64);
  return test::finish(passed);
}
