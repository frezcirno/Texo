#include "test_utils.h"

extern "C" void solve(const float*, const float*, const float*, float*, int, int,
                      int);

static bool run_case(const char* name, int n, int d_model, int heads,
                     bool zero = false) {
  const size_t count = size_t(n) * d_model;
  std::vector<float> q(count, 0.f), k(count, 0.f), v(count, 0.f);
  if (!zero) {
    std::mt19937 rng(n * 131 + d_model * 7 + heads);
    std::normal_distribution<float> dist(0.f, 1.f);
    for (float& x : q) x = dist(rng);
    for (float& x : k) x = dist(rng);
    for (float& x : v) x = dist(rng);
  }

  const int d = d_model / heads;
  const double scale = 1.0 / std::sqrt(double(d));
  std::vector<double> expected(count), score(n);
  for (int h = 0; h < heads; ++h) {
    for (int i = 0; i < n; ++i) {
      const float* qi = &q[size_t(i) * d_model + h * d];
      double max_score = -INFINITY;
      for (int j = 0; j < n; ++j) {
        const float* kj = &k[size_t(j) * d_model + h * d];
        double dot = 0;
        for (int c = 0; c < d; ++c) dot += double(qi[c]) * kj[c];
        score[j] = dot * scale;
        max_score = std::max(max_score, score[j]);
      }
      double sum = 0;
      for (int j = 0; j < n; ++j) sum += score[j] = std::exp(score[j] - max_score);
      double* out = &expected[size_t(i) * d_model + h * d];
      for (int c = 0; c < d; ++c) out[c] = 0;
      for (int j = 0; j < n; ++j)
        for (int c = 0; c < d; ++c)
          out[c] += score[j] / sum * v[size_t(j) * d_model + h * d + c];
    }
  }

  test::DeviceArray<float> device_q(q), device_k(k), device_v(v);
  bool passed = test::check_output<float>(name, expected, [&](float* output) {
    solve(device_q.data(), device_k.data(), device_v.data(), output, n, d_model,
          heads);
  }, 1e-4, 1e-4);
  passed &= device_q.unchanged(q);
  passed &= device_k.unchanged(k);
  passed &= device_v.unchanged(v);
  return passed;
}

int main() {
  bool passed = true;
  passed &= run_case("single-token", 1, 8, 2);
  passed &= run_case("d1-heads", 5, 4, 4);
  passed &= run_case("odd-d3", 7, 6, 2);
  passed &= run_case("odd-d5-n33", 33, 15, 3);
  passed &= run_case("zero-inputs", 4, 8, 2, true);
  passed &= run_case("leetgpu-example", 2, 4, 2);
  passed &= run_case("d64-n100", 100, 256, 4);
  passed &= run_case("d32-n257", 257, 256, 8);
  passed &= run_case("d96-n77", 77, 384, 4);
  passed &= run_case("d200-n65", 65, 200, 1);
  passed &= run_case("d1024-n130", 130, 1024, 1);
  passed &= run_case("many-heads", 129, 512, 64);
  passed &= run_case("long-split", 2048, 64, 1);
  passed &= run_case("perf-shape", 1024, 1024, 16);
  return test::finish(passed);
}
