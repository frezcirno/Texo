#include "test_utils.h"

extern "C" void solve(const float*, const float*, const float*, float*, int, int,
                      int, int);

static bool run_case(const char* name, int q_heads, int kv_heads, int seq_len,
                     int head_dim, bool zero = false) {
  const size_t q_count = size_t(q_heads) * seq_len * head_dim;
  const size_t kv_count = size_t(kv_heads) * seq_len * head_dim;
  std::vector<float> q(q_count, 0.f), k(kv_count, 0.f), v(kv_count, 0.f);
  if (!zero) {
    std::mt19937 rng(q_heads * 131 + seq_len * 7 + head_dim);
    std::normal_distribution<float> dist(0.f, 1.f);
    for (float& x : q) x = dist(rng);
    for (float& x : k) x = dist(rng);
    for (float& x : v) x = dist(rng);
  }

  const int group = q_heads / kv_heads;
  const double scale = 1.0 / std::sqrt(double(head_dim));
  std::vector<double> expected(q_count), score(seq_len);
  for (int h = 0; h < q_heads; ++h) {
    const float* kh = &k[size_t(h / group) * seq_len * head_dim];
    const float* vh = &v[size_t(h / group) * seq_len * head_dim];
    for (int i = 0; i < seq_len; ++i) {
      const float* qi = &q[(size_t(h) * seq_len + i) * head_dim];
      double max_score = -INFINITY;
      for (int j = 0; j < seq_len; ++j) {
        double dot = 0;
        for (int c = 0; c < head_dim; ++c) dot += double(qi[c]) * kh[size_t(j) * head_dim + c];
        score[j] = dot * scale;
        max_score = std::max(max_score, score[j]);
      }
      double sum = 0;
      for (int j = 0; j < seq_len; ++j) sum += score[j] = std::exp(score[j] - max_score);
      double* out = &expected[(size_t(h) * seq_len + i) * head_dim];
      for (int c = 0; c < head_dim; ++c) out[c] = 0;
      for (int j = 0; j < seq_len; ++j)
        for (int c = 0; c < head_dim; ++c)
          out[c] += score[j] / sum * vh[size_t(j) * head_dim + c];
    }
  }

  test::DeviceArray<float> device_q(q), device_k(k), device_v(v);
  bool passed = test::check_output<float>(name, expected, [&](float* output) {
    solve(device_q.data(), device_k.data(), device_v.data(), output, q_heads,
          kv_heads, seq_len, head_dim);
  }, 1e-4, 1e-4);
  passed &= device_q.unchanged(q);
  passed &= device_k.unchanged(k);
  passed &= device_v.unchanged(v);
  return passed;
}

int main() {
  bool passed = true;
  passed &= run_case("mqa-single-token", 4, 1, 1, 8);
  passed &= run_case("tiny-d4", 2, 1, 2, 4);
  passed &= run_case("zero-inputs", 4, 2, 4, 8, true);
  passed &= run_case("groups4", 8, 2, 16, 32);
  passed &= run_case("s32-d64", 4, 2, 32, 64);
  passed &= run_case("s30-d32", 4, 2, 30, 32);
  passed &= run_case("s100-g2", 6, 3, 100, 32);
  passed &= run_case("mistral-s255", 8, 1, 255, 64);
  passed &= run_case("mha", 8, 8, 64, 32);
  passed &= run_case("s128-d64", 8, 2, 128, 64);
  passed &= run_case("d96-s77", 6, 2, 77, 96);
  passed &= run_case("d128-s300", 4, 2, 300, 128);
  passed &= run_case("d256-s129", 4, 2, 129, 256);
  passed &= run_case("d136-s33", 2, 1, 33, 136);
  passed &= run_case("long-s4096", 2, 1, 4096, 64);
  passed &= run_case("perf-shape", 32, 8, 1024, 128);
  return test::finish(passed);
}
