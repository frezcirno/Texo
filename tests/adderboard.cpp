#include "test_utils.h"

#include <math_constants.h>

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

extern "C" void solve(const int* prompts, float* output, const float* weights, int batch_size);

namespace {

constexpr int kPromptLen = 31;
constexpr int kDecodeSteps = 11;
constexpr int kVocabSize = 10;
constexpr float kRmsEps = 1e-6f;
constexpr float kEmbedConst = 1000.0f;
constexpr float kOmega = 2.0f * CUDART_PI / 19.0f;
constexpr float kAttnScale = 52.9176499f;

struct Float2 {
  float x, y;
};

struct Case {
  std::string name;
  std::vector<std::pair<uint64_t, uint64_t>> pairs;
};

static std::array<float, 10> model_weights() {
  constexpr float kConstNorm = 1.4142135623730950488f;
  constexpr float kDigitScale = kEmbedConst / kConstNorm;
  constexpr float kCarryAlpha = 256.0f / kConstNorm;
  constexpr float kPhi = kOmega * 10.3f;
  std::array<float, 10> w{};
  w[0] = kEmbedConst;
  w[1] = 1e-3f;
  w[2] = std::cos(kPhi);
  w[3] = -std::sin(kPhi);
  w[4] = -22.0f * kDigitScale;
  w[5] = kCarryAlpha * -94.0f / kConstNorm;
  w[6] = kCarryAlpha * kDigitScale;
  w[7] = (100.0f / kCarryAlpha) / kConstNorm;
  w[8] = (0.1f / 1e-3f) / kConstNorm;
  w[9] = -kDigitScale / 50.0f;
  return w;
}

static std::vector<int> encode_prompts(
    const std::vector<std::pair<uint64_t, uint64_t>>& pairs) {
  std::vector<int> prompts(pairs.size() * kPromptLen, 0);
  for (size_t batch = 0; batch < pairs.size(); ++batch) {
    uint64_t a = pairs[batch].first;
    uint64_t b = pairs[batch].second;
    for (int i = 0; i < 10; ++i) {
      prompts[batch * kPromptLen + 1 + i] = int(a % 10);
      prompts[batch * kPromptLen + 20 + i] = int(b % 10);
      a /= 10;
      b /= 10;
    }
  }
  return prompts;
}

static Float2 rms_norm(Float2 x) {
  const float inv_rms = 1.0f / std::sqrt((x.x * x.x + x.y * x.y) * 0.5f + kRmsEps);
  return {x.x * inv_rms, x.y * inv_rms};
}

static Float2 rope(Float2 x, int position) {
  const float angle = float(position) * kOmega;
  const float sine = std::sin(angle);
  const float cosine = std::cos(angle);
  return {x.x * cosine - x.y * sine, x.x * sine + x.y * cosine};
}

static float silu(float x) {
  return x / (1.0f + std::exp(-x));
}

static Float2 embed(const std::array<float, 10>& w, int digit) {
  return {w[0] - w[1] * float(digit * digit), -float(digit)};
}

// Scalar reference of the LeetGPU model. It deliberately evaluates only the
// final causal-attention row, which is mathematically equivalent for this
// single-layer architecture and matches the fused CUDA implementation.
static std::vector<float> reference_logits(const std::vector<int>& prompts,
                                           const std::array<float, 10>& w,
                                           int batch_size) {
  std::vector<float> output(size_t(batch_size) * kDecodeSteps * kVocabSize);
  for (int batch = 0; batch < batch_size; ++batch) {
    int generated[kDecodeSteps];
    for (int step = 0; step < kDecodeSteps; ++step) {
      const int position = 30 + step;
      const auto token_at = [&](int index) {
        return index < kPromptLen ? prompts[size_t(batch) * kPromptLen + index]
                                  : generated[index - kPromptLen];
      };

      const Float2 query_hidden = rms_norm(embed(w, token_at(position)));
      Float2 query = rms_norm({query_hidden.x * w[2], query_hidden.x * w[3]});
      query = rope(query, position);

      float scores[41];
      float max_score = -std::numeric_limits<float>::infinity();
      for (int key_position = 0; key_position <= position; ++key_position) {
        const Float2 hidden = rms_norm(embed(w, token_at(key_position)));
        const Float2 key = rope(rms_norm({hidden.x, 0.0f}), key_position);
        const float score = (query.x * key.x + query.y * key.y) * kAttnScale;
        scores[key_position] = score;
        max_score = std::max(max_score, score);
      }

      float denominator = 0.0f;
      float weighted_value = 0.0f;
      for (int key_position = 0; key_position <= position; ++key_position) {
        const Float2 hidden = rms_norm(embed(w, token_at(key_position)));
        const float value = hidden.y * w[4];
        const float probability_unnormalized = std::exp(scores[key_position] - max_score);
        denominator += probability_unnormalized;
        weighted_value += probability_unnormalized * value;
      }

      Float2 h = embed(w, token_at(position));
      h.y += weighted_value / denominator;

      const Float2 h_norm = rms_norm(h);
      const float g0 = h_norm.x * w[5] + h_norm.y * w[6];
      const float g1 = h_norm.x * (w[5] - w[6] / kEmbedConst) + h_norm.y * w[6];
      h.y += w[7] * (silu(g1) * h_norm.x - silu(g0) * h_norm.x);

      Float2 final = rms_norm(h);
      final.x *= w[8];
      final.y *= w[9];

      int best_digit = 0;
      float best_logit = -std::numeric_limits<float>::infinity();
      for (int digit = 0; digit < kVocabSize; ++digit) {
        const Float2 digit_embedding = embed(w, digit);
        const float logit = final.x * digit_embedding.x + final.y * digit_embedding.y;
        output[(size_t(batch) * kDecodeSteps + step) * kVocabSize + digit] = logit;
        if (logit > best_logit) {
          best_logit = logit;
          best_digit = digit;
        }
      }
      generated[step] = best_digit;
    }
  }
  return output;
}

static std::array<int, kDecodeSteps> sum_digits(uint64_t a, uint64_t b) {
  std::array<int, kDecodeSteps> digits{};
  uint64_t sum = a + b;
  for (int& digit : digits) {
    digit = int(sum % 10);
    sum /= 10;
  }
  return digits;
}

static int argmax(const float* logits) {
  int best = 0;
  for (int digit = 1; digit < kVocabSize; ++digit)
    if (logits[digit] > logits[best]) best = digit;
  return best;
}

static bool run_case(const Case& c) {
  constexpr size_t kGuard = 4;
  constexpr float kMarker = 1234567.0f;
  constexpr double kAtol = 1e-2;
  constexpr double kRtol = 1e-2;
  const int batch_size = int(c.pairs.size());
  const auto prompts = encode_prompts(c.pairs);
  const auto weights = model_weights();
  const auto expected = reference_logits(prompts, weights, batch_size);
  const size_t output_count = expected.size();

  test::DeviceArray<int> device_prompts(prompts);
  test::DeviceArray<float> device_weights(
      std::vector<float>(weights.begin(), weights.end()));
  std::vector<float> initial(output_count + 2 * kGuard, 0.0f);
  std::fill(initial.begin(), initial.begin() + kGuard, kMarker);
  std::fill(initial.end() - kGuard, initial.end(), kMarker);
  test::DeviceArray<float> device_output(initial);

  bool passed = true;
  double max_error = 0.0;
  for (int call = 0; call < 2; ++call) {
    device_output.upload(initial);
    solve(device_prompts.data(), device_output.data() + kGuard, device_weights.data(), batch_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto actual = device_output.download();

    for (size_t i = 0; i < kGuard; ++i) {
      if (actual[i] != kMarker || actual[kGuard + output_count + i] != kMarker) {
        std::fprintf(stderr, "%s call=%d output guard overwritten\n", c.name.c_str(), call + 1);
        passed = false;
        break;
      }
    }
    for (size_t i = 0; i < output_count; ++i) {
      const double error = std::abs(double(actual[kGuard + i]) - double(expected[i]));
      max_error = std::max(max_error, error);
      if (!std::isfinite(actual[kGuard + i]) ||
          error > kAtol + kRtol * std::abs(double(expected[i]))) {
        std::fprintf(stderr, "%s call=%d logit=%zu got=%.9g expected=%.9g\n",
                     c.name.c_str(), call + 1, i, actual[kGuard + i], expected[i]);
        passed = false;
        break;
      }
    }
    for (int batch = 0; batch < batch_size; ++batch) {
      const auto expected_digits = sum_digits(c.pairs[batch].first, c.pairs[batch].second);
      for (int step = 0; step < kDecodeSteps; ++step) {
        const float* logits = actual.data() +
            kGuard + (size_t(batch) * kDecodeSteps + step) * kVocabSize;
        if (argmax(logits) != expected_digits[step]) {
          std::fprintf(stderr, "%s call=%d batch=%d step=%d decoded=%d expected=%d\n",
                       c.name.c_str(), call + 1, batch, step, argmax(logits), expected_digits[step]);
          passed = false;
          break;
        }
      }
    }
  }
  passed &= device_prompts.unchanged(prompts);
  passed &= device_weights.unchanged(std::vector<float>(weights.begin(), weights.end()));
  std::printf("%s %-24s batch=%d logits=%zu max_abs_error=%.3g\n",
              passed ? "PASS" : "FAIL", c.name.c_str(), batch_size, output_count, max_error);
  return passed;
}

static std::vector<Case> quick_cases() {
  std::vector<Case> cases;
  cases.push_back({"zero", {{0, 0}}});
  cases.push_back({"published-examples", {{3, 5}, {99, 1}}});
  cases.push_back({"carry-chain", {{9999999999ULL, 1}}});
  cases.push_back({"all-zero-b8", std::vector<std::pair<uint64_t, uint64_t>>(8, {0, 0})});
  cases.push_back({"max-values-b4",
                   std::vector<std::pair<uint64_t, uint64_t>>(
                       4, {9999999999ULL, 9999999999ULL})});

  std::mt19937_64 rng(42);
  for (const auto& shape : std::array<std::pair<const char*, int>, 5>{
           {{"random-b1", 1}, {"random-b16", 16}, {"random-b64", 64},
            {"random-b30", 30}, {"random-b100", 100}}}) {
    Case c{shape.first, {}};
    for (int i = 0; i < shape.second; ++i)
      c.pairs.push_back({rng() % 10000000000ULL, rng() % 10000000000ULL});
    cases.push_back(std::move(c));
  }
  return cases;
}

static Case large_case() {
  Case c{"performance-b100000", {}};
  c.pairs.reserve(100000);
  std::mt19937_64 rng(123);
  for (int i = 0; i < 100000; ++i)
    c.pairs.push_back({rng() % 10000000000ULL, rng() % 10000000000ULL});
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  bool large = false;
  if (argc == 2 && std::strcmp(argv[1], "--large") == 0) large = true;
  else if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [--large]\n", argv[0]);
    return 2;
  }

  bool passed = true;
  if (large) {
    passed &= run_case(large_case());
  } else {
    for (const auto& c : quick_cases()) passed &= run_case(c);
  }
  return test::finish(passed);
}
