#include <cmath>
#include <cuda_runtime.h>
#include <math_constants.h>

template <typename T> __device__ T pow2(T x) { return x * x; }

struct Weight {
  float embed_w0;
  float embed_w1;
  float q_proj_q0;
  float q_proj_q1;
  float v_proj_v0;
  float mlp_gate_a;
  float mlp_gate_c;
  float mlp_carry;
  float rms_n0;
  float rms_n1;
};

constexpr const float kOmega = 2 * CUDART_PI / 19;

// 0.5 * logf(10) / (cosf(0.3 * kOmega) - cosf(0.7 * kOmega));
constexpr const float kAttnScale = 52.9176499f;

template <typename T> __device__ __forceinline__ T silu(const T x) {
  return x / (1 + exp(-x));
}

template <typename T>
__host__ __device__ __forceinline__ T rms_norm(const T &val) {
  auto inv_rms = rsqrtf((val.x * val.x + val.y * val.y) / 2 + 1e-6);
  return {val.x * inv_rms, val.y * inv_rms};
}

template <typename T>
__host__ __device__ __forceinline__ T rope(const T &val, const int t) {
  float s, c;
  sincosf(t * kOmega, &s, &c);
  return {val.x * c - val.y * s, val.x * s + val.y * c};
}

__global__ void forward(const int *__restrict__ prompts, // (batch_size, 31)
                        float *__restrict__ output,      // (batch_size, 11, 10)
                        const Weight *__restrict__ w, const size_t batch_size) {
  const size_t batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch >= batch_size)
    return;

  // [:31] -> prompts
  // [0,
  // a_rev_0, ..., a_rev_9,
  // 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // b_rev_0, ..., b_rev_9,
  // 0]
  // [31:42] -> generated
  int generated[11];
  float2 k_cache[41];
  float v_cache[41];

  // Prefill
  for (int t = 0; t < 30; ++t) {
    const int d = prompts[batch * 31 + t];
    float2 emb = {w->embed_w0 - w->embed_w1 * d * d, static_cast<float>(-d)};

    float2 hidd = rms_norm(emb);

    float2 K = {hidd.x, 0};
    K = rms_norm(K);
    K = rope(K, t);

    float2 V = {hidd.y * w->v_proj_v0, 0};

    k_cache[t] = K;
    v_cache[t] = V.x;
  }

  // Decode
  for (int t = 30; t < 41; ++t) {
    const int d = (t == 30 ? prompts[batch * 31 + t] : generated[t - 31]);
    float2 emb = {w->embed_w0 - w->embed_w1 * d * d, static_cast<float>(-d)};
    float2 hidden = rms_norm(emb);

    // Attn proj
    float2 Q = {hidden.x * w->q_proj_q0, hidden.x * w->q_proj_q1};
    Q = rms_norm(Q);
    Q = rope(Q, t);

    float2 K = {hidden.x, 0};
    K = rms_norm(K);
    K = rope(K, t);

    float2 V = {hidden.y * w->v_proj_v0, 0};

    k_cache[t] = K;
    v_cache[t] = V.x;

    // (Q * Kt) * scale * V
    float max_score = -CUDART_INF_F;
    for (int j = 0; j <= t; ++j) {
      float score = (Q.x * k_cache[j].x + Q.y * k_cache[j].y) * kAttnScale;
      max_score = fmaxf(max_score, score);
    }

    float denom = 0.0f;
    float weighted_v0 = 0.0f;
    for (int j = 0; j <= t; ++j) {
      float score = (Q.x * k_cache[j].x + Q.y * k_cache[j].y) * kAttnScale;
      float e = expf(score - max_score);
      denom += e;
      weighted_v0 += e * v_cache[j];
    }

    float2 h = {emb.x, emb.y + weighted_v0 / denom};

    // mlp
    float2 h_norm = rms_norm(h);
    auto g0 = h_norm.x * w->mlp_gate_a + h_norm.y * w->mlp_gate_c;
    auto g1 = h_norm.x * (w->mlp_gate_a - w->mlp_gate_c / 1000.0f) +
              h_norm.y * w->mlp_gate_c;
    auto activation = silu(g1) * h_norm.x - silu(g0) * h_norm.x;

    h.y += w->mlp_carry * activation;

    float2 out = rms_norm(h);
    out.x *= w->rms_n0;
    out.y *= w->rms_n1;

    int best_digit = 0;
    float best_logit = -CUDART_INF_F;
    for (int d = 0; d < 10; ++d) {
      float logit = out.x * (w->embed_w0 - w->embed_w1 * d * d) +
                    out.y * (-static_cast<float>(d));

      output[(batch * 11 + (t - 30)) * 10 + d] = logit;

      if (logit > best_logit) {
        best_logit = logit;
        best_digit = d;
      }
    }
    generated[t - 30] = best_digit;
  }
}

// prompts, output, weights are device pointers
extern "C" void solve(const int *prompts,   // (batch_size, 31)
                      float *output,        // (batch_size, 11, 10)
                      const float *weights, // (10,)
                      int batch_size) {
  //
  const Weight *w = reinterpret_cast<const Weight *>(weights);
  forward<<<(batch_size + 255) / 256, 256>>>(prompts, output, w, batch_size);
}
