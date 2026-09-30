#include <cuda_runtime.h>

__global__ void top_p_kernel(const float *logits, // (vocab_size,)
                             const float *p,      // (1,)
                             const int *seed,     // (1,)
                             int *sampled_token,  // (N,)
                             int vocab_size) {
  //
}

extern "C" void solve(const float *logits, // (vocab_size,)
                      const float *p,      // (1,)
                      const int *seed,     // (1,)
                      int *sampled_token,  // (N,)
                      int vocab_size) {
  top_p_kernel<<<>>>(logits, p, seed, sampled_token, vocab_size);
}
