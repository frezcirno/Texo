#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace {
void check(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "%s failed: cuBLAS status %d\n", operation, int(status));
    std::exit(EXIT_FAILURE);
  }
}

// The standalone benchmark uses one selected device and the default stream.
// Create once before timing; destroying a handle on every call would synchronize.
struct Handle {
  cublasHandle_t value;
  Handle() {
    check(cublasCreate(&value), "cublasCreate");
    // Keep split-K reductions in FP32 as well as the matrix accumulation.
    check(cublasSetMathMode(value, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION),
          "cublasSetMathMode");
  }
  ~Handle() { cublasDestroy(value); }
};
} // namespace

// Row-major C = alpha * A * B + beta * C; FP16 storage, FP32 accumulation.
extern "C" void solve(const half* A, const half* B, half* C,
                      int M, int N, int K, float alpha, float beta) {
  if (M <= 0 || N <= 0) return;
  static thread_local Handle handle;

  // cuBLAS is column-major: reinterpret row-major buffers as transposes and
  // compute C^T = B^T * A^T. No data transpose or extra allocation is needed.
  check(cublasGemmEx(handle.value, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K, &alpha,
                     B, CUDA_R_16F, N,
                     A, CUDA_R_16F, std::max(1, K),
                     &beta, C, CUDA_R_16F, N,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
        "cublasGemmEx");
}
