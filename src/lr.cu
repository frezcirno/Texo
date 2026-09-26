#include <cuda_runtime.h>

#include <cmath>
#include <cuda_runtime.h>

template <bool transposeA = false, typename T1, typename T2, typename T3>
__global__ void gemm_kernel(const T1 *__restrict__ A, // (M, K)
                            const T2 *__restrict__ B, // (K, N)
                            T3 *__restrict__ C,       // (M, N)
                            size_t M, size_t N, size_t K) {
  const size_t cx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t cy = blockIdx.y * blockDim.y + threadIdx.y;
  if (cy >= M || cx >= N)
    return;
  T3 sum = 0.0;
  for (size_t i = 0; i < K; i++) {
    sum += T3(transposeA ? A[i * M + cy] : A[cy * K + i]) * T3(B[i * N + cx]);
  }
  C[cy * N + cx] = sum;
}

template <bool transposeA = false, typename T1, typename T2, typename T3>
void gemm(const T1 *__restrict__ A, // (M, K)
          const T2 *__restrict__ B, // (K, N)
          T3 *__restrict__ C,       // (M, N)
          size_t M, size_t N, size_t K) {
  gemm_kernel<transposeA, T1, T2, T3>
      <<<dim3((N + 15) / 16, (M + 15) / 16), dim3(16, 16)>>>(A, B, C, M, N, K);
}

__global__ void
cholesky_decomposion_kernel(const double *__restrict__ A, // (N, N)
                            double *__restrict__ L,       // (N, N)
                            size_t N) {
  // 一个 block；所有线程共同执行外层循环。
  const auto tid = threadIdx.x;

  for (int j = 0; j < N; ++j) {
    if (tid == 0) {
      // 按对角公式计算 L[j, j]
      double sum = 0.0;
      for (int k = 0; k < j; k++) {
        double x = L[size_t(j) * N + k];
        sum += x * x;
      }
      L[size_t(j) * N + j] = sqrt(A[size_t(j) * N + j] - sum);
    }

    __syncthreads(); // 等待 L[j, j] 写完

    for (int i = j + 1 + tid; i < N; i += blockDim.x) {
      // 按非对角公式计算 L[i, j]
      // 每个线程内部，先用普通 for 循环完成 k 方向的求和
      double sum = 0.0;
      for (int k = 0; k < j; k++) {
        double x = L[size_t(i) * N + k];
        double y = L[size_t(j) * N + k];
        sum += x * y;
      }
      L[size_t(i) * N + j] =
          (A[size_t(i) * N + j] - sum) / L[size_t(j) * N + j];
    }

    __syncthreads(); // 等待整列写完，才能进入下一列
  }
}

void cholesky_decomposion(const double *__restrict__ A, // (N, N)
                          double *__restrict__ L,       // (N, N)
                          size_t N) {
  //
  cholesky_decomposion_kernel<<<1, 256>>>(A, L, N);
}

// solve A @ X = B; launch with one thread to preserve row dependencies.
template <bool transposeA = false>
__global__ void triangular_solve_kernel(const double *__restrict__ A, // (N, N)
                                        const double *__restrict__ B, // (N,)
                                        double *__restrict__ X,       // (N, 1)
                                        size_t N) {
  for (int step = threadIdx.x; step < N; step += blockDim.x) {
    int i = transposeA ? N - step - 1 : step;
    double sum = 0.0;
    if (transposeA) {
      for (int k = i + 1; k < N; k++) {
        sum += A[k * N + i] * X[k];
      }
    } else {
      for (int k = 0; k < i; k++) {
        sum += A[i * N + k] * X[k];
      }
    }
    X[i] = (B[i] - sum) / A[i * N + i];
  }
}

__global__ void convert_beta(const double *beta_double, float *beta, size_t N) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < N)
    beta[i] = float(beta_double[i]);
}

__device__ float sigmoid(float x) { return 1 / (1 + expf(-x)); }

__global__ void error_kernel(const float *__restrict__ X,    // (M, K)
                             const float *__restrict__ y,    // (M,)
                             const float *__restrict__ beta, // (K,)
                             double *__restrict__ error,     // (M,)
                             int M, int K) {
  const size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || k >= K)
    return;
  double sum = 0.0;
  for (size_t i = 0; i < K; i++) {
    sum += double(X[m * K + i]) * double(beta[i * 1 + k]);
  }
  error[k] = sigmoid(sum) - y[k];
}

void error(const float *__restrict__ X,    // (M, K)
           const float *__restrict__ y,    // (M,)
           const float *__restrict__ beta, // (K,)
           double *__restrict__ error,     // (M,)
           int M, int K) {
  // error[i] = sigmoid(X @ beta) - y[i]
  error_kernel<<<dim3(1, (M + 15) / 16), dim3(16, 16)>>>(X, y, beta, error, M,
                                                         K);
}

// X, y, beta are device pointers
extern "C" void solve(const float *X, // (n_samples, n_features)
                      const float *y, // (n_samples,)
                      float *beta,    // (n_features, 1)
                      int n_samples, int n_features) {
  constexpr auto lr = 0.01;

  double *error; // (n_samples,)
  cudaMalloc(&error, n_samples * sizeof(float));
  double *gradient; // (n_features,)
  cudaMalloc(&gradient, n_features * sizeof(double));

  // use GD
  while (true) {
    // error = sigmoid(X @ beta) - y
    error_kernel<<<(n_samples + 255) / 256, 256>>>(X, y, beta, error, n_samples,
                                                   n_features);
    // gradient = Xt @ error
    gemm<true>(X, error, gradient, n_samples, 1, n_features);

    // W = diag(pi * (1 - pi))
    // Hessian = Xt @ W @ X

    // solve: H @ Delta = gradient
    // beta -= Delta
  }

  cudaFree(error);
  cudaFree(gradient);

  if (0) { // beta = (X^T @ X)^-1 @ (X^T @ y)
    double *XtX, *Xty;
    cudaMalloc(&XtX, n_features * n_features * sizeof(double));
    cudaMalloc(&Xty, n_features * sizeof(double));
    gemm<true>(X, X, XtX, n_features, n_features, n_samples);
    gemm<true>(X, y, Xty, n_features, 1, n_samples);

    // solve: XtX @ beta = Xty
    // XtX = L @ Lt
    // L: 下三角矩阵
    double *L;
    cudaMalloc(&L, n_features * n_features * sizeof(double));
    cudaMemset(L, 0, n_features * n_features * sizeof(double));
    cholesky_decomposion(XtX, L, n_features);

    // solve: L @ z = Xty
    double *z;
    cudaMalloc(&z, n_features * sizeof(double));
    triangular_solve_kernel<<<1, 1>>>(L, Xty, z, n_features);

    // solve: Lt @ beta = z
    // Keep solved coefficients in FP64 while later rows depend on them.
    double *beta_double;
    cudaMalloc(&beta_double, n_features * sizeof(double));
    triangular_solve_kernel<true><<<1, 1>>>(L, z, beta_double, n_features);
    convert_beta<<<(n_features + 255) / 256, 256>>>(beta_double, beta,
                                                    n_features);

    cudaFree(beta_double);
    cudaFree(z);
    cudaFree(L);
    cudaFree(XtX);
    cudaFree(Xty);
  }
}
