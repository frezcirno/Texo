#include <cmath>
#include <cuda_runtime.h>

template <typename T> __device__ T sigmoid(T x) { return 1 / (1 + exp(-x)); }

template <typename T> __device__ inline T warp_sum(T val) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, off);
  }
  return val;
}

template <typename T1, typename T2, typename Tout>
__global__ void mv_sigmoid_kernel(const T1 *__restrict__ A, // (M, N)
                                  const T2 *__restrict__ B, // (N,)
                                  Tout *__restrict__ C,     // (M,)
                                  size_t M, size_t N) {
  // C = sigmoid(A @ B)
  const size_t lane = threadIdx.x & 31;
  const size_t warp = threadIdx.x >> 5;
  const size_t warps_per_block = blockDim.x / 32;
  const size_t row = blockIdx.x * warps_per_block + warp;

  // 同一个 warp 的 row 相同，因此整个 warp 一起退出
  if (row >= M) {
    return;
  }

  Tout sum = 0.0f;
  for (int i = lane; i < N; i += 32) {
    sum += Tout(A[row * N + i]) * Tout(B[i]);
  }

  sum = warp_sum(sum);

  if (lane == 0) {
    C[row] = sigmoid(sum);
  }
}

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

template <bool transposeA = false, typename T1, typename T2, typename Tout>
void gemm_fn(const T1 *__restrict__ A, // (M, K)
             const T2 *__restrict__ B, // (K, N)
             Tout *__restrict__ C,     // (M, N)
             size_t M, size_t N, size_t K) {
  gemm_kernel<transposeA, T1, T2, Tout>
      <<<dim3((N + 15) / 16, (M + 15) / 16), dim3(16, 16)>>>(A, B, C, M, N, K);
}

template <bool transposeA = false, typename T1, typename T2, typename T3,
          typename Tout>
__global__ void gradient_kernel(const T1 *__restrict__ A,  // (M, K)
                                const T2 *__restrict__ B1, // (K,)
                                const T3 *__restrict__ B2, // (K,)
                                Tout *__restrict__ C,      // (M,)
                                size_t M, size_t N, size_t K) {
  const size_t cx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t cy = blockIdx.y * blockDim.y + threadIdx.y;
  if (cy >= M || cx >= N)
    return;
  Tout sum = 0.0;
  for (size_t i = 0; i < K; i++) {
    sum += Tout(transposeA ? A[i * M + cy] : A[cy * K + i]) *
           Tout(B1[i * N + cx] - B2[i * N + cx]);
  }
  C[cy * N + cx] = sum;
}

template <bool transposeA = false, typename T1, typename T2, typename T3,
          typename Tout>
void gradient_fn(const T1 *__restrict__ A,  // (M, K)
                 const T2 *__restrict__ B1, // (K,)
                 const T3 *__restrict__ B2, // (K,)
                 Tout *__restrict__ C,      // (M,)
                 size_t M, size_t N, size_t K) {
  gradient_kernel<transposeA>
      <<<dim3((N + 15) / 16, (M + 15) / 16), dim3(16, 16)>>>(A, B1, B2, C, M, N,
                                                             K);
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

template <typename Ta, typename Tb, typename Tc>
__global__ void subtract_kernel(Ta *__restrict__ A,       // (N,)
                                const Tb *__restrict__ B, // (N,)
                                const Tc alpha, const size_t N) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
    return;
  auto x = A[tid];
  auto y = B[tid];
  A[tid] = x - alpha * y;
}

template <typename T1, typename T2, typename T3>
__global__ void
hessian_kernel(const T1 *X,                       // (n_samples, n_features)
               const T2 *__restrict__ prediction, // (n_samples,)
               T3 *__restrict__ hessian,          // (n_features, n_features)
               const size_t n_samples, const size_t n_features) {
  const size_t cx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t cy = blockIdx.y * blockDim.y + threadIdx.y;
  if (cy >= n_features || cx >= n_features)
    return;
  T3 sum = 0.0;
  for (size_t i = 0; i < n_samples; i++) {
    // A[cy][i] = Xt[cy][i] * D[i] = X[i][cy] * D[i]
    auto pi = prediction[i];
    T3 a = X[i * n_features + cy] * pi * (1 - pi);
    sum += a * T3(X[i * n_features + cx]);
  }
  hessian[cy * n_features + cx] = sum;
}

template <typename T1, typename T2, typename T3>
void hessian_fn(const T1 *X,                       // (n_samples, n_features)
                const T2 *__restrict__ prediction, // (n_samples,)
                T3 *__restrict__ hessian,          // (n_features, n_features)
                const size_t n_samples, const size_t n_features) {
  // M[i][j] = Xt[i][j] * D[j]
  // hessian = M @ X
  hessian_kernel<<<dim3((n_features + 15) / 16, (n_features + 15) / 16),
                   dim3(16, 16)>>>(X, prediction, hessian, n_samples,
                                   n_features);
}

// X, y, beta are device pointers
extern "C" void solve(const float *X, // (n_samples, n_features)
                      const float *y, // (n_samples,)
                      float *beta,    // (n_features,)
                      int n_samples, int n_features) {
  constexpr auto lr = 0.01;

  double *prediction; // (n_samples,)
  cudaMalloc(&prediction, n_samples * sizeof(*prediction));
  double *gradient; // (n_features,)
  cudaMalloc(&gradient, n_features * sizeof(*gradient));
  double *hessian; // (n_features, n_features)
  cudaMalloc(&hessian, n_features * n_features * sizeof(*hessian));
  double *L; // (n_features, n_features)
  cudaMalloc(&L, n_features * n_features * sizeof(double));
  double *z;
  cudaMalloc(&z, n_features * sizeof(double));
  double *delta; // (n_features,)
  cudaMalloc(&delta, n_features * sizeof(*delta));

  cudaMemset(beta, 0, n_features * sizeof(float));

  // use GD
  for (int step = 0; step < 100; step++) {
    // prediction = sigmoid(X @ beta)
    mv_sigmoid_kernel<<<(n_samples + 7) / 8, 256>>>(X, beta, prediction,
                                                    n_samples, n_features);

    // gradient = Xt @ (prediction - y)
    gradient_fn<true>(X, prediction, y, gradient, n_features, 1, n_samples);

    // W = diag(pi * (1 - pi))
    // hessian = Xt @ W @ X
    hessian_fn(X, prediction, hessian, n_samples, n_features);

    // solve: hessian @ delta = gradient
    // solve: L @ Lt = hessian
    cholesky_decomposion(hessian, L, n_features);
    // solve: L @ (Lt @ delta) = gradient
    triangular_solve_kernel<<<1, 1>>>(hessian, gradient, z, n_features);
    // solve: Lt @ delta = z
    triangular_solve_kernel<true><<<1, 1>>>(L, z, delta, n_features);

    // beta -= lr * delta
    subtract_kernel<<<(n_features + 255) / 256, 256>>>(beta, delta, lr,
                                                       n_features);
  }

  cudaFree(prediction);
  cudaFree(gradient);
  cudaFree(hessian);
  cudaFree(L);
  cudaFree(z);
  cudaFree(delta);
}
