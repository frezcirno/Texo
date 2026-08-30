#include "gemm.h"

int dgemm_cpu(int m, int n, int k,
              double *A, // m x k
              double *B, // k x n
              double *C  // m x n
) {

  // 进行矩阵乘法计算
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      C[i * n + j] = 0; // 初始化 C 的元素
      for (int l = 0; l < k; l++) {
        C[i * n + j] += A[i * k + l] * B[l * n + j]; // 矩阵乘法计算
      }
    }
  }

  return 0; // 返回成功状态
}
