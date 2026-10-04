#include <cudnn.h>

#include <cstdio>

namespace {
cudnnHandle_t handle = nullptr;
cudnnTensorDescriptor_t descriptor = nullptr;
int configured_size = 0;

void cleanup() {
  if (descriptor != nullptr) {
    cudnnDestroyTensorDescriptor(descriptor);
    descriptor = nullptr;
  }
  if (handle != nullptr) {
    cudnnDestroy(handle);
    handle = nullptr;
  }
  configured_size = 0;
}
} // namespace

extern "C" bool softmax_cudnn_setup(int N) {
  cleanup();
  if (N <= 0) {
    return false;
  }

  cudnnStatus_t status = cudnnCreate(&handle);
  if (status == CUDNN_STATUS_SUCCESS) {
    status = cudnnCreateTensorDescriptor(&descriptor);
  }
  if (status == CUDNN_STATUS_SUCCESS) {
    status = cudnnSetTensor4dDescriptor(descriptor, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, 1, 1, 1, N);
  }
  if (status != CUDNN_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuDNN softmax setup failed: %s\n",
                 cudnnGetErrorString(status));
    cleanup();
    return false;
  }

  configured_size = N;
  return true;
}

extern "C" void softmax_cudnn(const float *input, float *output, int N) {
  if (N != configured_size || handle == nullptr || descriptor == nullptr) {
    std::fprintf(stderr, "cuDNN softmax called without matching setup\n");
    return;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const cudnnStatus_t status = cudnnSoftmaxForward(
      handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
      descriptor, input, &beta, descriptor, output);
  if (status != CUDNN_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuDNN softmax failed: %s\n",
                 cudnnGetErrorString(status));
  }
}

extern "C" void softmax_cudnn_cleanup() { cleanup(); }
