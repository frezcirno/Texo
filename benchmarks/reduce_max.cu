// Standalone maximum-reduction comparison.
// 对比两种 CUDA max reduction:
//   Kernel 1: pure tree reduction — 每个 thread 处理 1 个元素，gridDim = ceil(N / BLOCK_SIZE)
//   Kernel 2: serial-then-tree (grid-stride loop) — 每个 thread 在寄存器里串行 reduce 多个元素，
//             然后做 block-level tree reduce，gridDim 固定为 SM 数的小倍数
//
// Build from the repository root: make build/sm_80/reduce_max_compare
//        (根据你的卡调 -arch: sm_80=A100, sm_89=4090, sm_90=H100)
// Run: build/sm_80/reduce_max_compare [N]; default N = 1 << 26 (~256 MB)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cfloat>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); } } while (0)

constexpr int BLOCK_SIZE = 256;          // 两个 kernel 共用
constexpr int WARP_SIZE  = 32;

// -----------------------------------------------------------------------------
// 共用工具:  warp-level max via shuffle
// -----------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    }
    return v;
}

// atomicMax for float via CAS on the int bit-pattern (assumes non-negative or
// uses ordered int bit trick; here we just CAS until we win for simplicity)
__device__ __forceinline__ void atomic_max_float(float* addr, float val) {
    int* iaddr = reinterpret_cast<int*>(addr);
    int old = __float_as_int(*addr);
    int assumed;
    do {
        assumed = old;
        float cur = __int_as_float(assumed);
        if (val <= cur) break;
        old = atomicCAS(iaddr, assumed, __float_as_int(val));
    } while (assumed != old);
}

// -----------------------------------------------------------------------------
// Kernel 1: pure tree reduction.  每个 thread 只读 1 个元素。
//   gridDim = ceil(N / BLOCK_SIZE)
// -----------------------------------------------------------------------------
__global__ void reduce_max_k1(const float* __restrict__ in, float* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (idx < N) ? in[idx] : -FLT_MAX;

    // Warp-level reduce
    v = warp_reduce_max(v);

    // 每个 warp 的 leader 写入 shared memory
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int wid  = threadIdx.x / WARP_SIZE;
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    // 第一个 warp 再 reduce 一次得到 block max
    if (wid == 0) {
        int n_warps = BLOCK_SIZE / WARP_SIZE;
        v = (lane < n_warps) ? smem[lane] : -FLT_MAX;
        v = warp_reduce_max(v);
        if (lane == 0) atomic_max_float(out, v);
    }
}

// -----------------------------------------------------------------------------
// Kernel 2: serial-then-tree (grid-stride loop).
//   每个 thread 在自己的寄存器里串行 reduce 多个元素，再做 block-level tree。
//   gridDim 由 caller 控制 (通常 = SM 数的小倍数，与 N 解耦)。
// -----------------------------------------------------------------------------
__global__ void reduce_max_k2(const float* __restrict__ in, float* __restrict__ out, int N) {
    float v = -FLT_MAX;

    // 外层: serial reduction in register, grid-stride loop
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride) {
        v = fmaxf(v, in[i]);
    }

    // 内层: 和 Kernel 1 完全一样的 block-level tree
    v = warp_reduce_max(v);
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int wid  = threadIdx.x / WARP_SIZE;
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    if (wid == 0) {
        int n_warps = BLOCK_SIZE / WARP_SIZE;
        v = (lane < n_warps) ? smem[lane] : -FLT_MAX;
        v = warp_reduce_max(v);
        if (lane == 0) atomic_max_float(out, v);
    }
}

// -----------------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------------
float cpu_max(const float* a, int N) {
    float m = -FLT_MAX;
    for (int i = 0; i < N; ++i) m = std::fmax(m, a[i]);
    return m;
}

struct BenchResult {
    float ms_avg;
    float ms_min;
    float result;
    float gbps;
};

template <typename Reset, typename Launcher>
BenchResult bench(Reset reset, Launcher launch, int N, int warmup, int iters) {
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        reset();
        launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Time
    float total_ms = 0.f, min_ms = 1e30f;
    for (int i = 0; i < iters; ++i) {
        // Output initialization is excluded from kernel timing.
        reset();
        CUDA_CHECK(cudaEventRecord(ev_start));
        launch();
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        total_ms += ms;
        min_ms = std::fmin(min_ms, ms);
    }

    BenchResult r;
    r.ms_avg = total_ms / iters;
    r.ms_min = min_ms;
    r.gbps = (double(N) * sizeof(float)) / (r.ms_min * 1e-3) / 1e9;
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    return r;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : (1 << 26);  // 默认 ~64M floats
    if (argc > 2 || N <= 0) {
        fprintf(stderr, "Usage: %s [positive-N]\n", argv[0]);
        return EXIT_FAILURE;
    }
    printf("N = %d  (%.2f MB)\n", N, N * sizeof(float) / 1024.0 / 1024.0);

    // 查询 SM 数，给 Kernel 2 的 gridDim 用
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  SMs=%d\n", prop.name, prop.multiProcessorCount);
    printf("BLOCK_SIZE = %d\n\n", BLOCK_SIZE);

    // 数据
    std::vector<float> h(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1000.f, 1000.f);
    for (int i = 0; i < N; ++i) h[i] = dist(rng);
    // 塞一个已知最大值方便校验
    int max_idx = N / 2;
    h[max_idx] = 9999.f;
    float ref = cpu_max(h.data(), N);
    printf("CPU reference max = %.6f\n\n", ref);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    const int warmup = 5, iters = 50;
    float neg_inf = -FLT_MAX;

    auto reset_out = [&] {
        CUDA_CHECK(cudaMemcpy(d_out, &neg_inf, sizeof(float), cudaMemcpyHostToDevice));
    };
    auto read_out = [&] {
        float v;
        CUDA_CHECK(cudaMemcpy(&v, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        return v;
    };

    bool passed = true;

    // -------------------- Kernel 1 --------------------
    {
        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        printf("Kernel 1 (pure tree, 1 elem/thread):  grid=%d  block=%d\n", grid, BLOCK_SIZE);
        auto launch = [&] {
            reduce_max_k1<<<grid, BLOCK_SIZE>>>(d_in, d_out, N);
        };
        auto r = bench(reset_out, launch, N, warmup, iters);
        float res = read_out();
        passed &= (res == ref);
        printf("  avg=%.3f ms  min=%.3f ms  BW=%.1f GB/s  result=%.6f  %s\n\n",
               r.ms_avg, r.ms_min, r.gbps, res, (res == ref ? "OK" : "MISMATCH"));
    }

    // -------------------- Kernel 2, 几种 gridDim --------------------
    // gridDim 是关键调参旋钮: SM 数的几倍
    int sm = prop.multiProcessorCount;
    int grids_k2[] = { sm, sm * 2, sm * 4, sm * 8, sm * 16, sm * 32 };
    for (int grid : grids_k2) {
        int elems_per_thread = (N + grid * BLOCK_SIZE - 1) / (grid * BLOCK_SIZE);
        printf("Kernel 2 (serial+tree, grid-stride):  grid=%d (%dx SM)  ~%d elems/thread\n",
               grid, grid / sm, elems_per_thread);
        auto launch = [&] {
            reduce_max_k2<<<grid, BLOCK_SIZE>>>(d_in, d_out, N);
        };
        auto r = bench(reset_out, launch, N, warmup, iters);
        float res = read_out();
        passed &= (res == ref);
        printf("  avg=%.3f ms  min=%.3f ms  BW=%.1f GB/s  result=%.6f  %s\n\n",
               r.ms_avg, r.ms_min, r.gbps, res, (res == ref ? "OK" : "MISMATCH"));
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
