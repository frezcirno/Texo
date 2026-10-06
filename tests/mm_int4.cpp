#include "test_utils.h"
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <string>

extern "C" void solve(const half*, const uint8_t*, const half*, half*, int, int,
                      int, int);

// LeetGPU contract: atol = rtol = 0.01 against an FP32 reference.
constexpr double ATOL = 0.01, RTOL = 0.01;

struct Inputs {
  int m, n, k, g;
  std::vector<half> x, scales;
  std::vector<uint8_t> w;
};

static Inputs make_inputs(int m, int n, int k, int g, unsigned seed,
                          bool zero_x = false) {
  Inputs in{m, n, k, g, {}, {}, {}};
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.f, 1.f);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_real_distribution<float> unit(0.f, 1.f);
  in.x.resize(size_t(m) * k);
  for (auto& v : in.x) v = __float2half_rn(zero_x ? 0.f : normal(rng));
  in.w.resize(size_t(n) * (k / 2));
  for (auto& v : in.w) v = uint8_t(byte(rng));
  in.scales.resize(size_t(n) * (k / g));
  for (auto& v : in.scales) v = __float2half_rn(unit(rng) * 0.1f + 0.01f);
  return in;
}

static double reference(const Inputs& in, int row, int col) {
  double dot = 0;
  const int groups = in.k / in.g;
  for (int i = 0; i < in.k; ++i) {
    const int byte = in.w[size_t(col) * (in.k / 2) + i / 2];
    const int q = (i % 2 == 0 ? byte >> 4 : byte & 15) - 8;
    dot += double(__half2float(in.x[size_t(row) * in.k + i])) * q *
           double(__half2float(in.scales[size_t(col) * groups + i / in.g]));
  }
  return dot;
}

// rows < 0 checks every row; otherwise that many rows spread over M.
static bool run_case(const Inputs& in, int rows = -1, int c_offset = 0) {
  constexpr size_t GUARD = 8;
  const uint16_t MARKER = 0x7bcd;
  test::DeviceArray<half> dx(in.x), ds(in.scales);
  test::DeviceArray<uint8_t> dw(in.w);
  const size_t count = size_t(in.m) * in.n, prefix = GUARD + c_offset;
  std::vector<uint16_t> initial(prefix + count + GUARD, MARKER);
  test::DeviceArray<uint16_t> dy(initial);

  std::vector<int> checked;
  if (rows < 0 || rows >= in.m) {
    for (int r = 0; r < in.m; ++r) checked.push_back(r);
  } else {
    for (int i = 0; i < rows; ++i)
      checked.push_back(int((int64_t(in.m - 1) * i) / std::max(1, rows - 1)));
  }

  bool passed = true;
  double max_error = 0;
  for (int call = 0; call < 2 && passed; ++call) {
    // NaN poison in the first call, a stale finite value in the second.
    std::fill(initial.begin() + prefix, initial.begin() + prefix + count,
              call == 0 ? uint16_t(0x7e00) : uint16_t(0x4400));
    dy.upload(initial);
    solve(dx.data(), dw.data(), ds.data(),
          reinterpret_cast<half*>(dy.data() + prefix), in.m, in.n, in.k, in.g);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto actual = dy.download();
    for (size_t i = 0; i < actual.size(); ++i) {
      if ((i < prefix || i >= prefix + count) && actual[i] != MARKER) {
        std::fprintf(stderr, "call=%d output guard overwritten at %zu\n",
                     call + 1, i);
        passed = false;
        break;
      }
    }
    for (int row : checked) {
      for (int col = 0; col < in.n && passed; ++col) {
        uint16_t bits = actual[prefix + size_t(row) * in.n + col];
        __half_raw raw;
        raw.x = bits;
        const half h(raw);
        const double got = __half2float(h), want = reference(in, row, col);
        const double error = std::abs(got - want);
        max_error = std::max(max_error, error);
        if (!std::isfinite(got) || error > ATOL + RTOL * std::abs(want)) {
          std::fprintf(stderr, "call=%d row=%d col=%d got=%.6g expected=%.6g\n",
                       call + 1, row, col, got, want);
          passed = false;
        }
      }
      if (!passed) break;
    }
  }
  passed &= dx.unchanged(in.x) & dw.unchanged(in.w) & ds.unchanged(in.scales);
  std::printf("%s M=%d N=%d K=%d G=%d rows=%zu offset=%d max_abs_error=%.3g\n",
              passed ? "PASS" : "FAIL", in.m, in.n, in.k, in.g, checked.size(),
              c_offset, max_error);
  return passed;
}

static bool run_example() {
  Inputs in{2, 4, 4, 2, {}, {}, {}};
  for (float v : {1.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 1.f})
    in.x.push_back(__float2half_rn(v));
  in.w = {153, 153, 170, 170, 119, 119, 136, 136};
  in.scales.assign(8, __float2half_rn(0.5f));
  return run_case(in);
}

static void bench(int m, int n, int k, int g, int iters) {
  Inputs in = make_inputs(m, n, k, g, 0);
  test::DeviceArray<half> dx(in.x), ds(in.scales), dy(size_t(m) * n);
  test::DeviceArray<uint8_t> dw(in.w);
  auto call = [&] { solve(dx.data(), dw.data(), ds.data(), dy.data(), m, n, k, g); };
  for (int i = 0; i < 3; ++i) call();
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> times;
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    call();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    times.push_back(ms);
  }
  std::sort(times.begin(), times.end());
  const float med = times[times.size() / 2];
  std::printf("BENCH M=%d N=%d K=%d G=%d median=%.4f ms min=%.4f ms %.1f TFLOPS\n",
              m, n, k, g, med, times[0], 2.0 * m * n * k / med / 1e9);
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--bench") {
    bench(4096, 4096, 4096, 128, argc > 2 ? std::atoi(argv[2]) : 50);
    return 0;
  }
  bool passed = run_example();
  // LeetGPU functional cases.
  const int official[][4] = {{2, 4, 4, 2},     {3, 5, 8, 4},      {16, 16, 32, 16},
                             {32, 64, 64, 32}, {64, 128, 128, 64}, {30, 50, 64, 32},
                             {100, 200, 128, 64}, {255, 100, 128, 64},
                             {128, 256, 512, 128}};
  passed &= run_case(make_inputs(1, 2, 4, 2, 1, true));
  unsigned seed = 2;
  for (auto& c : official) passed &= run_case(make_inputs(c[0], c[1], c[2], c[3], seed++));
  // Tensor core path edges: partial M/N/K tiles, odd N (scalar stores),
  // group 16 vs 32 steps, groups spanning K tiles, K tail inside a tile.
  const int extra[][4] = {{1, 1, 32, 32},     {129, 131, 96, 32},  {7, 257, 160, 16},
                          {200, 129, 192, 64}, {131, 64, 384, 128}, {65, 300, 1056, 32},
                          {17, 33, 48, 16},    {5, 9, 30, 2},       {9, 7, 40, 8},
                          {300, 200, 1024, 16}, {1, 4096, 4096, 128}};
  for (auto& c : extra) passed &= run_case(make_inputs(c[0], c[1], c[2], c[3], seed++));
  passed &= run_case(make_inputs(130, 66, 256, 128, seed++), -1, 1);  // odd y offset
  {
    // Zero scales: first, middle and last groups, and an all-zero column.
    Inputs in = make_inputs(70, 136, 512, 32, seed++);
    const int groups = in.k / in.g;
    for (int n = 0; n < in.n; ++n)
      for (int gi = 0; gi < groups; ++gi)
        if (n == 5 || (n % 3 == 0 && (gi == 0 || gi == groups / 2 || gi == groups - 1)))
          in.scales[size_t(n) * groups + gi] = __float2half_rn(0.f);
    passed &= run_case(in);
  }
  if (argc > 1 && std::string(argv[1]) == "--quick") return test::finish(passed);
  // Performance shape, checked on sampled rows.
  passed &= run_case(make_inputs(4096, 4096, 4096, 128, 0), 24);
  passed &= run_case(make_inputs(8192, 520, 8192, 32, 5), 12);
  return test::finish(passed);
}
