#include "test_utils.h"
#include <cstdint>
#include <utility>

extern "C" void solve(const int8_t*, const int8_t*, int8_t*, int, int, int,
                      float, float, float, int, int, int);

struct Quantization {
  float a, b, c;
  int za, zb, zc;
};

static double round_to_even(double value) {
  double lower = std::floor(value);
  double fraction = value - lower;
  return fraction < 0.5 || (fraction == 0.5 && std::fmod(lower, 2.0) == 0)
             ? lower : lower + 1;
}

static std::vector<int8_t> random_input(size_t size, unsigned seed, int lo = -128,
                                       int hi = 127) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(lo, hi);
  std::vector<int8_t> values(size);
  for (auto& value : values) value = int8_t(dist(rng));
  return values;
}

static bool run_case(const char* name, int m, int n, int k,
                     std::vector<int8_t> a, std::vector<int8_t> b,
                     Quantization q, int a_offset = 0, int b_offset = 0,
                     int c_offset = 0, const std::vector<int8_t>& known = {}) {
  // Independent integer dot products, followed by double scaling and round
  // halfway to even. Binary scales and bounded sums make the main
  // cases exact in FP32 too; decimal-scale cases stay away from half ties.
  std::vector<int8_t> reference = known;
  if (reference.empty()) {
    reference.resize(size_t(m) * n);
    for (int row = 0; row < m; ++row) {
      for (int col = 0; col < n; ++col) {
        int64_t dot = 0;
        for (int inner = 0; inner < k; ++inner)
          dot += int64_t(int(a[size_t(row) * k + inner]) - q.za) *
                 (int(b[size_t(inner) * n + col]) - q.zb);
        double scaled = double(dot) * double(q.a) * double(q.b) / double(q.c);
        double value = round_to_even(scaled) + q.zc;
        reference[size_t(row) * n + col] = int8_t(std::max(-128.0, std::min(127.0, value)));
      }
    }
  } else if (reference.size() != size_t(m) * n) {
    return false;
  }
  // Known platform/analytic answers bypass the generic double formula, also
  // avoiding a cubic CPU reference for the optional full-size regression.

  constexpr size_t GUARD = 16;
  constexpr int8_t MARKER = 91;
  // Leave input tails unpadded for memcheck, including the unaligned cases.
  a.insert(a.begin(), a_offset, MARKER);
  b.insert(b.begin(), b_offset, MARKER);
  if (a.empty()) a.push_back(MARKER);
  if (b.empty()) b.push_back(MARKER);
  test::DeviceArray<int8_t> da(a), db(b);
  const size_t prefix = GUARD + c_offset;
  std::vector<int8_t> initial(prefix + reference.size() + GUARD, MARKER);
  test::DeviceArray<int8_t> dc(initial);
  bool passed = true;
  for (int call = 0; call < 2; ++call) {
    // Opposite endpoint poisons expose missing stores and accidental C += ...,
    // even when the correct quantized answer is itself an endpoint.
    std::fill(initial.begin() + prefix, initial.begin() + prefix + reference.size(),
              call == 0 ? int8_t(127) : int8_t(-128));
    dc.upload(initial);
    solve(k == 0 ? nullptr : da.data() + a_offset,
          k == 0 ? nullptr : db.data() + b_offset, dc.data() + prefix,
          m, n, k, q.a, q.b, q.c, q.za, q.zb, q.zc);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto actual = dc.download();
    for (size_t i = 0; i < actual.size(); ++i) {
      if ((i < prefix || i >= prefix + reference.size()) && actual[i] != MARKER) {
        std::fprintf(stderr, "%s call=%d output guard overwritten at %zu\n",
                     name, call + 1, i);
        passed = false;
        break;
      }
    }
    for (size_t i = 0; i < reference.size(); ++i) {
      if (actual[prefix + i] != reference[i]) {
        std::fprintf(stderr, "%s call=%d row=%zu col=%zu got=%d expected=%d\n",
                     name, call + 1, i / n, i % n,
                     int(actual[prefix + i]), int(reference[i]));
        passed = false;
        break;
      }
    }
    passed &= da.unchanged(a);
    passed &= db.unchanged(b);
  }
  std::printf("%s %-22s M=%d N=%d K=%d scales=%.9g/%.9g/%.9g zp=%d/%d/%d offsets=%d/%d/%d\n",
              passed ? "PASS" : "FAIL", name, m, n, k,
              q.a, q.b, q.c, q.za, q.zb, q.zc, a_offset, b_offset, c_offset);
  return passed;
}

static bool run_random(const char* name, int m, int n, int k, Quantization q,
                       int a_offset = 0, int b_offset = 0, int c_offset = 0) {
  return run_case(name, m, n, k, random_input(size_t(m) * k, 42),
                  random_input(size_t(k) * n, 123), q, a_offset, b_offset, c_offset);
}

static bool run_large() {
  const int m = 8192, n = 4096, k = 2048, pairs = (k - 2) / 2;
  auto a = random_input(size_t(m) * k, 77, -127, 127);
  auto b = random_input(size_t(k) * n, 99, -127, 127);
  std::vector<int8_t> expected(size_t(m) * n);
  for (int row = 0; row < m; ++row) {
    for (int inner = 0; inner < pairs; ++inner)
      a[size_t(row) * k + pairs + inner] = a[size_t(row) * k + inner];
    const int sign = row % 2 ? -1 : 1;
    a[size_t(row) * k + k - 2] = int8_t(sign);
    a[size_t(row) * k + k - 1] = 0;
    for (int col = 0; col < n; ++col)
      expected[size_t(row) * n + col] = int8_t(sign * (col % 255 - 127));
  }
  for (int inner = 0; inner < pairs; ++inner)
    for (int col = 0; col < n; ++col)
      b[size_t(pairs + inner) * n + col] = -b[size_t(inner) * n + col];
  for (int col = 0; col < n; ++col) {
    b[size_t(k - 2) * n + col] = int8_t(col % 255 - 127);
    b[size_t(k - 1) * n + col] = 0;
  }
  std::puts("Checking 8192x4096x2048: paired products cancel; every output has an analytic reference.");
  std::fflush(stdout);
  return run_case("large-cancellation", m, n, k, std::move(a), std::move(b),
                  {0.1f, 0.1f, 0.01f, 0, 0, 0}, 0, 0, 0, expected);
}

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--large") == 0)
    return test::finish(run_large());
  if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [--large]\n", argv[0]);
    return EXIT_FAILURE;
  }
  bool passed = true;
  // Net scale 1/256 leaves many unsaturated outputs across the full INT8 range.
  const Quantization normal = {0.125f, 0.25f, 8.0f, 0, 0, 0};
  struct Shape { int m, n, k; };
  const Shape shapes[] = {
    {1, 1, 1}, {1, 17, 7}, {17, 1, 7}, {2, 3, 5},
    {15, 15, 15}, {16, 16, 16}, {17, 17, 17},
    {16, 15, 7}, {15, 16, 7}, {16, 17, 7}, {17, 16, 7},
    {31, 32, 9}, {32, 31, 9}, {32, 32, 32}, {33, 33, 33},
    {3, 5, 257}, {3, 5, 1025}, {33, 65, 1}, {65, 67, 31}
  };
  for (auto shape : shapes)
    passed &= run_random("random-boundary", shape.m, shape.n, shape.k, normal);

  for (Quantization q : {Quantization{0.125f, 0.25f, 8, 7, 0, 0},
                         Quantization{0.125f, 0.25f, 8, 0, -11, 0},
                         Quantization{0.125f, 0.25f, 8, 0, 0, 19},
                         Quantization{0.125f, 0.25f, 8, 7, -11, 19},
                         Quantization{0.125f, 0.25f, 8, -128, 127, -128},
                         Quantization{0.125f, 0.25f, 8, 127, -128, 127}})
    passed &= run_random("zero-points", 17, 19, 7, q);

  const Quantization shifted = {0.25f, 0.5f, 0.125f, 7, -11, 19};
  passed &= run_case("quantized-zero-A", 17, 19, 7, std::vector<int8_t>(17 * 7, 7),
                     random_input(7 * 19, 123), shifted);
  passed &= run_case("quantized-zero-B", 17, 19, 7, random_input(17 * 7, 42),
                     std::vector<int8_t>(7 * 19, -11), shifted);
  std::vector<int8_t> identity_a(17 * 17, shifted.za), identity_b(19 * 19, shifted.zb);
  for (int i = 0; i < 17; ++i) identity_a[i * 17 + i] = int8_t(shifted.za + 1);
  for (int i = 0; i < 19; ++i) identity_b[i * 19 + i] = int8_t(shifted.zb + 1);
  passed &= run_case("left-identity", 17, 19, 17, identity_a,
                     random_input(17 * 19, 123, -32, 31), shifted);
  passed &= run_case("right-identity", 17, 19, 19,
                     random_input(17 * 19, 42, -32, 31), identity_b, shifted);
  std::vector<int8_t> tail(17 * 257, shifted.za);
  for (int row = 0; row < 17; ++row) tail[row * 257 + 256] = int8_t(row + shifted.za + 1);
  passed &= run_case("last-K-only", 17, 19, 257, tail,
                     random_input(257 * 19, 123, -16, 15), shifted);

  const std::vector<int8_t> odd = {-5, -3, -1, 0, 1, 3, 5};
  for (int zc : {0, 7, -7})
    passed &= run_case("half-to-even", 7, 1, 1, odd, {1}, {0.5f, 1, 1, 0, 0, zc});
  for (float scale : {std::nextafter(0.5f, 0.0f), std::nextafter(0.5f, 1.0f)})
    passed &= run_case("adjacent-to-half", 2, 1, 1, {-1, 1}, {1}, {scale, 1, 1, 0, 0, 0});
  // Exercise each scale independently; all three produce half-integer ties.
  passed &= run_case("scale-B", 7, 1, 1, odd, {1}, {1, 0.5f, 1, 0, 0, 0});
  passed &= run_case("scale-C", 7, 1, 1, odd, {1}, {1, 1, 2, 0, 0, 0});
  passed &= run_case("decimal-scales", 17, 19, 7, random_input(17 * 7, 42, -8, 7),
                     random_input(7 * 19, 123, -8, 7), {0.1f, 0.2f, 0.3f, 1, -2, 3});

  passed &= run_case("saturation", 12, 1, 1,
                     {-128, -127, -65, -64, -63, 0, 62, 63, 64, 65, 126, 127},
                     {2}, {1, 1, 1, 0, 0, 0});
  passed &= run_case("signed-extremes", 2, 2, 1, {127, -128}, {127, -128}, {1, 1, 1, 0, 0, 0});
  for (int zc : {-128, 127})
    passed &= run_case("shift-then-saturate", 5, 1, 1, {-127, -1, 0, 1, 127},
                       {1}, {1, 1, 1, 0, 0, zc});
  const Quantization unaligned = {0.125f, 0.25f, 8, 7, -11, 19};
  passed &= run_random("unaligned-A", 17, 19, 7, unaligned, 1, 0, 0);
  passed &= run_random("unaligned-B", 17, 19, 7, unaligned, 0, 1, 0);
  passed &= run_random("unaligned-C", 17, 19, 7, unaligned, 0, 0, 1);
  passed &= run_random("unaligned-all", 17, 19, 7, unaligned, 1, 3, 7);
  for (int zc : {-128, -7, 0, 127})
    passed &= run_random("K-zero", 17, 19, 0, {0.125f, 0.25f, 8, 7, -11, zc}, 0, 0, 1);
  passed &= run_case("leetgpu-58-not-59", 3, 5, 2,
                     {29, -32, -16, 35, -35, -38},
                     {36, 50, 42, 8, -30, 47, -2, -35, 7, -30},
                     {0.05f, 0.1f, 0.01f, 0, 0, 0}, 0, 0, 0,
                     {-128, 127, 127, 4, 45, 127, -128, -128, 58, -128,
                      -128, -128, -70, -128, 127});
  passed &= run_case("leetgpu-negative-tie", 3, 5, 2,
                     {-29, 32, 16, -35, 35, 38},
                     {36, 50, 42, 8, -30, 47, -2, -35, 7, -30},
                     {0.05f, 0.1f, 0.01f, 0, 0, 0}, 0, 0, 0,
                     {127, -128, -128, -4, -45, -128, 127, 127, -58, 127,
                      127, 127, 70, 127, -128});
  // Matching integer products cancel exactly, leaving a small known residual.
  // Scaling every term before a long FP32 reduction can lose an output unit.
  for (bool random : {false, true}) {
    const int m = 17, n = 19, k = 2048, pairs = (k - 2) / 2;
    auto a = random ? random_input(size_t(m) * k, 77, -127, 127)
                    : std::vector<int8_t>(size_t(m) * k, 127);
    auto b = random ? random_input(size_t(k) * n, 99, -127, 127)
                    : std::vector<int8_t>(size_t(k) * n, 127);
    for (int row = 0; row < m; ++row) {
      for (int inner = 0; inner < pairs; ++inner)
        a[size_t(row) * k + pairs + inner] = a[size_t(row) * k + inner];
      a[size_t(row) * k + k - 2] = row % 2 ? -1 : 1;
      a[size_t(row) * k + k - 1] = 0;
    }
    for (int inner = 0; inner < pairs; ++inner)
      for (int col = 0; col < n; ++col)
        b[size_t(pairs + inner) * n + col] = -b[size_t(inner) * n + col];
    for (int col = 0; col < n; ++col) {
      b[size_t(k - 2) * n + col] = int8_t(col - n / 2);
      b[size_t(k - 1) * n + col] = 0;
    }
    passed &= run_case(random ? "random-cancellation" : "ordered-cancellation",
                       m, n, k, a, b, {0.1f, 0.1f, 0.01f, 0, 0, 0});
  }
  // 33026 * 255 * 255 exceeds INT32_MAX; the small scaled result is unsaturated.
  passed &= run_case("wide-positive-dot", 1, 1, 33026,
                     std::vector<int8_t>(33026, 127), std::vector<int8_t>(33026, 127),
                     {1, 1, 33554432.0f, -128, -128, 0});
  passed &= run_case("wide-negative-dot", 1, 1, 33026,
                     std::vector<int8_t>(33026, -128), std::vector<int8_t>(33026, 127),
                     {1, 1, 33554432.0f, 127, -128, 0});
  return test::finish(passed);
}
