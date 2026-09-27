#include "test_utils.h"
#include <limits>
#include <utility>

extern "C" void solve(const float*, float*, int, int);

static std::vector<double> reference_power(const std::vector<float>& input,
                                          int n, int p) {
  const size_t count = size_t(n) * n;
  std::vector<double> result(count, 0.0);
  for (int i = 0; i < n; ++i) result[size_t(i) * n + i] = 1.0;
  // Deliberately use P successive CPU double products, independently of the
  // recursive squaring/odd-power algorithm in the CUDA implementation.
  for (int step = 0; step < p; ++step) {
    std::vector<double> next(count, 0.0);
    for (int row = 0; row < n; ++row)
      for (int k = 0; k < n; ++k) {
        const double value = result[size_t(row) * n + k];
        for (int col = 0; col < n; ++col)
          next[size_t(row) * n + col] += value * double(input[size_t(k) * n + col]);
      }
    result.swap(next);
  }
  return result;
}

static bool run_case(const char* name, int n, int p, std::vector<float> input,
                     std::vector<double> expected, bool exact,
                     int input_offset, int output_offset, double rtol) {
  constexpr size_t GUARD = 4;
  constexpr float MARKER = 1234567.0f;
  constexpr double ATOL = 1e-5;
  const size_t count = size_t(n) * n;
  const size_t prefix = GUARD + output_offset;
  if (expected.empty()) expected = reference_power(input, n, p);
  // No input suffix padding, so memcheck can catch reads beyond N*N.
  test::DeviceArray<float> device_input(count + input_offset);
  test::DeviceArray<float> device_output(prefix + count + GUARD);
  bool passed = true;
  double max_error = 0;
  for (int call = 0; call < 2; ++call) {
    if (call == 1) {
      // (-A^T)^P = (-1)^P (A^P)^T. Changing orientation also changes the
      // even-power result of nonsymmetric matrices, catching stale outputs.
      for (int row = 0; row < n; ++row)
        for (int col = row + 1; col < n; ++col) {
          std::swap(input[size_t(row) * n + col], input[size_t(col) * n + row]);
          std::swap(expected[size_t(row) * n + col], expected[size_t(col) * n + row]);
        }
      for (float& value : input) value = -value;
      if (p % 2)
        for (double& value : expected) value = -value;
    }
    std::vector<float> stored_input(count + input_offset, MARKER);
    std::copy(input.begin(), input.end(), stored_input.begin() + input_offset);
    device_input.upload(stored_input);
    std::vector<float> initial(prefix + count + GUARD, MARKER);
    std::fill(initial.begin() + prefix, initial.begin() + prefix + count,
              call == 0 ? std::numeric_limits<float>::quiet_NaN() : -4321.5f);
    device_output.upload(initial);

    solve(device_input.data() + input_offset, device_output.data() + prefix, n, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto actual = device_output.download();
    for (size_t i = 0; i < actual.size(); ++i) {
      if ((i < prefix || i >= prefix + count) && actual[i] != MARKER) {
        std::fprintf(stderr, "%s call=%d output guard overwritten at %zu\n",
                     name, call + 1, i);
        passed = false;
        break;
      }
    }
    bool reported_mismatch = false;
    for (size_t i = 0; i < count; ++i) {
      const double got = actual[prefix + i];
      const double error = std::abs(got - expected[i]);
      max_error = std::max(max_error, error);
      const double tolerance = exact ? 0.0 : ATOL + rtol * std::abs(expected[i]);
      if (!std::isfinite(expected[i]) || !std::isfinite(got) || error > tolerance) {
        if (!reported_mismatch)
          std::fprintf(stderr,
                       "%s call=%d N=%d P=%d index=%zu got=%.12g expected=%.12g\n",
                       name, call + 1, n, p, i, got, expected[i]);
        reported_mismatch = true;
        passed = false;
      }
    }
    // P=1 is a copy, including the sign bits of zeros.
    if (p == 1 && std::memcmp(actual.data() + prefix, input.data(),
                              count * sizeof(float)) != 0) {
      std::fprintf(stderr, "%s call=%d P=1 did not preserve input bits\n",
                   name, call + 1);
      passed = false;
    }
    passed &= device_input.unchanged(stored_input);
  }
  std::printf("%s %-22s N=%d P=%d offsets=%d/%d max_abs_error=%.3g\n",
              passed ? "PASS" : "FAIL", name, n, p, input_offset, output_offset,
              max_error);
  return passed;
}

static std::vector<float> dense_input(int n) {
  auto input = test::random_values(size_t(n) * n, 42 + unsigned(n));
  // Keep high powers well-scaled, while retaining nonsymmetric signed entries.
  for (float& value : input) value /= 16.0f * n;
  for (int i = 0; i < n; ++i) input[size_t(i) * n + i] += 0.875f;
  return input;
}

static std::vector<float> identity(int n) {
  std::vector<float> input(size_t(n) * n, 0.0f);
  for (int i = 0; i < n; ++i) input[size_t(i) * n + i] = 1.0f;
  return input;
}

// A = I + u*v^T has A^P = I + (((1+s)^P-1)/s)*u*v^T, s = v^T*u.
// Dyadic entries make the input construction exact in FP32. This validates
// every entry of a large dense result without an O(P*N^3) CPU reference.
static std::vector<float> rank_one_input(int n, int p,
                                        std::vector<double>& expected) {
  int width = 1;
  while (width < n) width *= 2;
  std::vector<double> u(n), v(n);
  double s = 0;
  for (int i = 0; i < n; ++i) {
    u[i] = 1.0 + (i % 4) * 0.25;
    v[i] = double(i % 8 + 1) / (16.0 * width);
    s += u[i] * v[i];
  }
  const double factor = std::expm1(p * std::log1p(s)) / s;
  std::vector<float> input(size_t(n) * n);
  expected.resize(input.size());
  for (int row = 0; row < n; ++row)
    for (int col = 0; col < n; ++col) {
      const size_t index = size_t(row) * n + col;
      const double outer = u[row] * v[col];
      input[index] = float((row == col ? 1.0 : 0.0) + outer);
      expected[index] = (row == col ? 1.0 : 0.0) + factor * outer;
    }
  return input;
}

int main(int argc, char** argv) {
  if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--large") != 0)) {
    std::fprintf(stderr, "Usage: %s [--large]\n", argv[0]);
    return 2;
  }
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s\nMatrix power: CPU double/analytic references; "
              "atol=1e-5 rtol=%g; 2 calls per case\n", prop.name,
              argc == 2 ? 1e-3 : 1e-4);
  int total = 0, failed = 0;
  auto check = [&](const char* name, int n, int p, std::vector<float> input,
                   std::vector<double> expected = {}, bool exact = false,
                   int input_offset = 0, int output_offset = 0,
                   double rtol = 1e-4) {
    ++total;
    if (!run_case(name, n, p, std::move(input), std::move(expected), exact,
                  input_offset, output_offset, rtol)) ++failed;
  };
  if (argc == 2) {
    for (const auto& shape : {std::pair<int, int>{511, 3}, {512, 19},
                              {1023, 7}, {1024, 1}, {1024, 20}}) {
      std::vector<double> expected;
      auto input = rank_one_input(shape.first, shape.second, expected);
      // Long FP32 dot products and subsequent powers amplify rounding relative
      // to the analytic FP64 result. Use a separate large-case tolerance; keep
      // the small cases and exact structural checks unchanged.
      check("large-dense-analytic", shape.first, shape.second,
            std::move(input), std::move(expected), false, 0, 0, 1e-3);
    }
  } else {
    // The two examples from https://leetgpu.com/challenges/matrix-power.
    check("example-cube", 2, 3, {1, 2, 3, 4}, {37, 54, 81, 118}, true);
    check("example-square", 3, 2, {1, 0, 2, 0, 1, 0, 3, 0, 0},
          {7, 0, 2, 0, 1, 0, 3, 0, 6}, true);
    for (int p = 1; p <= 20; ++p)
      check("all-exponents", 3, p,
            {1, 0.25f, -0.125f, -0.125f, 0.875f, 0.25f, 0.375f, -0.25f, 1.125f});
    for (int n : {1, 2, 15, 16, 17, 31, 32, 33, 65})
      for (int p : {1, 2, 3, 20})
        check("dense-boundary", n, p, dense_input(n));
    for (int p : {1, 7, 20}) {
      auto input = identity(17);
      check("identity", 17, p, input,
            std::vector<double>(input.begin(), input.end()), true);
      check("zero", 17, p, std::vector<float>(17 * 17, 0),
            std::vector<double>(17 * 17, 0), true);
    }
    for (int p : {19, 20}) {
      const float diagonal[] = {-10, -2, -1, -0.5f, 0, 0.5f, 1, 2, 10};
      std::vector<float> input(17 * 17, 0);
      std::vector<double> expected(17 * 17, 0);
      for (int i = 0; i < 17; ++i) {
        input[i * 17 + i] = diagonal[i % 9];
        expected[i * 17 + i] = std::pow(double(diagonal[i % 9]), p);
      }
      check("diagonal-extremes", 17, p, std::move(input), std::move(expected));
      check("scalar-extreme", 1, p, {-10}, {std::pow(-10.0, p)});
    }
    for (int p : {16, 17, 20}) {
      std::vector<float> cycle(17 * 17, 0), shift(17 * 17, 0);
      std::vector<double> cycle_power(17 * 17, 0), shift_power(17 * 17, 0);
      for (int i = 0; i < 17; ++i) {
        cycle[i * 17 + (i + 1) % 17] = 1;
        cycle_power[i * 17 + (i + p) % 17] = 1;
        if (i + 1 < 17) shift[i * 17 + i + 1] = 1;
        if (i + p < 17) shift_power[i * 17 + i + p] = 1;
      }
      check("permutation-cycle", 17, p, std::move(cycle), std::move(cycle_power), true);
      check("nilpotent-shift", 17, p, std::move(shift), std::move(shift_power), true);
    }
    std::vector<float> edge(17 * 17, 0);
    edge[16] = 3;
    edge[16 * 17] = 2;
    edge[16 * 17 + 16] = 1;
    check("last-row-column", 17, 3, std::move(edge), {}, true);
    check("all-ones-growth", 3, 20, std::vector<float>(9, 1),
          std::vector<double>(9, std::pow(3.0, 19)));
    check("unaligned-input", 17, 7, dense_input(17), {}, false, 1, 0);
    check("unaligned-output", 17, 6, dense_input(17), {}, false, 0, 1);
    check("unaligned-both", 33, 19, dense_input(33), {}, false, 3, 1);
    // Also exercise the large-suite analytic oracle on a small dense case.
    std::vector<double> expected;
    auto input = rank_one_input(17, 5, expected);
    const auto cpu = reference_power(input, 17, 5);
    for (size_t i = 0; i < cpu.size(); ++i) {
      if (std::abs(cpu[i] - expected[i]) > 1e-11 * std::max(1.0, std::abs(cpu[i]))) {
        std::fprintf(stderr, "Analytic reference disagrees with CPU at %zu\n", i);
        return EXIT_FAILURE;
      }
    }
    check("dense-analytic", 17, 5, std::move(input), std::move(expected));
  }
  std::printf("Matrix power: %d/%d cases passed\n", total - failed, total);
  return test::finish(failed == 0);
}
