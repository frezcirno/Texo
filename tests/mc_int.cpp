#include "test_utils.h"
#include <utility>

extern "C" void solve(const float*, float*, float, float, int);

enum class OutputMode { Cleared, Poisoned, Reused };

static bool run_case(const char* name, std::vector<float> input, float a, float b,
                     OutputMode mode) {
  constexpr size_t GUARD = 4;
  constexpr float MARKER = 1234567.0f;
  constexpr double ATOL = 1e-2, RTOL = 1e-2;
  const size_t n = input.size();
  // cudaMalloc aligns the input for vector loads. Do not pad its end: memcheck
  // should detect a vector load that extends beyond a scalar tail.
  test::DeviceArray<float> device_input(n);
  test::DeviceArray<float> device_output(2 * GUARD + 1);
  const int calls = mode == OutputMode::Reused ? 3 : 2;
  bool passed = true;
  double max_error = 0;
  for (int call = 0; call < calls; ++call) {
    // Reuse the same allocations with changed data on the last call.
    if (call == calls - 1)
      for (float& value : input) value = -value;
    device_input.upload(input);

    // Compute the estimate from the actual supplied FP32 samples, not the
    // analytic integral of the function that might have generated them.
    double sum = 0;
    for (float value : input) sum += double(value);
    const double expected = sum * (double(b) - double(a)) / double(n);

    // Most cases clear output to isolate numerical/indexing errors. Dedicated
    // cases require solve to overwrite nonzero output and prior call results.
    if (mode != OutputMode::Reused || call == 0) {
      std::vector<float> initial(2 * GUARD + 1, MARKER);
      initial[GUARD] = mode == OutputMode::Poisoned
                           ? (call == 0 ? 1234.5f : -4567.0f) : 0.0f;
      device_output.upload(initial);
    }
    solve(device_input.data(), device_output.data() + GUARD, a, b, int(n));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto actual = device_output.download();
    const double got = actual[GUARD];
    const double error = std::abs(got - expected);
    max_error = std::max(max_error, error);
    if (!std::isfinite(got) || error > ATOL + RTOL * std::abs(expected)) {
      std::fprintf(stderr,
                   "%s call=%d N=%zu a=%.9g b=%.9g got=%.12g expected=%.12g\n",
                   name, call + 1, n, a, b, got, expected);
      passed = false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
      if (i != GUARD && actual[i] != MARKER) {
        std::fprintf(stderr, "%s call=%d output guard overwritten at %zu\n",
                     name, call + 1, i);
        passed = false;
        break;
      }
    }
    passed &= device_input.unchanged(input);
  }
  const char* mode_name = mode == OutputMode::Cleared ? "cleared" :
                         mode == OutputMode::Poisoned ? "poisoned" : "reused";
  std::printf("%s %-24s N=%zu output=%s max_abs_error=%.3g\n",
              passed ? "PASS" : "FAIL", name, n, mode_name, max_error);
  return passed;
}

static std::vector<float> cancellation_input(size_t n) {
  // The small residual is part of the FP32 input. FP32 addition can lose it
  // before a later conversion to double; the wide interval makes it observable.
  const float values[] = {10000.0f, 0.0001f, -10000.0f, 0.0f};
  std::vector<float> input(n);
  for (size_t i = 0; i < n; ++i) input[i] = values[i % 4];
  return input;
}

int main(int argc, char** argv) {
  if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--large") != 0)) {
    std::fprintf(stderr, "Usage: %s [--large]\n", argv[0]);
    return 2;
  }
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s\nMC integration CPU double reference; atol=1e-2 rtol=1e-2\n",
              prop.name);
  int total = 0, failed = 0;
  auto check = [&](const char* name, std::vector<float> input, float a, float b,
                   OutputMode mode = OutputMode::Cleared) {
    ++total;
    if (!run_case(name, std::move(input), a, b, mode)) ++failed;
  };

  if (argc == 2) {
    // Kept out of the quick suite and sanitizer runs. These are correctness
    // checks, not timings: CPU reference/copies/input-preservation are included.
    check("large-10-million", std::vector<float>(10000000, 1.25f), -2, 6);
    check("large-100-million", std::vector<float>(100000000, 3.25f), -1, 5);
  } else {
    check("hand-calculated", {1, 2, 3, 4}, 0, 2); // estimate = 5
    for (int n : {1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33,
                  127, 128, 129, 255, 256, 257, 258, 259,
                  511, 512, 513, 1023, 1024, 1025, 4097, 65537}) {
      check("mixed-boundary", test::random_values(n, 42 + unsigned(n)), -2.25f, 3.75f);
    }
    check("constant-positive", std::vector<float>(4097, 3.25f), -2, 5);
    check("constant-negative", std::vector<float>(4097, -3.25f), -2, 5);
    check("all-zero", std::vector<float>(257, 0.0f), -1000, 1000);
    check("positive-interval", {1, 2, 3, 4, 5}, 2, 5);
    check("negative-interval", {1, 2, 3, 4, 5}, -5, -1);
    check("narrow-interval", std::vector<float>(257, 10000.0f),
          999.0f, 999.0009765625f);

    // Isolate each of the 1/2/3 scalar tail positions after float4 groups.
    // A dropped sample then changes the whole result, even at loose tolerance.
    for (int tail = 1; tail <= 3; ++tail) {
      for (int pos = 0; pos < tail; ++pos) {
        std::vector<float> input(1024 + tail, 0.0f);
        input[1024 + pos] = 10000.0f;
        char name[64];
        std::snprintf(name, sizeof(name), "tail-%d-position-%d", tail, pos);
        check(name, std::move(input), -1000, 1000);
      }
    }
    check("cancellation-small", cancellation_input(4), -1000, 1000);
    check("cancellation-blocks", cancellation_input(1028), -1000, 1000);
    check("large-tail", test::random_values(1048579, 123), -1000, 1000);
    check("overwrite-nonzero", {1, 2, 3, 4}, 0, 2, OutputMode::Poisoned);
    check("overwrite-zero-sum", std::vector<float>(257, 0.0f),
          -1000, 1000, OutputMode::Poisoned);
    check("consecutive-calls", {1, 2, 3, 4}, 0, 2, OutputMode::Reused);
    check("consecutive-blocks", std::vector<float>(4097, 3.25f),
          -2, 5, OutputMode::Reused);
  }
  std::printf("MC integration: %d/%d cases passed\n", total - failed, total);
  return test::finish(failed == 0);
}
