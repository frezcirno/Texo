#include "test_utils.h"

#include <complex>
#include <string>

extern "C" void solve(const float*, float*, int, int);

struct Case {
  std::string name;
  int rows;
  int cols;
  std::vector<float> signal;
  int input_offset = 0;
  int output_offset = 0;
};

static std::vector<float> patterned_signal(int rows, int cols, int seed) {
  std::vector<float> signal(2 * size_t(rows) * cols);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const int index = row * cols + col;
      signal[2 * index] =
          float((row * 17 + col * 29 + seed * 7) % 43 - 21) * 0.125f;
      signal[2 * index + 1] =
          float((row * 31 + col * 11 + seed * 13) % 47 - 23) * 0.0625f;
    }
  }
  return signal;
}

// A double-precision, separable DFT is intentionally used instead of an FFT:
// it is simple enough to serve as an independent oracle for the CUDA FFT.
static std::vector<double> reference_dft(const std::vector<float>& signal,
                                         int rows, int cols) {
  const size_t count = size_t(rows) * cols;
  std::vector<std::complex<double>> row_fft(count);
  std::vector<std::complex<double>> spectrum(count);

  for (int row = 0; row < rows; ++row) {
    for (int frequency = 0; frequency < cols; ++frequency) {
      std::complex<double> sum = 0.0;
      for (int col = 0; col < cols; ++col) {
        const size_t index = size_t(row) * cols + col;
        const double angle = -2.0 * M_PI * double(frequency) * col / cols;
        const std::complex<double> w(std::cos(angle), std::sin(angle));
        sum += std::complex<double>(signal[2 * index], signal[2 * index + 1]) * w;
      }
      row_fft[size_t(row) * cols + frequency] = sum;
    }
  }

  for (int frequency_row = 0; frequency_row < rows; ++frequency_row) {
    for (int frequency_col = 0; frequency_col < cols; ++frequency_col) {
      std::complex<double> sum = 0.0;
      for (int row = 0; row < rows; ++row) {
        const double angle = -2.0 * M_PI * double(frequency_row) * row / rows;
        const std::complex<double> w(std::cos(angle), std::sin(angle));
        sum += row_fft[size_t(row) * cols + frequency_col] * w;
      }
      spectrum[size_t(frequency_row) * cols + frequency_col] = sum;
    }
  }

  std::vector<double> result(2 * count);
  for (size_t i = 0; i < count; ++i) {
    result[2 * i] = spectrum[i].real();
    result[2 * i + 1] = spectrum[i].imag();
  }
  return result;
}

static bool run_case(const Case& c) {
  constexpr size_t GUARD = 4;
  constexpr float MARKER = 1234567.0f;
  constexpr double ATOL = 2e-3;
  constexpr double RTOL = 2e-4;

  const size_t complex_count = size_t(c.rows) * c.cols;
  const size_t float_count = 2 * complex_count;
  const size_t input_prefix = GUARD + size_t(c.input_offset);
  const size_t output_prefix = GUARD + size_t(c.output_offset);
  if (c.rows <= 0 || c.cols <= 0 || c.input_offset < 0 || c.output_offset < 0 ||
      c.signal.size() != float_count) {
    std::fprintf(stderr, "%s: invalid test case\n", c.name.c_str());
    return false;
  }

  const std::vector<double> expected = reference_dft(c.signal, c.rows, c.cols);
  std::vector<float> stored_input(input_prefix + float_count + GUARD, MARKER);
  std::copy(c.signal.begin(), c.signal.end(), stored_input.begin() + input_prefix);
  std::vector<float> initial_output(output_prefix + float_count + GUARD, MARKER);

  test::DeviceArray<float> input(stored_input);
  test::DeviceArray<float> output(initial_output);
  bool passed = true;
  double max_error = 0.0;

  // Repeating each call catches accidental persistence in a future implementation.
  for (int call = 0; call < 2; ++call) {
    output.upload(initial_output);
    solve(input.data() + input_prefix, output.data() + output_prefix, c.rows, c.cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto actual = output.download();
    for (size_t i = 0; i < actual.size(); ++i) {
      if ((i < output_prefix || i >= output_prefix + float_count) &&
          actual[i] != MARKER) {
        std::fprintf(stderr, "%s call=%d output guard overwritten at %zu\n",
                     c.name.c_str(), call + 1, i);
        passed = false;
        break;
      }
    }
    for (size_t i = 0; i < float_count; ++i) {
      const double got = actual[output_prefix + i];
      const double error = std::abs(got - expected[i]);
      max_error = std::max(max_error, error);
      if (!std::isfinite(got) || error > ATOL + RTOL * std::abs(expected[i])) {
        std::fprintf(stderr,
                     "%s call=%d component=%zu got=%.9g expected=%.9g error=%.3g\n",
                     c.name.c_str(), call + 1, i, got, expected[i], error);
        passed = false;
        break;
      }
    }
    passed &= input.unchanged(stored_input);
  }

  std::printf("%s %-26s M=%d N=%d max_abs_error=%.3g\n",
              passed ? "PASS" : "FAIL", c.name.c_str(), c.rows, c.cols, max_error);
  return passed;
}

int main(int argc, char** argv) {
  const bool large = argc == 2 && std::string(argv[1]) == "--large";
  if (argc > 2 || (argc == 2 && !large)) {
    std::fprintf(stderr, "usage: %s [--large]\n", argv[0]);
    return EXIT_FAILURE;
  }

  bool passed = true;

  passed &= run_case({"unit-complex", 1, 1, {2.5f, -1.25f}});
  passed &= run_case({"statement-impulse", 2, 2,
                      {1.0f, 0.0f, 0.0f, 0.0f,
                       0.0f, 0.0f, 0.0f, 0.0f}});
  passed &= run_case({"non-power-of-two", 3, 5, patterned_signal(3, 5, 9)});
  passed &= run_case({"non-power-rectangular", 17, 33,
                      patterned_signal(17, 33, 10), 2, 1});
  passed &= run_case({"single-row-complex", 1, 64, patterned_signal(1, 64, 1)});
  passed &= run_case({"single-column-complex", 64, 1, patterned_signal(64, 1, 2)});
  passed &= run_case({"wide-rectangular", 8, 32, patterned_signal(8, 32, 3), 1, 3});
  passed &= run_case({"tall-rectangular", 32, 8, patterned_signal(32, 8, 4), 3, 1});
  passed &= run_case({"many-transforms", 64, 128, patterned_signal(64, 128, 5)});

  if (large) {
    // These exercise the largest dynamic shared-memory allocation (48 KiB) in
    // the row and column kernels without making the default test slow.
    passed &= run_case({"max-row-length", 1, 4096, patterned_signal(1, 4096, 6)});
    passed &= run_case({"max-column-length", 4096, 1, patterned_signal(4096, 1, 7)});
    passed &= run_case({"large-square", 256, 256, patterned_signal(256, 256, 8)});
  }
  return test::finish(passed);
}
