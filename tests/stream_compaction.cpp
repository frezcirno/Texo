#include "test_utils.h"
#include <string>

extern "C" void solve(const float*, int, float*);

static bool run_case(const char* name, std::vector<float> input) {
  constexpr size_t GUARD = 4;
  constexpr float MARKER = 1234567.0f;
  std::vector<float> expected(input.size(), 0.0f);
  size_t selected = 0;
  for (float value : input)
    if (value > 0.0f) expected[selected++] = value;

  test::DeviceArray<float> device_input(input);
  std::vector<float> initial(GUARD + input.size() + GUARD, MARKER);
  std::fill(initial.begin() + GUARD, initial.begin() + GUARD + input.size(), 0.0f);
  test::DeviceArray<float> output(initial);
  bool passed = true;
  for (int call = 0; call < 2; ++call) {
    output.upload(initial);  // The challenge specifies a zero-initialized output buffer.
    solve(device_input.data(), int(input.size()), output.data() + GUARD);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto actual = output.download();
    for (size_t i = 0; i < GUARD; ++i)
      if (actual[i] != MARKER || actual[GUARD + input.size() + i] != MARKER) {
        std::fprintf(stderr, "%s call=%d output guard overwritten\n", name, call + 1);
        passed = false;
        break;
      }
    for (size_t i = 0; i < input.size(); ++i)
      if (actual[GUARD + i] != expected[i]) {
        std::fprintf(stderr, "%s call=%d index=%zu got=%.9g expected=%.9g\n", name,
                     call + 1, i, actual[GUARD + i], expected[i]);
        passed = false;
        break;
      }
    passed &= device_input.unchanged(input);
  }
  std::printf("%s %-24s N=%zu selected=%zu\n", passed ? "PASS" : "FAIL", name,
              input.size(), selected);
  return passed;
}

static std::vector<float> mixed_values(int n) {
  std::vector<float> values(n);
  for (int i = 0; i < n; ++i) {
    const int residue = (i * 11 + 3) % 9;
    values[i] = residue < 3 ? float(i + 1) * 0.125f
               : residue == 3 ? 0.0f
                              : -float(i + 1) * 0.25f;
  }
  return values;
}

int main() {
  bool passed = true;
  passed &= run_case("problem-example", {1.0f, -2.0f, 3.0f, 0.0f, -1.0f, 4.0f});
  passed &= run_case("single-zero", {0.0f});
  passed &= run_case("single-positive", {5.0f});
  passed &= run_case("all-positive", std::vector<float>(33, 1.25f));
  passed &= run_case("all-non-positive", std::vector<float>(257, -1.0f));
  for (int n : {31, 32, 33, 255, 256, 257, 513, 1025, 65537})
    passed &= run_case(("mixed-" + std::to_string(n)).c_str(), mixed_values(n));
  return test::finish(passed);
}
