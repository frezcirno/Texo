#include "test_utils.h"
#include <cstdint>
#include <limits>

extern "C" void solve(const int*, unsigned int*, int, int);

static uint32_t reference_hash(uint32_t word) {
  uint32_t hash = 2166136261u;
  // Hash four bytes from least significant to most significant, modulo 2^32.
  for (int byte = 0; byte < 4; ++byte) {
    hash = uint32_t((uint64_t(hash) ^ (word % 256)) * 16777619u);
    word /= 256;
  }
  return hash;
}

int main() {
  static_assert(sizeof(int) == sizeof(uint32_t), "32-bit input required");
  bool passed = true;
  // Known one-round anchors, including byte-order-sensitive input.
  if (reference_hash(0) != 0x4b95f515u || reference_hash(1) != 0xfb69b604u) {
    std::fprintf(stderr, "CPU FNV-1a reference self-check failed\n");
    return EXIT_FAILURE;
  }
  for (int n : {1, 31, 32, 33, 255, 256, 257, 1025, 65537}) {
    std::vector<int> input(n);
    std::mt19937 rng(42);
    for (int& value : input) {
      uint32_t bits = rng();
      std::memcpy(&value, &bits, sizeof(bits));
    }
    const int special[] = {0, 1, -1, std::numeric_limits<int>::min(),
                           std::numeric_limits<int>::max(), 0x01020304};
    for (int i = 0; i < std::min(n, 6); ++i) input[i] = special[i];
    test::DeviceArray<int> di(input);
    for (int rounds : {0, 1, 2, 7, 32}) {
      std::vector<double> reference(n);
      for (int i = 0; i < n; ++i) {
        uint32_t word = uint32_t(input[i]);
        for (int r = 0; r < rounds; ++r) word = reference_hash(word);
        reference[i] = word;
      }
      char label[32];
      std::snprintf(label, sizeof(label), "FNV-1a R=%d", rounds);
      passed &= test::check_output<unsigned int>(label, reference, [&](unsigned int* output) {
        solve(di.data(), output, n, rounds);
      });
      passed &= di.unchanged(input);
    }
  }
  return test::finish(passed);
}
