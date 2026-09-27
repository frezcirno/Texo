#include "test_utils.h"
#include <limits>
#include <string>
#include <utility>

extern "C" void solve(const float*, int*, int);

struct Case {
  std::string name;
  std::vector<float> points;
  // Optional known nearest neighbors avoid an O(N^2) CPU scan for large cases.
  std::vector<int> neighbors;
  int input_offset = 0;
  int output_offset = 0;
};

static double distance_squared(const std::vector<float>& points, int i, int j) {
  double distance = 0;
  for (int axis = 0; axis < 3; ++axis) {
    const double delta = double(points[size_t(i) * 3 + axis]) -
                         double(points[size_t(j) * 3 + axis]);
    distance += delta * delta;
  }
  return distance;
}

static std::vector<double> reference_distances(const Case& c) {
  const int n = int(c.points.size() / 3);
  std::vector<double> distances(n, std::numeric_limits<double>::infinity());
  for (int i = 0; i < n; ++i) {
    if (!c.neighbors.empty()) {
      distances[i] = distance_squared(c.points, i, c.neighbors[i]);
    } else {
      for (int j = 0; j < n; ++j)
        if (i != j)
          distances[i] = std::min(distances[i], distance_squared(c.points, i, j));
    }
  }
  return distances;
}

static bool run_case(Case c) {
  constexpr size_t GUARD = 4;
  constexpr int MARKER = 1234567;
  const int n = int(c.points.size() / 3);
  const size_t prefix = GUARD + c.output_offset;
  // Do not pad the input tail: out-of-bounds reads must remain visible to memcheck.
  test::DeviceArray<float> device_points(c.points.size() + c.input_offset);
  test::DeviceArray<int> device_indices(prefix + n + GUARD);
  bool passed = true;
  for (int call = 0; call < 2; ++call) {
    if (call == 1) {
      // Rotate point order and apply an isometry (x,y,z) -> (-y,-z,-x).
      // Nearest distances stay the same, but the required indices change.
      auto old_points = c.points;
      auto old_neighbors = c.neighbors;
      for (int i = 0; i < n; ++i) {
        const int old_i = (i + 1) % n;
        for (int axis = 0; axis < 3; ++axis)
          c.points[size_t(i) * 3 + axis] =
              -old_points[size_t(old_i) * 3 + (axis + 1) % 3];
        if (!old_neighbors.empty())
          c.neighbors[i] = (old_neighbors[old_i] + n - 1) % n;
      }
    }
    const auto expected = reference_distances(c);
    std::vector<float> stored_points(c.points.size() + c.input_offset, float(MARKER));
    std::copy(c.points.begin(), c.points.end(), stored_points.begin() + c.input_offset);
    device_points.upload(stored_points);
    std::vector<int> initial(prefix + n + GUARD, MARKER);
    std::fill(initial.begin() + prefix, initial.begin() + prefix + n,
              call == 0 ? -1 : std::numeric_limits<int>::min());
    device_indices.upload(initial);

    solve(device_points.data() + c.input_offset, device_indices.data() + prefix, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto actual = device_indices.download();
    for (size_t i = 0; i < actual.size(); ++i) {
      if ((i < prefix || i >= prefix + n) && actual[i] != MARKER) {
        std::fprintf(stderr, "%s call=%d output guard overwritten at %zu\n",
                     c.name.c_str(), call + 1, i);
        passed = false;
        break;
      }
    }
    int mismatches = 0;
    // N=1 is allowed in the statement, but no j!=i exists and no sentinel is
    // specified. Check storage safety only, without inventing an output value.
    if (n > 1) {
      for (int i = 0; i < n; ++i) {
        const int j = actual[prefix + i];
        const bool valid = j >= 0 && j < n && j != i;
        const double got = valid ? distance_squared(c.points, i, j) :
                                   std::numeric_limits<double>::infinity();
        // Ties are unspecified: accept every non-self candidate at the minimum
        // distance. Integer/dyadic test data avoid ambiguous near-tie rounding.
        if (!valid || got != expected[i]) {
          if (mismatches == 0)
            std::fprintf(stderr,
                         "%s call=%d point=%d neighbor=%d d2=%.12g min_d2=%.12g\n",
                         c.name.c_str(), call + 1, i, j, got, expected[i]);
          ++mismatches;
        }
      }
    } else {
      std::printf("INFO singleton call=%d index=%d (unspecified by challenge)\n",
                  call + 1, actual[prefix]);
    }
    if (mismatches) {
      std::fprintf(stderr, "%s call=%d mismatches=%d/%d\n",
                   c.name.c_str(), call + 1, mismatches, n);
      passed = false;
    }
    passed &= device_points.unchanged(stored_points);
  }
  std::printf("%s %-24s N=%d offsets=%d/%d%s\n", passed ? "PASS" : "FAIL",
              c.name.c_str(), n, c.input_offset, c.output_offset,
              n == 1 ? " storage-only" : "");
  return passed;
}

static std::vector<float> random_points(int n) {
  std::mt19937 rng(42 + unsigned(n));
  std::uniform_int_distribution<int> dist(-1000, 1000);
  std::vector<float> points(size_t(n) * 3);
  for (float& value : points) value = float(dist(rng));
  // Integer squared distances are <=12,000,000, exactly representable in FP32.
  return points;
}

static Case paired_grid(const char* name, int n) {
  Case c{name, std::vector<float>(size_t(n) * 3), std::vector<int>(n)};
  // Each pair is 1/8 apart; distinct grid cells are at least 31/8 apart.
  // 37^3 cells accommodate 100,000 points entirely inside [-1000,1000]^3.
  for (int i = 0; i < n; ++i) {
    const int pair = i / 2;
    c.points[size_t(i) * 3] = float(4 * (pair % 37)) + (i % 2 ? 0.125f : 0.0f);
    c.points[size_t(i) * 3 + 1] = float(4 * ((pair / 37) % 37));
    c.points[size_t(i) * 3 + 2] = float(4 * (pair / (37 * 37)));
    c.neighbors[i] = i ^ 1;
  }
  return c;
}

static std::vector<Case> quick_cases() {
  std::vector<Case> cases;
  // A full block is also independently selectable when debugging unsafe tails.
  cases.push_back({"boundary-256", random_points(256), {}});
  cases.push_back({"example", {0, 0, 0, 1, 0, 0, 5, 5, 5}, {1, 0, 1}});
  for (int n : {2, 3, 4, 7, 15, 16, 17, 31, 32, 33, 127, 128, 129,
                255, 257, 258, 511, 512, 513, 1023, 1024, 1025})
    cases.push_back({"boundary-" + std::to_string(n), random_points(n), {}});
  for (int axis = 0; axis < 3; ++axis) {
    std::vector<float> points(12, 0);
    const float coordinates[] = {0, 1, 4, 10};
    for (int i = 0; i < 4; ++i) points[3 * i + axis] = coordinates[i];
    cases.push_back({"axis-" + std::to_string(axis), std::move(points), {1, 0, 1, 2}});
  }
  cases.push_back({"euclidean-not-manhattan", {0, 0, 0, 3, 0, 0, 2, 2, 0}, {2, 2, 1}});
  cases.push_back({"coordinate-extremes", {-1000, -1000, -1000, 1000, 1000, 1000}, {1, 0}});
  cases.push_back({"close-distinct", {0, 0, 0, 0.000244140625f, 0, 0,
                                     0.00048828125f, 0.00048828125f, 0}, {}});
  cases.push_back({"equidistant", {0, 0, 0, 1, 0, 0, -1, 0, 0,
                                  0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1}, {}});
  cases.push_back({"duplicate-points", {1, 2, 3, 4, 5, 6, 1, 2, 3, -7, -8, -9}, {}});
  cases.push_back({"all-identical", std::vector<float>(33 * 3, 2.0f), {}});
  auto tail = std::vector<float>(257 * 3, 0.0f);
  for (int i = 0; i < 256; ++i) tail[3 * i] = float(3 * i);
  tail[256 * 3] = 0.125f;
  cases.push_back({"last-point-nearest", std::move(tail), {}});
  auto pairs = paired_grid("paired-grid", 258);
  pairs.neighbors.clear(); // Validate the pair construction with exhaustive CPU search.
  cases.push_back(std::move(pairs));
  cases.push_back({"unaligned-input", random_points(257), {}, 1, 0});
  cases.push_back({"unaligned-output", random_points(257), {}, 0, 1});
  cases.push_back({"unaligned-both", random_points(259), {}, 3, 1});
  cases.push_back({"singleton-storage", {1, 2, 3}, {}});
  return cases;
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  const bool large = argc == 2 && std::strcmp(argv[1], "--large") == 0;
  const bool list = argc == 2 && std::strcmp(argv[1], "--list-cases") == 0;
  const bool selected = argc == 3 && std::strcmp(argv[1], "--case") == 0;
  if (argc != 1 && !large && !list && !selected) {
    std::fprintf(stderr, "Usage: %s [--large | --case NAME | --list-cases]\n", argv[0]);
    return 2;
  }
  auto cases = large ? std::vector<Case>{paired_grid("large-10000", 10000),
                                        paired_grid("large-100000", 100000)} : quick_cases();
  if (list) {
    for (const auto& c : cases) std::puts(c.name.c_str());
    return 0;
  }
  if (selected) {
    auto found = std::find_if(cases.begin(), cases.end(),
                              [&](const Case& c) { return c.name == argv[2]; });
    if (found == cases.end()) {
      std::fprintf(stderr, "Unknown case: %s (see --list-cases)\n", argv[2]);
      return 2;
    }
    Case c = std::move(*found);
    cases.clear();
    cases.push_back(std::move(c));
  }
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s\nNearest neighbor: non-self minimum squared Euclidean distance; "
              "ties accepted; 2 calls per case\n", prop.name);
  int failed = 0;
  for (auto& c : cases)
    if (!run_case(std::move(c))) ++failed;
  std::printf("Nearest neighbor: %zu/%zu cases passed\n", cases.size() - failed, cases.size());
  return test::finish(failed == 0);
}
