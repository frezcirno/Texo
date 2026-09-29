# LeetGPU maximum window sum reference bug

Investigated on 2026-09-29 using the official
[AlphaGPU/leetgpu-challenges](https://github.com/AlphaGPU/leetgpu-challenges)
repository, cloned alongside `texo` at `../leetgpu-challenges`.
The inspected commit is `e6579a64e20e30d63f8cc18c79d59d3c95588db1`.
The upstream checkout and `src/max_subarray_sum.cu` were left unchanged.

## Cause

In [challenge.py, lines 26–38](https://github.com/AlphaGPU/leetgpu-challenges/blob/e6579a64e20e30d63f8cc18c79d59d3c95588db1/challenges/medium/51_max_subarray_sum/challenge.py#L26-L38),
the initial assignment `max_sum = current_sum` aliases the same scalar Tensor.
The first `current_sum += ...` mutates both names' value. The following
`torch.max` creates a separate Tensor, but the first window's sum is already lost.

For N > window_size, the reference therefore maximizes windows starting at
indices 1 through N-window_size, omitting the window at index 0. It gives a wrong
answer when the first window is strictly better than every later window.
For N == window_size, the loop does not run and the reference is correct.

The smallest repair to the **upstream reference** is to change its initialization
to `max_sum = current_sum.clone()`. Alternatively, avoid the in-place update to
`current_sum`. This does not require changing the CUDA prefix scan or excluding
the first window from the CUDA computation. An in-memory copy of the reference
with the `clone()` change was verified on all four directed cases on CPU;
it returns the correct answer in each, including 285 for the large case.
The upstream file was not edited.

## Reproduction

`scripts/check_max_subarray_sum_reference.py` imports the unmodified official
challenge. It executes its example, six functional tests and performance test
with seed 42, then four directed cases. It compares the official CUDA PyTorch
reference with an independent Python integer sliding-window oracle and the
current CUDA `solve`. Each CUDA case runs three times on the same output;
the output starts at INT_MAX and is not reset by the caller between calls.
Input preservation is also checked.

```sh
# From the parent of texo, if the checkout does not already exist:
git clone https://github.com/AlphaGPU/leetgpu-challenges.git

# From texo, with a Python environment containing CUDA-enabled PyTorch:
mkdir -p build/sm_80
/usr/local/cuda/bin/nvcc -O3 -std=c++14 -arch=sm_80 -shared \
  -Xcompiler -fPIC src/max_subarray_sum.cu \
  -o build/sm_80/max_subarray_sum_reference_probe.so
CUDA_VISIBLE_DEVICES=3 /home/zixuantan/miniconda3/envs/cp311/bin/python \
  scripts/check_max_subarray_sum_reference.py \
  --library build/sm_80/max_subarray_sum_reference_probe.so
```

The script accepts `--repo` for another checkout location. Its exit status
checks CUDA correctness against the independent oracle; reference disagreements
are printed explicitly, not adopted as expected answers. No online submission
or platform warmup runner is invoked; that runner is not in the cloned repository.

## Results

On A800 GPU 3 with PyTorch 2.5.1+cu121, the current CUDA implementation passes
all 12 cases, three calls each. The official reference disagrees in three
directed cases:

| Input / case | Window | Independent oracle | CUDA | Official reference |
| --- | ---: | ---: | ---: | ---: |
| `[9, -8]` | 1 | 9 | 9 | -8 |
| `[-1, -9]` | 1 | -1 | -1 | -9 |
| `[9, -8]` | 2 | 1 | 1 | 1 |
| Constructed N=50000 case | 25000 | 285 | 285 | 268 |

The large directed case has a first-window sum of 285, a second-window sum of
268, and smaller later sums. This reproduces the reported numerical discrepancy,
but it is **constructed data**, not the platform's truncated input. The platform's
full input and random seed are unavailable, so identity with that particular
input is not established.

The official generated cases for local seed 42 all agree with the independent
oracle; none has a uniquely best first window. Passing those random cases alone
would miss the bug.

The normal maintained suite also passes 80 quick and four large tests after the
user's output-reset fix. A separate stress probe checked 1000 random inputs at
N=50000/window_size=25000, five calls each, including every prefix sum. All
passed. Sampled memcheck and racecheck runs of that probe reported zero errors
and zero hazards. These checks do not prove the platform uses the same runner.

Artifacts:

- `build/sm_80/max_subarray_sum-official-reference.log`: official-reference comparison.
- `build/sm_80/max_subarray_sum-current-{quick,large}.log`: current normal suite.
- `build/max-subarray-repeat-review-skvknfqd/`: source snapshots and stress probes.

Keep the CUDA implementation faithful to the statement: include the first
window. The earlier suggestion to replace host-side output initialization with
a GPU kernel was an unconfirmed hypothesis; it does not repair this reference bug.
