# Testing and benchmarking

## Build and check

```bash
make -j2 all compile-kernels
make check
make check-full
make sanitize
```

`all` links the operator tests/benchmarks. `compile-kernels` additionally compiles
every file in `src/` independently, including examples without runtime tests.
No kernels from different standalone operators are linked into one library.

`check` runs all maintained operator tests with modest benchmark sizes. It covers
partial blocks, multiple blocks, finite outputs, and numerical agreement according
to each executable's tolerances. It does not run the unimplemented top-k exercise.
`check-full` adds the default MSE suite, including 50 million inputs to catch the
rounding error seen with repeated float atomic additions, plus top-k selection with
N=50000000 and k=100. The top-k suite compares against a CPU partial sort and reports
the best of two warm wrapper wall times, including allocation and freeing.
`check-full` also runs an INT8 8192x4096x2048 cancellation regression with an
analytic reference for every output element, plus Monte Carlo integration with
10 million and 100 million samples, matrix-power analytic cases through N=1024,
nearest-neighbor cases with 10,000/100,000 points, batch normalization at
N=5000/10000,C=1024, NCHW max pooling with N=4,k=3,s=2 at large spatial
and channel dimensions, integer occurrence counts with 16,777,217/100M
elements, 3D counts at 500^3/1000^3, subarray sums over 100M-element arrays,
2D subarray sums over 10000x10000 matrices, 3D subarray sums over 500^3 tensors,
and RMS normalization with N=99999/100000.

The newer elementwise, matrix addition/copy, reversal, interleave, 1D convolution,
hash and RGB-to-grayscale tests also run in `make check`. Each operator has its own
executable; six elementwise executables share `tests/elementwise.cpp`, while
sigmoid uses `tests/sigmoid.cpp`. Their small helper
`tests/test_utils.h` checks CUDA errors, CPU references, output guards, and repeated
calls, initializing output before each invocation. Input storage is checked for changes;
reversal instead verifies the in-place result and restoration after a second call.
The shared output helper preserves 16-byte alignment. Tests use positive sizes around
warp/block boundaries and non-multiples of block sizes.

Copy, addition, reversal, interleave, ReLU, clip and hashes use exact comparisons.
Leaky ReLU, sigmoid, SiLU and SwiGLU use CPU double references with atol=1e-6 and
rtol=3e-6; GEGLU uses atol=5e-6 and rtol=3e-6 to account for float erf cancellation.
1D cross-correlation uses atol=1e-4 and rtol=1e-5; RGB-to-grayscale uses atol=1e-7
and rtol=1e-6. Clip tests include interval
boundaries and equal bounds; hashes include signed bit patterns and multiple rounds.

`sigmoid_test` retains the basic mixed/zero/negative/positive checks and adds
warp/block boundaries, a million-element tail, independent unaligned input/output
pointers, NaN/infinities, signed zeros, tiny inputs and saturation near float exp
overflow. Every case runs twice on the same buffers with negated inputs on the
second call, nonzero output poison, output guards and bitwise input-preservation
checks. It uses a stable CPU double reference with the tolerances above; a dense
[-80,80] sweep uses relative tolerance only (3e-6) to check tiny normal results.
NaNs must propagate, infinities map exactly to 0/1, and signed zeros map exactly
to 0.5. Positive N is required by the existing operator contract. Run the 28
cases with `make run-sigmoid`; they also run in `make check` and under memcheck
in `make sanitize`.
Validated on 2026-09-25 with NVIDIA A800 80GB PCIe, CUDA Toolkit 12.6.20,
`-O3 -std=c++14 -arch=sm_80`: all 28 cases, the full `make check`, and the
sigmoid memcheck run passed (zero reported errors).

`batched_mm_test` has 44 cases covering rectangular matrices, independently
partial BATCH/M/N blocks, K=1 and long reductions, batch-specific data, an active
last batch with all others zero, identity/zero matrices, and a nonzero final
reduction element. It checks both multiplication with cleared C and three
consecutive `C += A*B` calls starting from nonzero C. Independent CPU double
products use atol=1e-5 and rtol=1e-5, with float rounding of C between accumulated
calls; zero products and inactive batches must preserve finite C exactly.
Unaligned A/B/C are tested separately and together. All calls check input storage
and output guards. K=0 passes null A/B and must preserve finite C; zero BATCH,
M or N must leave storage bitwise unchanged and permit null pointers.
Run it with `make run-batched-mm`; it also runs in `make check` and under
memcheck in `make sanitize`.
Validated on 2026-09-25 with NVIDIA A800 80GB PCIe, CUDA Toolkit 12.6.20,
`-O3 -std=c++14 -arch=sm_80`: all 44 cases, the full `make check`, and the
batched-MM memcheck run passed (zero reported errors).

`mm_int8_test` has 56 default cases covering rectangular matrices, independent
16x16 block boundaries, K=1/257/1025/2048, full signed INT8 inputs, independent input/output
zero points (including endpoints), quantized zeros and identity matrices, and
a nonzero last reduction element. It checks round-to-nearest, ties-to-even on
both signs, the adjacent FP32 values around 0.5, all three scales, nonbinary
scales, both saturation limits, and saturation after adding zero_point_C.
A/B/C are tested at independent and combined byte offsets. K=0 uses null A/B
and must fill C with zero_point_C. M/N are positive; empty outputs and invalid
scales are outside this operator's test contract.

The reference uses an INT64 dot product and double scaling, with an explicit
floor/parity implementation of ties-to-even rounding. Results
must match exactly. The main cases use binary scales and bounded sums to avoid
ambiguity from FP32 accumulation; the decimal-scale case stays away from half
boundaries. These are selected quantization regressions, not a claim of exact
FP32-versus-ideal agreement for all scales and reduction lengths. The reported
LeetGPU 3x5x2 failure pins all 15 expected bytes directly, including C[1,3]=58;
its sign-reversed case pins -58. A double formula applied to the binary FP32
scales can land on a different side of a half boundary, so those external
expectations are not regenerated from that formula. The old `roundf` kernel
reproduced 59 versus 58; `nearbyintf` passes the case and both-sign tie checks.
See NVIDIA's [rounding functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html)
for the difference between halfway-away-from-zero and halfway-to-even.
Two K=2048 cases pair integer products that cancel, leaving a small residual,
using scales 0.1/0.1/0.01. The former per-term FP32 scaling/reduction failed the
random case with 8 instead of 9 even after fixing the rounding rule. INT64
accumulation followed by one final scaling passes it. Two K=33026 cases also
check positive/negative dot products beyond INT32 range with unsaturated outputs.
Every case runs twice with C initialized to opposite INT8 endpoints, checking overwrites,
unchanged input bytes and output guards. Run `make run-mm-int8`; the test is
also included in `make check` and memcheck in `make sanitize`.

`make run-mm-int8 MM_INT8_ARGS=--large` runs the separate 8192x4096x2048 case,
also included in `make check-full`. Repeated A halves multiply opposite B halves;
the remaining term determines a row/column-specific integer output. This gives
an exact analytic reference for all 33,554,432 outputs without a cubic CPU
matrix multiply. It uses the reported dimensions/scales, with constructed data;
the platform's truncated input arrays cannot be replayed verbatim. The kernel
retains FP32 operations in the order `float(dot) * scale_A * scale_B / scale_C`;
precombining scales or using double can change half-boundary results.

The original lower-saturation branch assigned +128 before converting to INT8.
It happened to yield -128 and pass the baseline tests on this A800/compiler;
the branch now assigns -128 directly to avoid the out-of-range conversion.

Validated on 2026-09-25 with NVIDIA A800 80GB PCIe, CUDA Toolkit 12.6.20,
`-O3 -std=c++14 -arch=sm_80`: all 56 default cases, the full-size analytic
regression, and the complete `make check-full` passed. Memcheck passed all 56
default cases with zero errors. The `sm_75` (T4) build also passed; no physical
T4 runtime or online resubmission is claimed by these local checks.

`mc_int_test` checks `sum(y_samples) * (b-a) / n_samples` against CPU double
arithmetic with local tolerances atol=1e-2 and rtol=1e-2. The reference uses the
supplied samples, not an analytic integral or newly generated random points.
The 49-case quick suite covers scalar/vector/warp/block boundaries, individual scalar
tail positions, constants, zeros, positive/negative/narrow intervals, a
million-element tail, and cancellation with a small nonzero residual. Most
numerical cases clear the output; separate cases poison it or make consecutive
calls without clearing it, requiring overwrite semantics. Each case also changes
input signs on its last call, checks input preservation and output guards, and
leaves input tails unpadded for memcheck. Inputs are cudaMalloc-aligned; empty
inputs, nonfinite samples and invalid intervals are outside this test contract.

Run `make run-mc-int` for the quick suite (also in `make check`), or
`make run-mc-int MC_INT_ARGS=--large` for the separate 10-million/100-million
constant-input regressions (also in `make check-full`). These are correctness
checks, not performance benchmarks. `make sanitize-mc-int` runs the quick suite
under memcheck, initcheck, racecheck and synccheck, propagating both functional
failures and sanitizer errors; it is also included in `make sanitize`.

Initial validation on 2026-09-27 with NVIDIA A800 80GB PCIe: the unchanged
operator passes 43/49 quick cases and both large cases. Two cancellation cases
return zero instead of approximately +/-0.05 because the four FP32 values are
added before conversion to double. Four overwrite/reuse cases fail because
atomic additions retain the previous output. All four sanitizer tools report
zero memory/synchronization errors or race hazards, but the target correctly
returns failure for the numerical errors. Both sm_80 and sm_75 compile; only
sm_80 was run on hardware.

The 80-case `mat_pow_test` quick suite covers the two
[Matrix Power examples](https://leetgpu.com/challenges/matrix-power), every
exponent from 1 through 20, dimensions around 16/32 boundaries, scalar inputs,
identity/zero/diagonal matrices, permutation cycles, nilpotent shifts, a nonzero
last row/column, magnitude growth, and independently unaligned input/output
pointers. P=0 and negative powers are excluded by the challenge's P>=1 constraint.
General small cases use P sequential CPU double matrix products instead of the
CUDA recursion. Local quick-suite tolerances are atol=1e-5 and rtol=1e-4, with exact checks
for selected integer/structural cases and bitwise copies for P=1. Test matrices
keep intermediate powers finite; this does not promise numerical agreement for
every ill-conditioned matrix in the input range.

Every case runs twice using the same allocations, with NaN/finite output poison,
output guards, unpadded input tails and input-preservation checks. The second
input is the negated transpose, whose expected power follows independently from
`(-A^T)^P = (-1)^P (A^P)^T`. Run `make run-mat-pow` for the quick suite, also in
`make check`. `make run-mat-pow MAT_POW_ARGS=--large` runs five separate dense analytic
cases at N=511/512/1023/1024, including P=20; these also run in `make check-full`.
They use `A = I + u*v^T` and its closed-form power to validate all output entries
without a cubic CPU calculation. The large suite uses atol=1e-5 and rtol=1e-3
to allow accumulated rounding from long FP32 dot products and repeated powers;
these are local tolerances, not an assertion about hidden platform tolerances.
The analytic oracle is cross-checked against
the CPU products on a small case. These are correctness checks, not timings.
`make sanitize-mat-pow` runs the quick suite under memcheck with full leak checks
and initcheck; it is also part of `make sanitize`.

Validated on 2026-09-27 with NVIDIA A800 80GB PCIe and `-arch=sm_80`: all 80
quick cases and five large cases pass. Memcheck reports zero errors and zero
leaked allocations; initcheck reports zero errors. Both sm_80 and sm_75 builds
pass, with no physical T4 run or online submission implied. In the N=1024, P=20
analytic case, the maximum absolute difference from FP64 is about 2.75e-4;
the large-suite tolerance above accounts for this FP32 rounding difference.

The 39-case `nn_test` quick suite validates the [Nearest Neighbor](https://leetgpu.com/challenges/nearest-neighbor)
contract using an exhaustive CPU double squared-distance search for small
cases. Every returned index must be in range, exclude the query point itself,
and reach the minimum distance; any equally near point is accepted because the
statement does not specify tie-breaking. Integer and dyadic inputs avoid
ambiguous near-tie rounding, so comparisons are exact. The example, individual
coordinate axes, Euclidean versus Manhattan distance, coordinate extremes,
close distinct points, duplicate points, ties, last-point candidates,
warp/block boundaries and unaligned buffers are covered. The statement permits
N=1 but specifies no output when there is no other point; `singleton-storage`
checks input preservation and output guards without inventing an expected index.

Each case runs twice with the same allocations and negative output poison.
The second call rotates point order and applies a coordinate isometry, checking
index changes as well as unchanged input bytes and output guards. Input tails
are unpadded so memcheck can detect inactive threads reading beyond N. Run
`make run-nn` (also in `make check`); use `NN_ARGS="--case boundary-256"` to
isolate a case or `NN_ARGS=--list-cases` to list names without initializing CUDA.
`make run-nn NN_ARGS=--large` runs two separate 10,000/100,000-point paired-grid
cases, also in `make check-full`. Each pair is 1/8 apart and other pairs are
at least 31/8 apart, providing an O(N) analytic reference. A small paired-grid
case is checked with exhaustive CPU search. These are correctness checks, not
performance measurements. `make sanitize-nn` runs memcheck with leak checking
and initcheck, accepts `NN_ARGS`, and is included in `make sanitize`.

Revalidated on 2026-09-27 on A800 after correcting candidate indexing and adding
the thread-bound check: all 39 quick cases and both 10,000/100,000-point cases
pass. Memcheck reports zero errors and zero leaked allocations; initcheck reports
zero errors. N=1 returns -1 in the current implementation, while the test still
checks only storage safety because the challenge does not specify a sentinel.
Both sm_80 and sm_75 compile; only sm_80 was run on hardware. No physical T4 run
or online submission is implied by these local checks.

The 52-case `batch_norm_test` quick suite uses an independent CPU FP64 Welford reference for the
[Batch Normalization](https://leetgpu.com/challenges/batch-normalization)
formula. It reduces over rows per channel, divides variance by N, and adds
eps=1e-5 inside the square root. Local tolerances are atol=1e-4 and rtol=1e-4.
Cases cover both examples (using the formula rather than their rounded display
values), N/C boundaries, N=1, zero/constant channels, population variance,
independent channel statistics and affine parameters, epsilon-dominated small
variance, large offsets, active last rows/channels, and separately unaligned
input/gamma/beta/output pointers. A 10000-row constant-decimal regression checks
mean-accumulation drift against the analytic output beta. All inputs and affine
parameters remain in the challenge's stated ranges; zero/negative gamma,
alternative eps values, empty dimensions and in-place output are not tested.

Every case runs twice on the same allocations. The second call changes input
mean/variance and gamma/beta, with a fresh independent reference. All three
inputs are checked for bytewise preservation, output is poisoned before each
call, output guards detect overwrites, and input allocations have no suffix
padding for memcheck. Run `make run-batch-norm` (also in `make check`), use
`BN_ARGS="--case constant-decimal-long"` for an individual case, or
`BN_ARGS=--list-cases` to list quick cases without initializing CUDA.
`BN_ARGS=--large` runs the separate N=5000/10000,C=1024 cases, also in
`make check-full`; these are correctness checks, not timings.
`make sanitize-batch-norm` runs memcheck with leak checking and initcheck, accepts
`BN_ARGS`, and is included in `make sanitize`.

Initial validation on 2026-09-27 on A800: 51/52 quick cases and both large cases
pass. The constant-decimal regression fails: for a 10000-row channel of 0.1f,
gamma=10 and beta=0, the output is about 0.030723 instead of zero. The three
constant channels expose FP32 mean-accumulation drift, with a maximum output
error of about 9.49 across both calls. Memcheck and initcheck report zero memory
errors and memcheck reports zero leaked allocations, but the sanitizer target
correctly returns failure for the numerical mismatch. Both sm_80 and sm_75
compile; only sm_80 was run on hardware. The operator implementation was left
unchanged while adding these tests.

The 55-case `rms_norm_test` quick suite checks the
[RMS Normalization](https://leetgpu.com/challenges/rms-normalization) formula
against an independent CPU FP64 sum of squares over the entire 1D vector.
It divides by N and adds eps=1e-5 inside the square root, then applies scalar
gamma/beta without subtracting the mean. Local tolerances are atol=1e-5 and
rtol=1e-5. Both examples use the formula instead of their rounded displayed
outputs. Other cases cover scalar/float4/warp/block boundaries, every float4
component, one-to-three-element tails, many blocks, zeros, positive/negative
constants, mixed signs, epsilon-dominated inputs, different magnitudes across
blocks, a lone last element, affine parameter bounds and pointer offsets.

Each case makes two calls on the same input/output allocations, checking all
outputs, bytewise input preservation, output guards and NaN/finite output
poison. The second call reverses and rescales input and changes gamma/beta;
`repeat-identical` instead repeats identical inputs and parameters. Input
allocations have no suffix padding and remain 16-byte aligned, including the
four-float offset case; output offsets of one, two and three floats are tested.
Inputs and parameters remain in the challenge ranges. Arbitrarily unaligned
inputs, zero/negative gamma, alternative eps values, empty inputs, nonfinite
values and in-place operation are outside this test contract.

Run `make run-rms-norm` (also in `make check`), select a case with
`RMS_ARGS="--case example-1"`, or list names without initializing CUDA using
`RMS_ARGS=--list-cases`. `RMS_ARGS=--large` runs three separate cases at
N=99999/100000, including mixed magnitudes, also in `make check-full`.
`--large` can be combined with `--case NAME` or `--list-cases`. These are
correctness checks, not timings. `make sanitize-rms-norm` runs memcheck with
leak checks, initcheck, racecheck and synccheck; it accepts `RMS_ARGS`, is
included in `make sanitize`, and propagates functional and sanitizer failures.

Validated on 2026-09-28 on A800 GPU 3 after adding per-call initialization of
the temporary `rms` scalar before the atomic reduction: all 55 quick cases and
all three large cases pass, including repeated identical inputs and changed
inputs/parameters. Memcheck, initcheck and synccheck each report zero errors;
racecheck reports zero hazards, and memcheck reports zero leaked allocations.
All four sanitizer tools ran the complete quick suite. Large cases were
checked normally, without sanitizer instrumentation; the maximum absolute
error among those cases in this run was about 6.36e-6.

Both sm_80 and sm_75 builds pass; only sm_80 was run on hardware. No physical
T4 runtime or online submission is implied. Logs for this revision are in
`build/sm_80/rms_norm-validated-{quick,large,sanitize}.log`.

The 64-case `max_pooling_2d_test` quick suite covers both
[2D Max Pooling examples](https://leetgpu.com/challenges/2d-max-pooling), scalar
and identity windows, rectangular/single-row/single-column inputs, dimensions
around 16/32 boundaries, every kernel size and stride from 1 through 16,
nondivisible output sizes, overlapping and separated windows, all-negative/zero
inputs, repeated maxima, independent batch/channel planes, an active last
corner, N=100/C=512 limits, and separately unaligned input/output pointers.
The independent CPU oracle clips windows to the original input H/W, with floor
output dimensions and exact FP32 comparisons: max selects an input value, so
there is no reduction-rounding tolerance. Both published answers also check
the oracle itself.

The challenge does not explicitly specify the padding value. Local tests use
negative infinity following [standard MaxPool2d semantics](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html);
padding must not replace a negative maximum with zero. Cases use finite inputs,
positive output dimensions and `padding <= kernel_size/2`, so every window
contains input. Larger padding, all-padding windows, empty tensors, NaN/Inf
inputs and in-place operation are not covered. No claim about hidden platform
behavior for those ambiguous cases is made.

Each case runs twice on the same buffers; the second input is reversed and
affinely transformed. Checks include unchanged input bytes, NaN/finite output
poison, output guards and unpadded input tails. Run `make run-max-pooling-2d`
(also in `make check`), use `POOL_ARGS="--case example-1"` to isolate a case,
or `POOL_ARGS=--list-cases` to list cases without CUDA initialization.
`POOL_ARGS=--large` runs separate N=4,k=3,s=2,p=1 cases with
`C/H/W=8/1024/1023` and `512/63/65`, also included in `make check-full`.
`--large` can be combined with `--case NAME` or `--list-cases`. These are
correctness checks, not performance measurements. `make sanitize-max-pooling-2d`
runs memcheck with leak checking and initcheck, accepts `POOL_ARGS`, and is
included in `make sanitize`; functional or sanitizer failures return nonzero.

Validated on 2026-09-28 on A800 GPU 3 after correcting output dimensions,
thread bounds, padding and input addressing: all 64 quick cases and both
large cases pass. The quick suite also passes the Makefile sanitizer target:
memcheck and initcheck each report zero errors, with zero leaked allocations.
Input reads now use the original H/W, while output writes use the pooled
dimensions. Both sm_80 and sm_75 builds pass; only sm_80 was run on hardware.
The large cases were run for correctness, without sanitizer instrumentation.
No physical T4 run or online submission is implied. Logs for this revision
are in `build/sm_80/max_pooling_2d-validated-*.log`.

The 58-case `count_test` quick suite checks the
[Count Array Element](https://leetgpu.com/challenges/count-array-element)
operation with an exact CPU integer count. It covers both examples, scalar,
int4/warp/block boundaries, each int4 component independently, every position
of 1/2/3-element scalar tails, all/none/clustered matches, a lone match at the
end of a million-element array, input/K endpoints, aligned input offsets, and
independent four-byte output offsets. All inputs/K remain in [1,100000] and
N is positive. Input pointers remain 16-byte aligned for the current int4
kernel; arbitrary unaligned views and out-of-contract integers are excluded.
The statement's performance note gives K=501010, which contradicts its stated
range; the large tests follow the stated range and do not claim to reproduce
that benchmark's data.

Every case checks unchanged input bytes and output guards, with no padded
input suffix. Each final call changes both the target K and matching positions
on the same buffers. Most cases clear output before each call to isolate
counting/indexing. Two nonzero-output cases and two consecutive-call cases
separately test overwrite semantics. Those reuse cases make three calls without
clearing output after the first call; other cases make two calls. Run
`make run-count` (also in `make check`), select a case using
`COUNT_ARGS="--case consecutive-calls"`, or list names using
`COUNT_ARGS=--list-cases` without initializing CUDA.

`COUNT_ARGS=--large` runs two separate correctness regressions (also in
`make check-full`): an all-match count of 16,777,217, which cannot be stored
exactly in FP32, and 100 million elements with a periodic mixture of matches.
Both use the same CPU reference and changed-input second call; output is
cleared to isolate integer counting. `--large` can be combined with
`--case NAME` or `--list-cases`. These tests do not measure performance.
`make sanitize-count` runs memcheck, initcheck, racecheck and synccheck, accepts
`COUNT_ARGS`, and participates in `make sanitize`. It returns nonzero for
either incorrect results or sanitizer errors.

Initial validation on 2026-09-28 on A800 GPU 3: 54/58 quick cases and both
large cases pass. All four failures are output-overwrite/reuse regressions:
the unchanged implementation atomically adds to output without resetting it.
For example, two consecutive calls on `[1,2,3,4,1]`, K=1, produce 2 then 4
instead of 2 then 2. Memcheck/initcheck/synccheck report zero errors and
racecheck reports zero hazards, while the sanitizer target correctly fails
for the wrong counts. Both sm_80 and sm_75 compile; only sm_80 was run on
hardware. No full repository check or online submission is claimed. Logs are
in `build/sm_80/count-{quick,large,sanitize}.log`.

The 73-case `count3d_test` quick suite checks
[Count 3D Array Element](https://leetgpu.com/challenges/count-3d-array-element)
with an exact CPU count of P across all N*M*K elements. It covers both
examples, tiny flattened arrays, independent N/M/K boundaries through 1000,
all six permutations of a rectangular shape with identical flattened data,
P distinct from K, input/P endpoints, all/none matches, individual int4
components and scalar-tail positions, each axis's last slice, a lone match
at the final element, and aligned input/four-byte output offsets. Most final
calls change both P and matching positions; a control and the four overwrite/
reuse cases keep P=K to isolate output-reset errors from argument mixups.

Inputs are finite-range INT32 values in [1,100], dimensions are in [1,1000],
and input pointers remain 16-byte aligned for the current vectorized kernel.
Empty tensors and arbitrary unaligned input views are not tested. Input tails
are unpadded, all input bytes must remain unchanged, and output guards check
overwrites. Baseline cases clear output before each of two calls. Two cases
start with nonzero output; two others run three calls without clearing the
previous result. Run `make run-count3d` (also in `make check`), select a case
with `COUNT3D_ARGS="--case example-1"`, or list names with
`COUNT3D_ARGS=--list-cases` without initializing CUDA.

`COUNT3D_ARGS=--large` runs two separate correctness regressions, also in
`make check-full`: the statement's 500^3 shape (125M elements) using a
deterministic mixture, and the maximum 1000^3 shape (1B elements) with
999,999,999 matches on the first call and one on the second. The first count
cannot be represented exactly in FP32. CPU reference, input preservation and
changed-input second calls are retained, with output cleared each time.
These are correctness checks, not benchmark timings. Cases allocate inputs
one at a time; the maximum case uses approximately 4 GB of GPU storage and
8 GB of host storage including the preservation check. `--large` can be
combined with `--case NAME` or `--list-cases`; listing large cases does not
allocate their input arrays.
`make sanitize-count3d` runs memcheck, initcheck, racecheck and synccheck,
accepts `COUNT3D_ARGS`, and is included in `make sanitize`. Incorrect results
or sanitizer findings cause a nonzero exit status.

Validated on 2026-09-28 on A800 GPU 3 after the wrapper was updated to pass
P as the target: 69/73 quick cases and both 500^3/1000^3 cases pass. The four
failures are output-overwrite/reuse checks; all counting checks with cleared
output pass, including P != K. In `consecutive-calls`, the first two calls
produce 9 then 18 instead of 9 then 9. The current wrapper still does not
reset output before atomic additions. Memcheck/initcheck/synccheck report
zero errors, racecheck reports zero hazards, and the sanitizer target fails
for incorrect counts. Both sm_80 and sm_75 compile; only sm_80 was run on
hardware. Logs are in `build/sm_80/count3d-{quick,large,sanitize}.log`.

The 74-case `slice_sum_test` quick suite follows the
[Subarray Sum](https://leetgpu.com/challenges/subarray-sum) contract:
`sum(input[S..E])`, with **inclusive** zero-based endpoints. Its INT64 CPU
reference uses every selected element and requires exact INT32 output. Both
examples, singleton/full/interior slices, warp/block/vector boundaries,
every start offset modulo four, each scalar-tail position, distinct endpoint
values, out-of-slice sentinels, the final element, output offsets, and changed
ranges on reused allocations are covered. Values remain in [1,10], so every
expected sum is positive and at most 1,000,000,000. Empty slices, negative
inputs, arbitrary unaligned base allocations and aliasing are not tested.

The input base is cudaMalloc-aligned but `input+S` need not be; no suffix
padding hides overreads. All input bytes must be preserved and output guards
must remain unchanged. Baseline cases clear output and make two calls, with
all values changed to `11-value` on the final call. Two nonzero-output and
two consecutive-call cases check overwrite semantics; reuse cases make three
calls without clearing the previous result. Single-element overwrite/reuse
cases isolate output-reset failures from multi-element indexing errors.
`range-change` and `consecutive-blocks` also change S/E on the final call.
Launch errors mark a case as failed; unrecoverable execution errors still
terminate the process. Memcheck with `--destroy-on-device-error kernel` can
be used for diagnostics to terminate faulting kernels and continue later cases.

Run `make run-slice-sum` (also in `make check`), select individual cases with
`SLICE_SUM_ARGS="--case example-1"`, or list names using
`SLICE_SUM_ARGS=--list-cases` without initializing CUDA.
`SLICE_SUM_ARGS=--large` runs two separate 100M-element regressions, also
included in `make check-full`: a full slice with the maximum sum 1,000,000,000,
and an aligned interior slice of length 99,999,995. The odd interior sum
cannot be stored exactly in FP32. Both retain exact CPU references, changed
inputs, cleared outputs and preservation checks. `--large` can be combined
with `--case NAME` or `--list-cases`; large arrays are allocated only when
executing their case. These are correctness checks, not performance timings.
`make sanitize-slice-sum` runs memcheck, initcheck, racecheck and synccheck,
accepts `SLICE_SUM_ARGS`, and participates in `make sanitize`. Both numerical
failures and sanitizer findings cause a nonzero exit status.

Validated on 2026-09-28 on A800 GPU 3 after correcting the inclusive length
and scalar-loop indexing: 70/74 quick cases and both 100M-element cases pass.
All four failures concern overwriting nonzero/previous output. For the
single-element input `[7]`, consecutive calls return 7 then 14 instead of
7 then 7 because the wrapper does not reset output before atomic additions.
The complete quick suite reports zero memcheck/initcheck/synccheck errors
and zero racecheck hazards; sanitizer execution still returns failure for
the incorrect sums. Both sm_80 and sm_75 compile, with only sm_80 run on
hardware. Current logs are in `build/sm_80/slice_sum-{quick,large,sanitize}.log`.

The 71-case `slice_sum2d_test` quick suite follows
[2D Subarray Sum](https://leetgpu.com/challenges/2d-subarray-sum): row and
column endpoints are inclusive, and rows use the original matrix width M.
An independent nested-loop INT64 CPU reference checks exact INT32 results;
the two published answers also validate the oracle. Cases cover both
examples, independent rectangle-height/width boundaries, flattened areas
around warp/block boundaries, single-row/column matrices, all four matrix
corners, individually distinguished rectangle corners, original row strides,
out-of-region sentinels, last rows/columns, output offsets, and changed regions.
Inputs remain in [1,10] and dimensions in [1,10000], so sums fit INT32.
Empty/invalid regions, negative inputs and input/output aliasing are excluded.

Inputs have no allocation suffix padding and must remain bitwise unchanged;
output guards detect overwrites. Baseline cases clear output and run twice,
transforming inputs to `11-value` on the final call. Two cases start from
nonzero output and two reuse cases run three times without clearing previous
results; singletons isolate the output-reset requirement. Region changes also
alter the rectangle's shape/area on the same input/output allocations.
Run `make run-slice-sum2d` (also in `make check`), select a case with
`SLICE_SUM2D_ARGS="--case original-row-stride"`, or list cases using
`SLICE_SUM2D_ARGS=--list-cases` without initializing CUDA.

`SLICE_SUM2D_ARGS=--large` runs two separate 10000x10000 regressions, also in
`make check-full`: the full matrix with sum 1,000,000,000, and an interior
9999x9995 region with the odd sum 99,940,005 (not exactly representable in FP32).
Both use cleared output, independent references, changed-input second calls,
guards and input preservation. Large inputs allocate one case at a time;
`--large` also accepts `--case NAME` or `--list-cases`, and listing does not
allocate the large arrays. These are correctness checks, not timings.
`make sanitize-slice-sum2d` runs memcheck, initcheck, racecheck and synccheck,
accepts `SLICE_SUM2D_ARGS`, and participates in `make sanitize`. Functional
failures and sanitizer findings both produce a nonzero exit status.

Validated on 2026-09-28 on A800 GPU 3: 67/71 quick cases and both 10000x10000
cases pass. All cleared-output checks pass, including row strides and inclusive
boundaries. The four overwrite/reuse cases fail because output is not reset
before atomic additions: two consecutive calls on the single-element matrix
`[[7]]` return 7 then 14 instead of 7 then 7. Memcheck/initcheck/synccheck
report zero errors and racecheck reports zero hazards; the sanitizer target
still returns failure for incorrect results. Both sm_80 and sm_75 compile,
with only sm_80 run on hardware. The operator implementation was not changed
while adding tests. Logs are in `build/sm_80/slice_sum2d-{quick,large,sanitize}.log`.

The 87-case `slice_sum3d_test` quick suite follows
[3D Subarray Sum](https://leetgpu.com/challenges/3d-subarray-sum): all six
endpoints are inclusive, and input row/depth strides are the original K and
M*K. An independent three-loop INT64 CPU reference checks exact INT32 sums;
the two published answers also validate the oracle. Coverage includes both
examples, scalar/axis-line inputs, independent depth/height/width boundaries,
flattened volumes around warp/block boundaries, distinct values per depth,
original strides, out-of-region sentinels, all eight tensor corners, individually
distinguished cuboid corners, final depth/row/column slices, output offsets,
and changed cuboids on reused allocations. Dimensions stay in [1,500], values
in [1,10], and every expected sum fits INT32. Empty/invalid regions, negative
inputs and aliasing are outside the tested contract.

No allocation suffix padding hides final-layer overreads. All input bytes
must be preserved and output guards must remain unchanged. Baseline cases
clear output and run twice, with inputs transformed to `11-value` on the last
call. Two nonzero-output cases and two three-call reuse cases require overwrite
semantics; singleton cases isolate output-reset failures from 3D indexing.
`region-change` and `consecutive-blocks` also change the cuboid on the final
call. Run `make run-slice-sum3d` (also in `make check`), select a case using
`SLICE_SUM3D_ARGS="--case distinct-depths"`, or list cases with
`SLICE_SUM3D_ARGS=--list-cases` without initializing CUDA.

`SLICE_SUM3D_ARGS=--large` runs two separate 500^3 regressions, also registered
in `make check-full`: a full tensor with the maximum sum 1,250,000,000, and an
interior 499x497x495 cuboid with the odd sum 122,761,485 (not exactly FP32).
They retain exact CPU references, cleared outputs, changed-input second calls,
input preservation and output guards. Large arrays allocate one case at a
time; `--large` can be combined with `--case NAME` or `--list-cases`, and
listing does not allocate the arrays. These are correctness checks, not timings.
`make sanitize-slice-sum3d` runs memcheck, initcheck, racecheck and synccheck,
accepts `SLICE_SUM3D_ARGS`, and participates in `make sanitize`. Functional
failures and sanitizer findings cause nonzero exit status.

Validation on 2026-09-28 on A800 GPU 3 after the row-index fix: the quick
suite reports 83/87 passing, and both 500^3 large cases pass. The row
coordinate now wraps within each depth using `(i / size.x) % size.y`.
The remaining failures are `overwrite-singleton`, `overwrite-interior`,
`consecutive-calls` and `consecutive-blocks`: `solve` does not clear output
before `atomicAdd`. For example, the singleton sum 7 returns 24 when output
starts at 17, and repeated calls return 7 then 14 instead of 7 then 7.

The full quick suite runs under all four sanitizer tools: memcheck, initcheck
and synccheck report zero errors, and racecheck reports zero hazards. The
sanitizer target still returns failure for the four incorrect results. Both
sm_80 and sm_75 compile; only sm_80 was run on hardware. This verification
did not change the operator implementation. Current logs are in
`build/sm_80/slice_sum3d-{quick,large,sanitize}.log`.

`sanitize` runs memory checks on RMS normalization, 1D/2D/3D subarray sums, 1D/3D integer counting, max pooling, batch normalization, nearest neighbor, GEMM, matrix power, batched and INT8 matrix multiplication, blur,
categorical cross entropy, MSE, top-k, interleave, sigmoid and Monte Carlo integration,
and synchronization checks on subarray sums, counting, the two Cooperative Groups loss reductions and top-k. It is a
selected set, not a sanitizer audit of every operator.

## Optional Triton GEMM

The optional Triton GEMM runner consumes `gemm_cublas_bench --list-cases`,
which exports the CUDA suite's 57 cases as JSON lines without initializing a GPU.
Run `make check-gemm-triton PYTHON=/path/to/python`, or include it in the main
suite with `make check WITH_TRITON=1 PYTHON=/path/to/python`. The runner uses an
independent CPU double reference, tests two consecutive calls, input preservation
and output guards, and checks all seven autotune candidates on representative
cases. `make sanitize-gemm-triton` selects tail, misalignment and K=0 cases for
memcheck/racecheck/synccheck; `WITH_TRITON=1` also includes it in `make sanitize`.
See [Triton GEMM](gemm-triton.md) for the separate performance comparison target
and its Python submission overhead.

## Matrix-vector dispatch

`mv_bench --check-only` checks 28 shapes through `solve`, the block-per-row
kernel, and two warp-per-row block sizes. Eleven additional shapes cross the
dispatch boundaries at M=64/1024 and N=1024/2048/4096, including partial rows and
columns. All four variants retain the CPU double reference, output guards, and
two consecutive calls without clearing output. The tolerance remains
`1e-5 + 2e-6 * sum(abs(A*x))`.

The selector uses a 256-thread block per row for N >= 1024 when M <= 64,
N >= 2048 when 65 <= M <= 1024, and N >= 4096 for larger M. Shorter rows use
one warp per row in 256-thread blocks. The rule is an A800 heuristic, not an
optimal threshold for every shape or GPU. Both kernels remain available for
direct comparison in `make run-mv`.

Measured on 2026-09-26 on NVIDIA A800 80GB PCIe, CUDA Toolkit 12.6.20,
`-O3 -std=c++14 -arch=sm_80`, on an otherwise idle GPU. Before/after sweeps
covered 32 shapes: M in {16,64,256,1024,4096}, N in
{128,256,512,1024,2048,4096}, plus 4096x64 and 16x16384. Each variant used
10 warmup launches and the median per-launch CUDA Event time of five batches
of 500 launches, reusing the same buffers. Allocation, copies, and the CPU
reference are excluded; launch submission gaps can affect small timings.

| M x N | Previous warp-only solve (us) | Dispatched solve (us) | Selected kernel |
| --- | ---: | ---: | --- |
| 16 x 1024 | 3.533 | 3.178 | block |
| 64 x 1024 | 3.535 | 3.437 | block |
| 256 x 2048 | 4.927 | 4.483 | block |
| 1024 x 2048 | 6.281 | 5.829 | block |
| 4096 x 2048 | 16.114 | 16.255 | warp |
| 4096 x 64 | 3.494 | 3.598 | warp |
| 4096 x 1024 | 7.381 | 7.516 | warp |
| 1024 x 4096 | 9.347 | 7.631 | block |
| 4096 x 4096 | 46.232 | 42.082 | block |
| 16 x 16384 | 16.835 | 5.503 | block |

All 32 measured shapes passed correctness before and after the change. Small
timing differences are not evidence of a universal crossover point.
The complete `make check` passed. The 28-shape suite also passed Compute
Sanitizer memcheck and synccheck with zero errors on the A800. The `sm_75`
build passed compilation; these results
do not include a physical T4 runtime or a T4-tuned performance threshold.

## Architecture and device selection

```bash
make -j2 NVCC_ARCH=sm_75
CUDA_VISIBLE_DEVICES=0 make check NVCC_ARCH=sm_75
```

A T4 needs `sm_75`; A100/A800 use `sm_80`. Architecture-specific output directories
prevent accidental reuse of a binary for a different GPU. Set `CUDA_VISIBLE_DEVICES`
to select an available physical device; tests use logical device 0.

The build workflow in `.github/workflows/build.yml` has no GPU. It checks compilation
and linking only. A successful workflow is not evidence of numerical correctness
or T4 runtime validation.

## Measurement scope

- Matrix-vector and GEMM benchmarks warm up and report the median of five batches
  of repeated launches. They reuse input buffers and exclude host/device copies.
  GEMM timing uses `beta=0` to prevent repeated accumulation into C.
  `make check-gemm` validates scalar, tiled, all ten WMMA variants, and cuBLAS;
  `make run-gemm-compare GEMM_ARGS="1024 1024 1024 100"` compares their timings.
  The 52-case GEMM suite also checks misaligned A/B base pointers on complete
  tiles, one/two/three full K chunks, multiple output blocks, independently partial
  M/N/K dimensions, K=0, and unaligned C with aligned A/B. Wide 1024x2048 outputs
  with K=32/96 cover larger block/warp tile builds, nontrivial alpha/beta, and
  aligned/unaligned output stores. Additional two/four/five/seven/eight-chunk
  cases cover short prologues, ring wraparound, and drain for multistage input
  buffers. Three/five/seven BK=64 chunks also exercise ring wraparound/drain,
  nontrivial alpha/beta, and unaligned C. The 96x128 tile is checked at its
  192-block automatic-dispatch boundary with one/three K chunks, alpha/beta,
  and unaligned C; forced BM=96 builds also cover a non-64-row five-chunk case.
  Alpha/beta checks vary each scalar independently and initialize C with NaNs
  for beta=0 on native, tail, K=0, and unaligned-output paths, including beta=-0.
  The reference ignores old C when beta=0.
  Four additional 256x256 cases cover one/four/five/seven K chunks, general
  alpha/beta, NaN old C, and unaligned C for larger tiles. The forced
  `gemm_wmma_tiled_pipeline_large_dynamic_test` runs the same suite using
  a 128x256 block, 32x64 warp, four stages, and 96 KiB dynamic shared memory.
  It is included in `check-gemm`, `check`, and all three sanitizer tools;
  it is separate from the fourteen default benchmark implementations.
  The scheduled version adds forced 128x128/BK32 three- and four-stage tests,
  and a 96 KiB 128x256/BK64 two-stage test with 16 warps. Five further cases
  cover BK64 one/three/nine-chunk drains and the two automatic grid bands.
  The shared GEMM suite has 57 cases; all forced tests participate in check
  and memcheck/racecheck/synccheck. Large-shape optional `*_perf` targets use
  `benchmarks/gemm.cu` with a cuBLAS FP32-reduction reference; they supplement
  the independent CPU checks rather than replacing them.
  `sanitize` runs memcheck,
  racecheck, and synccheck over the complete suite for all nine pipelined versions.
  All use 256-byte-aligned benchmark buffers, with output guards and a separate
  unaligned correctness case. cuBLAS handle creation happens before timing.
  CUDA-event batches include any host submission gaps between GPU operations.
- Attention, softmax, and convolution benchmarks also report wall-clock timings.
  A wrapper may include allocation, deallocation, or synchronization; read the
  printed fields and wrapper before comparing with a kernel-only number.
- Sum compares the manual, Cooperative Groups, and CUB wrappers. CUB caches its
  workspace; the CG implementation allocates per call. These are wrapper timings,
  not a controlled measurement of reduction instructions alone.
- Maximum and sum use simple reference inputs and are basic benchmark checks, not
  exhaustive numerical validation suites.

Record GPU model, toolkit, compiler flags, input shapes, repetitions, cache reuse,
and any competing GPU work with benchmark results. Small kernels can be dominated
by launch overhead; large inputs can be dominated by memory bandwidth. Recheck
correctness whenever an optimization changes accumulation order or precision.

## Example custom runs

```bash
build/sm_80/reduce_bench 1048576 20
build/sm_80/max_bench 1048576 20
build/sm_80/softmax_bench 1025 10 3
build/sm_80/attention_bench 17 33 16 10 3
build/sm_80/conv2d_bench 17 35 3 5 10 3
build/sm_80/conv3d_bench 9 11 13 3 3 3 10 3
build/sm_80/mv_bench --check-only
build/sm_80/gemm_bench --check-only
build/sm_80/gauss_blur_test 17 35 2 4
build/sm_80/mse_test 50000000
make run-swiglu
make run-interleave
make run-conv1d
```

Known empty-input conventions differ: categorical cross entropy leaves output
unchanged; MSE writes zero; GEMM and blur with empty output shapes do no work.
Tests document these explicitly rather than inventing a universal tensor contract.
