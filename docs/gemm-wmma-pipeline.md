# Double-buffered WMMA GEMM

`src/gemm_wmma_tiled_pipeline.cu` stages the next K chunk while computing the
current chunk with WMMA. It keeps the default BM=64, BN=64, BK=32, WM=32, WN=32,
SKEW=16 configuration of `gemm_wmma_tiled.cu`: four warps per block, with four
16x16 FP32 accumulator fragments per warp.

The contract is row-major FP16 A[M,K], B[K,N], C[M,N], with FP32 accumulation and
`C = alpha * A * B + beta * C`. Dimensions are nonnegative; M/N=0 is a no-op,
and K=0 applies the epilogue to a zero accumulator. Beta=0 does not read C.
Input and output pointers need only half alignment. Address arithmetic uses
`size_t`, and output tiles use a flattened grid with a grid-stride loop, avoiding
the 65535-block grid.y limit on tall matrices. Template overrides have the same
names as in the synchronous tiled implementation. Their shared-memory allocation
must fit the target GPU.

## Loading and synchronization

Each thread owns disjoint groups of eight half elements in the logical A/B tile.
On sm_80+, complete groups whose source address is 16-byte aligned use
`__pipeline_memcpy_async(..., 16)`. Shared rows and group starts preserve the
required alignment. Incomplete groups, unaligned sources, and sm_75 builds use
scalar loads and zero-fill invalid elements. This handles both unaligned base
pointers and row strides such as K=67 or N=129.

There are two A buffers and two B buffers; the accumulator is not duplicated.

1. Load tile 0, commit, wait, then synchronize the block.
2. Enqueue tile t+1 into buffer `(t & 1) ^ 1`, if it exists.
3. Compute tile t from buffer `t & 1` while that copy is in flight.
4. Wait for each thread's submitted copies, then synchronize the block.
5. Repeat, with no extra prefetch on the final iteration.

The block barrier after the wait has two purposes: every thread's next input
data must be ready, and every warp must finish reading the current buffer before
it can be overwritten. Scalar fallback stores are also ordered by that barrier.
`__syncwarp()` before commit reconverges lanes after their bounds/alignment
branches. Each warp retains its own output scratch tile and warp barriers.
See NVIDIA's [pipeline primitive requirements](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#pipeline-primitives-interface).

## Validation

Validated with CUDA 12.6.20 on NVIDIA A800 80GB PCIe, physical GPU 4:

- `make check` passed, including all six GEMM implementations.
- `make sanitize` passed. The pipeline executable's complete 21-case suite passed
  memcheck, racecheck, and synccheck with zero errors or hazards.
- Cases include K=0, one/two/three/many K chunks, partial M/N/K tiles, independently
  misaligned A and B pointers, unaligned C, nontrivial alpha/beta, and repeated calls.
- The custom 4194241x1x1 case passed, exercising a flattened grid with 65536 blocks.
- Both sm_80 and sm_75 compiled. T4 hardware was not available for runtime tests.
  Running the sm_75 executable on this A800 attempted PTX JIT, which the installed
  driver rejected as an unsupported toolchain. To test the synchronous branch,
  `-gencode arch=compute_75,code=sm_80` generated native A800 code from that branch:
  all 21 cases passed, and its disassembly contained no LDGSTS instructions.

The normal sm_80 binary contains `LDGSTS.E.BYPASS.128`, `LDGDEPBAR`, `DEPBAR`,
and `HMMA.16816.F32` instructions. Resource usage from `cuobjdump`:

| Resource | Original tiled | Pipeline |
|---|---:|---:|
| Registers/thread | 96 | 62 |
| Static shared memory/block | 15360 bytes | 26624 bytes |
| Local memory | 0 | 0 |

## Timings

Measured on 2026-09-13, physical GPU 5 (A800 80GB PCIe), CUDA 12.6.20,
`nvcc -O3 -std=c++14 -arch=sm_80 -Xcompiler -Wall`. Both variants used the
unmodified CPU-reference loop in `tests/gemm.cu`, alpha=1, beta=0, aligned buffers,
five warmups, and the median of five batches of 100 calls. CUDA events exclude
allocations, transfers, and CPU validation; they include GPU execution and any
host submission gaps. Each variant ran serially on that GPU and passed its
CPU-reference checks. Clocks were not locked.

| M x N x K | Original tiled (us) | Pipeline (us) | Speedup |
|---|---:|---:|---:|
| 128 x 128 x 128 | 15.340 | 8.468 | 1.81x |
| 512 x 512 x 512 | 49.111 | 18.954 | 2.59x |
| 1024 x 1024 x 1024 | 118.518 | 43.878 | 2.70x |
| 1009 x 513 x 257 | 33.055 | 27.843 | 1.19x |

These gains include overlapping copies with computation, wider input loads,
changed register allocation, fewer block barriers, and the flattened block
mapping. A separate experiment would be needed to quantify each contribution.
The odd-stride case uses scalar fallback for many groups. Larger shared-memory
usage and limited work on small or thin shapes can restrict the benefit.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=4 make check
CUDA_VISIBLE_DEVICES=4 make sanitize
CUDA_VISIBLE_DEVICES=5 make run-gemm-wmma-tiled-pipeline GEMM_ARGS="1024 1024 1024 100"
CUDA_VISIBLE_DEVICES=5 build/sm_80/gemm_wmma_tiled_bench 1024 1024 1024 100

# The pipeline is included in both profiler targets.
CUDA_VISIBLE_DEVICES=5 make nsys-gemm NSYS_GEMMS="gemm_wmma_tiled gemm_wmma_tiled_pipeline"
# Requires performance-counter access, as described in the README.
CUDA_VISIBLE_DEVICES=5 make ncu-gemm NCU_GEMMS="gemm_wmma_tiled gemm_wmma_tiled_pipeline"
```
