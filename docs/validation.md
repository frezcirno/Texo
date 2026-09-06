# Validation record

Repository preparation on 2026-09-06 used:

- Linux with CUDA Toolkit 12.6 (`nvcc V12.6.20`).
- NVIDIA A800 80GB PCIe for runtime checks.
- `-O3 -std=c++14 -arch=sm_80 -Xcompiler -Wall` for runtime binaries.

Completed checks:

| Command | Result / scope |
| --- | --- |
| `make all compile-kernels` | Tests linked and standalone sources compiled for `sm_80` |
| `make all compile-kernels NVCC_ARCH=sm_75` | Compilation/linking for T4; no T4 runtime available |
| `make check-full` | Passed, including the 50-million-element MSE cases |
| `make sanitize` | All selected memory/synchronization checks reported zero errors |
| `git diff --check` | No whitespace errors |

The blur suite includes empty images and even-size kernels with the documented
integer anchor. The GEMM suite includes empty output dimensions, nonzero beta,
and a long all-ones accumulation that previously exposed FP16 accumulation error.

GitHub Actions is configured for compilation only. The workflow itself has not
been run on GitHub as part of this local preparation. These results are a snapshot,
not a guarantee for future commits, other input ranges, or other GPUs.
