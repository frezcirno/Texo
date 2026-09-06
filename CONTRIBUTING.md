# Contributing

This is a learning repository. Prefer a small, understandable implementation and a
reproducible comparison over an unexplained optimization.

1. Keep standalone kernels in `src/`, tests in `tests/`, and incomplete exercises
   in `experimental/`. Use snake_case filenames. Standalone `solve` entry points
   are intentional; keep comparison symbol renaming in the Makefile.
2. Describe input/output shapes, dtype, padding, alignment requirements, and empty
   input behavior. Do not silently change an existing operator's contract.
3. Add CPU-reference checks for the behavior being changed. Test partial blocks,
   repeated calls, and numerical tolerances; return nonzero on failures.
4. Run `make check`. Use `make check-full` for reduction precision changes and
   `make sanitize` for memory/synchronization changes. Record hardware and toolkit.
5. For performance changes, include before/after timings and exact shapes. State
   whether allocations, transfers, and synchronization are included.

Use two-space indentation in C++/CUDA and tabs for Makefile recipes. Avoid unrelated
formatting changes. Do not commit generated binaries, profiler dumps, credentials,
or local machine paths. Preserve attribution and license notices for external code.
