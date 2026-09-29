#!/usr/bin/env python3
"""Compare the checked-out LeetGPU reference, an integer oracle, and CUDA solve.

The upstream challenge is imported without changing it. This diagnostic is
separate from the normal tests: a disagreement with the upstream reference is
reported, never treated as the expected mathematical answer.
"""

import argparse
import ctypes
import importlib.util
from pathlib import Path
import subprocess
import sys

import torch


def window_sums(values, window):
    current = sum(values[:window])
    sums = [current]
    for end in range(window, len(values)):
        current += values[end] - values[end - window]
        sums.append(current)
    return sums


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path,
                        default=Path(__file__).resolve().parents[2] / "leetgpu-challenges")
    parser.add_argument("--library", type=Path, required=True,
                        help="Shared library built from src/max_subarray_sum.cu")
    args = parser.parse_args()
    repo = args.repo.resolve()
    challenge_path = repo / "challenges/medium/51_max_subarray_sum/challenge.py"
    sys.path.insert(0, str(repo / "challenges"))
    spec = importlib.util.spec_from_file_location("leetgpu_max_subarray_sum", challenge_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    challenge = module.Challenge(device="cuda")
    library = ctypes.CDLL(str(args.library.resolve()))
    solve = library.solve
    solve.argtypes = [entry[0] for entry in challenge.get_solve_signature().values()]
    solve.restype = None
    ptr = ctypes.POINTER(ctypes.c_int)

    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    print(f"upstream={repo} revision={revision}", flush=True)
    print(f"torch={torch.__version__} GPU={torch.cuda.get_device_name()} seed=42", flush=True)
    torch.manual_seed(42)
    cases = [("official-example", challenge.generate_example_test())]
    cases += [(f"official-functional-{i + 1}", case)
              for i, case in enumerate(challenge.generate_functional_test())]
    cases.append(("official-performance", challenge.generate_performance_test()))

    def add_case(name, values, window):
        cases.append((name, {
            "input": torch.tensor(values, dtype=torch.int32, device="cuda"),
            "output": torch.empty(1, dtype=torch.int32, device="cuda"),
            "N": len(values), "window_size": window,
        }))

    add_case("first-window-positive", [9, -8], 1)
    add_case("first-window-negative", [-1, -9], 1)
    add_case("single-window", [9, -8], 2)
    # Constructed data, NOT the platform's unavailable full input. The first
    # window sums to 285, the second to 268, and every subsequent one is smaller.
    values = [0] * 50000
    values[0] = 9
    values[1:28] = [10] * 27
    values[28] = 6
    values[25000:] = [-10] * 25000
    values[25000] = -8
    add_case("constructed-285-vs-268", values, 25000)

    cuda_failures = 0
    reference_mismatches = 0
    for name, case in cases:
        values = case["input"].cpu().tolist()
        sums = window_sums(values, case["window_size"])
        expected = max(sums)
        # Execute the original reference on CUDA with the original signature.
        challenge.reference_impl(**case)
        reference = case["output"].item()
        reference_mismatches += reference != expected
        actual = []
        case["output"].fill_(torch.iinfo(torch.int32).max)
        torch.cuda.synchronize()
        for _ in range(3):
            # Reuse the output without a reset between calls.
            solve(ctypes.cast(case["input"].data_ptr(), ptr),
                  ctypes.cast(case["output"].data_ptr(), ptr),
                  case["N"], case["window_size"])
            torch.cuda.synchronize()
            actual.append(case["output"].item())
        if any(result != expected for result in actual):
            cuda_failures += 1
        if case["input"].cpu().tolist() != values:
            raise AssertionError(f"{name}: CUDA modified the input")
        excluding_first = max(sums[1:]) if len(sums) > 1 else sums[0]
        print(f"{name}: N={case['N']} W={case['window_size']} "
              f"oracle={expected} upstream={reference} CUDA={actual} "
              f"first={sums[0]} best-after-first={excluding_first}", flush=True)

    print(f"CUDA: {len(cases) - cuda_failures}/{len(cases)} cases passed "
          f"(3 calls per case); upstream/oracle mismatches={reference_mismatches}", flush=True)
    return int(cuda_failures != 0)


if __name__ == "__main__":
    sys.exit(main())
