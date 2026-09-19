"""Run the CUDA suite's exported cases against the standalone Triton solve."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gemm_support import (check_close, check_device, check_guards, guarded_tensor,
                          fixed_triton_solve, load_triton, torch, triton_configs)


def run_case(case, solve):
    m, n, k = (case[key] for key in ("m", "n", "k"))
    alpha, beta = case["alpha"], case["beta"]
    generator = torch.Generator().manual_seed(12345)

    def values(shape):
        if case["ones"]:
            return torch.ones(shape, dtype=torch.float16)
        return (torch.rand(shape, generator=generator) * 2 - 1).half()

    a, b = values((m, k)), values((k, n))
    initial = (torch.rand((m, n), generator=generator) * 2 - 1).half()
    if case["nan_c"]:
        initial.fill_(float("nan"))
    # Use actual FP16 input values and independent CPU double accumulation.
    product = a.double() @ b.double()
    a_storage, da = guarded_tensor(a, case["a_offset"])
    b_storage, db = guarded_tensor(b, case["b_offset"])
    c_storage, dc = guarded_tensor(initial, case["c_offset"])
    a_before, b_before = a_storage.cpu(), b_storage.cpu()
    maximum = 0.0
    previous = initial
    for _ in range(2):
        solve(da, db, dc, m, n, k, alpha, beta)
        torch.cuda.synchronize()
        expected = alpha * product
        if beta != 0:
            expected = expected + beta * previous.double()
        # Match the CUDA suite's float-to-half reference conversion.
        expected = expected.float().half()
        actual = dc.cpu()
        maximum = max(maximum, check_close(actual, expected))
        check_guards(c_storage, m * n, case["c_offset"])
        if not torch.equal(a_storage.cpu(), a_before) or not torch.equal(b_storage.cpu(), b_before):
            raise AssertionError("input storage modified")
        previous = actual
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("src/gemm.triton.py"))
    parser.add_argument("--cases-binary", type=Path, required=True,
                        help="CUDA GEMM test executable supporting --list-cases")
    parser.add_argument("--case", action="append", dest="names", help="select a case by name")
    parser.add_argument("--all-configs", action="store_true",
                        help="also validate every autotune candidate on representative cases")
    parser.add_argument("--fixed-only", action="store_true",
                        help="check every candidate on selected cases without autotuning (for sanitizers)")
    args = parser.parse_args()
    if args.fixed_only and not args.all_configs:
        parser.error("--fixed-only requires --all-configs")
    output = subprocess.check_output([str(args.cases_binary.resolve()), "--list-cases"], text=True)
    cases = [json.loads(line) for line in output.splitlines()]
    if args.names:
        unknown = set(args.names) - {case["name"] for case in cases}
        if unknown:
            parser.error(f"unknown cases: {sorted(unknown)}")
        cases = [case for case in cases if case["name"] in args.names]
    check_device()
    module = load_triton(args.source)
    print(f"Source: {args.source}", flush=True)
    count = 0
    print("FP16 inputs/output; CPU double reference; atol=0.01 rtol=0.01; two calls/case.", flush=True)
    with torch.cuda.stream(torch.cuda.default_stream()):
        if not args.fixed_only:
            for case in cases:
                error = run_case(case, module.solve)
                count += 1
                print(f"{case['name']:<24} PASS max_abs_error={error:g}", flush=True)
        # Autotuning only checks the selected result. Check rejected candidates
        # too, including a full tile, tails, NaN C, K=0 and misaligned A/B.
        forced_names = {"multi-tile-tail", "unaligned-A-B", "full-output-K-zero",
                        "large-five-nan", "alpha-zero", "single-row", "random-long-K"}
        if args.all_configs:
            for index, config in enumerate(triton_configs(module)):
                for case in cases:
                    if args.fixed_only or case["name"] in forced_names:
                        error = run_case(case, fixed_triton_solve(module, config))
                        count += 1
                        print(f"config={index} {case['name']:<24} PASS max_abs_error={error:g}", flush=True)
    print(f"ALL PASSED ({count} checks)", flush=True)


if __name__ == "__main__":
    main()
