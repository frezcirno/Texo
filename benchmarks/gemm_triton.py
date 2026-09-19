"""Compare Triton, the selected CUDA solve, and cuBLAS using identical buffers.

CUDA-event batches include Python/ctypes launch gaps. These are warmed API
latencies, not isolated kernel durations. JIT/autotuning, reference generation,
allocations, transfers and result validation are outside the measured interval.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gemm_support import (NativeGemm, check_close, check_device, check_guards,
                          guarded_tensor, load_triton, selected_triton_config, torch, triton)


DEFAULT_SHAPES = [(n, n, n) for n in (1024, 1536, 2048, 2304, 3072, 4096, 6144, 8192)] + [
    (2048, 4096, 1024), (4096, 2048, 1024), (2048, 2048, 128), (384, 6144, 96)]


def positive_int(text):
    value = int(text)
    if value <= 0 or value > 2147483647:
        raise argparse.ArgumentTypeError("expected an integer in [1, INT_MAX]")
    return value


def time_call(call, iterations, batches):
    times = []
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(batches):
        start.record()
        for _ in range(iterations):
            call()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop) * 1000 / iterations)
    return statistics.median(times)


def compare_shape(module, native, shape, args, reference_modules=None):
    reference_modules = reference_modules or {}
    m, n, k = shape
    torch.manual_seed(12345)
    a = torch.empty((m, k), dtype=torch.float16, device="cuda").uniform_(-1, 1)
    b = torch.empty((k, n), dtype=torch.float16, device="cuda").uniform_(-1, 1)
    storage, c = guarded_tensor(torch.full((m, n), float("nan"),
                                          dtype=torch.float16, device="cuda"))
    calls = {name: impl.prepare(a, b, c, m, n, k) for name, impl in native.items()}
    calls["triton"] = lambda: module.solve(a, b, c, m, n, k, 1.0, 0.0)
    for name, reference in reference_modules.items():
        calls[name] = lambda impl=reference: impl.solve(a, b, c, m, n, k, 1.0, 0.0)
    calls["cublas"]()
    native["cublas"].check_error()
    torch.cuda.synchronize()
    check_guards(storage, m * n)
    reference = c.clone()
    errors = {}
    for name, call in calls.items():
        c.fill_(float("nan"))
        # First Triton call includes JIT and autotuning, outside all timings.
        call()
        torch.cuda.synchronize()
        errors[name] = check_close(c, reference)
        check_guards(storage, m * n)
    for impl in native.values():
        impl.check_error()
    selected = {"triton": selected_triton_config(module, m, n, k)}
    for name, reference in reference_modules.items():
        selected[name] = selected_triton_config(reference, m, n, k)
    if args.verify_only:
        print(f"PASS {m}x{n}x{k} max_errors={errors} config={selected} (no timing)", flush=True)
        return []
    for call in calls.values():
        for _ in range(5):
            call()
    torch.cuda.synchronize()
    order = ["triton", *reference_modules, "cuda", "cublas"]
    samples = {name: [] for name in order}
    for round_index in range(args.rounds):
        for name in order if round_index % 2 == 0 else order[::-1]:
            samples[name].append(time_call(calls[name], args.iterations, args.batches))
    for impl in native.values():
        impl.check_error()
    rows = []
    for name in order:
        us = statistics.mean(samples[name])
        rows.append(dict(M=m, N=n, K=k, implementation=name, mean_median_us=us,
                         round_medians_us=json.dumps(samples[name]), tflops=2 * m * n * k / us / 1e6,
                         max_abs_error=errors[name],
                         triton_config=json.dumps(selected[name]) if name in selected else ""))
    values = {r["implementation"]: r["mean_median_us"] for r in rows}
    for row in rows:
        row["vs_cuda_time_percent"] = 100 * (row["mean_median_us"] / values["cuda"] - 1)
        row["vs_cublas_time_percent"] = 100 * (row["mean_median_us"] / values["cublas"] - 1)
        for reference_name in reference_modules:
            row[f"vs_{reference_name}_time_percent"] = 100 * (row["mean_median_us"] / values[reference_name] - 1)
    print(f"{m}x{n}x{k}: " + ", ".join(f"{name}={values[name]:.3f} us" for name in order) + "; " +
          f"Triton/CUDA={values['triton'] / values['cuda']:.3f}x", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=Path("build/sm_80"))
    parser.add_argument("--cuda", default="gemm_wmma_tiled_pipeline_schedule")
    parser.add_argument("--source", type=Path, default=Path("src/gemm.triton.py"))
    parser.add_argument("--reference-source", type=Path, action="append", default=[],
                        help="also time another Triton version; may be repeated")
    parser.add_argument("--shape", type=positive_int, nargs=3, action="append", metavar=("M", "N", "K"))
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument("--batches", type=positive_int, default=5)
    parser.add_argument("--rounds", type=positive_int, default=2)
    parser.add_argument("--output", type=Path, help="CSV path, with a sibling metadata JSON")
    parser.add_argument("--verify-only", action="store_true", help="validate without running timing batches")
    args = parser.parse_args()
    for m, n, k in args.shape or DEFAULT_SHAPES:
        if max(m * n, m * k, k * n) > 2147483647:
            parser.error("shape exceeds the native kernels' int indexing contract")
    check_device()
    native = {"cuda": NativeGemm(args.bin_dir / f"{args.cuda}.so"),
              "cublas": NativeGemm(args.bin_dir / "gemm_cublas.so")}
    module = load_triton(args.source)
    reference_paths = {"triton_reference" + (f"_{i + 1}" if i else ""): path
                       for i, path in enumerate(args.reference_source)}
    reference_modules = {name: load_triton(path) for name, path in reference_paths.items()}
    print(f"Triton source: {args.source}; references: {reference_paths}", flush=True)
    print("FP16/FP32 accumulation, alpha=1 beta=0; identical input/output buffers.", flush=True)
    print("Timing: warmed CUDA events including Python/ctypes submission gaps; "
          "JIT/autotuning, allocations/copies and validation excluded.", flush=True)
    metadata = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                    triton=triton.__version__, torch_cuda=torch.version.cuda,
                    native_versions=native["cublas"].versions(), cuda=args.cuda,
                    triton_source=str(args.source),
                    triton_reference_sources={name: str(path) for name, path in reference_paths.items()},
                    iterations=args.iterations, batches=args.batches, rounds=args.rounds,
                    timing="mean of per-round median CUDA-event batches, including Python/ctypes launch gaps",
                    alpha=1, beta=0, seed=12345)
    rows = []
    with torch.cuda.stream(torch.cuda.default_stream()):
        for shape in args.shape or DEFAULT_SHAPES:
            rows.extend(compare_shape(module, native, shape, args, reference_modules))
    if args.output and rows:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
