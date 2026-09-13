"""Summarize wide CSV files exported by `ncu --page raw --csv`.

The first row contains metric names, the second units, and subsequent rows
contain one captured kernel launch each. Missing metrics (e.g. a basic capture)
remain empty in comparison.csv and display as N/A. No third-party dependencies.
"""
import argparse
import csv
import math
from pathlib import Path


METRICS = {
    "duration_us": "gpu__time_duration.sum",
    "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "tensor_active_pct": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "theoretical_occupancy_pct": "sm__maximum_warps_per_active_cycle_pct",
    "dram_throughput_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "shared_load_bank_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "shared_store_bank_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "registers_per_thread": "launch__registers_per_thread",
    "long_scoreboard_cycles_per_instruction": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "lg_throttle_cycles_per_instruction": "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "pm_dropped_samples": "profiler__pmsampler_dropped_samples",
}
TIME_TO_US = {
    "second": 1e6, "s": 1e6, "msecond": 1e3, "ms": 1e3,
    "usecond": 1, "us": 1, "nsecond": 1e-3, "ns": 1e-3,
}


def number(value):
    try:
        result = float(value.replace(",", ""))
        return result if math.isfinite(result) else None
    except (ValueError, AttributeError):
        return None


def read_report(path, variant):
    with path.open(newline="") as source:
        reader = csv.reader(source)
        # Ignore any diagnostic lines preceding the CSV header.
        header = next((r for r in reader if r and r[0] == "ID" and "Kernel Name" in r), None)
        if header is None:
            raise ValueError(f"{path}: missing NCU raw CSV header")
        units = dict(zip(header, next(reader)))
        results = []
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"{path}: incomplete CSV row")
            data = dict(zip(header, row))
            result = {"variant": variant, "id": data["ID"], "kernel": data["Kernel Name"]}
            result.update({name: number(data.get(metric)) for name, metric in METRICS.items()})
            if result["duration_us"] is not None:
                unit = units[METRICS["duration_us"]]
                if unit not in TIME_TO_US:
                    raise ValueError(f"{path}: unsupported duration unit {unit!r}")
                result["duration_us"] *= TIME_TO_US[unit]
            results.append(result)
    if not results:
        raise ValueError(f"{path}: no captured launches; check NCU_LAUNCH_SKIP and GEMM_ARGS")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("variants", nargs="+")
    args = parser.parse_args()
    results = []
    for variant in args.variants:
        results.extend(read_report(args.directory / f"{variant}.raw.csv", variant))
    output = args.directory / "comparison.csv"
    with output.open("w", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=["variant", "id", "kernel", *METRICS])
        writer.writeheader()
        writer.writerows(results)
    print(f"{'Variant':<20} {'ID':>4} {'Duration us':>13} {'Occupancy %':>13} {'Tensor active %':>16}")
    for result in results:
        cells = ["N/A" if result[k] is None else f"{result[k]:.3f}"
                 for k in ("duration_us", "achieved_occupancy_pct", "tensor_active_pct")]
        print(f"{result['variant']:<20} {result['id']:>4} {cells[0]:>13} {cells[1]:>13} {cells[2]:>16}")
    print(f"Summary: {output}")
    print("Occupancy and Tensor active use active-cycle denominators; neither is GEMM peak FLOP/s utilization.")


if __name__ == "__main__":
    main()
