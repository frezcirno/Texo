"""FP16 GEMM: C <- alpha * A @ B + beta * C; FP32 accumulation.

Candidate optimization for NVIDIA A800; GPU compilation/performance unverified.
Inputs: contiguous row-major GPU FP16 tensors, nonnegative M/N/K, with C
not overlapping A or B. Call on the tensors' current CUDA device/stream.

First use of each shape/path tunes into a separate output without changing C.
Later calls launch the chosen JIT kernel directly, with no temporary allocation.
Warm up every shape outside timing / an enclosing CUDA Graph capture.

Example:
    from gemm_a800_v2 import solve, get_tuned_configs
    solve(a, b, c, M, N, K, alpha, beta)
    print(get_tuned_configs())

Each solve call updates C once, including the initial tuning call. Restore the
original C outside timing when independently validating repeated calls.
"""

import torch
import triton
import triton.language as tl
import triton.testing


def _configs():
    # BM, BN, BK, warps, pipeline stages. Include smaller tiles for short K.
    choices = [
        (32, 64, 32, 2, 2),
        (64, 32, 32, 2, 2),
        (64, 64, 32, 4, 2),
        (64, 128, 32, 4, 2),
        (64, 64, 32, 4, 3),
        (64, 128, 32, 4, 3),
        (128, 64, 32, 4, 3),
        (128, 128, 32, 4, 4),
        (128, 128, 32, 8, 3),
        (64, 64, 64, 4, 3),
        (64, 128, 64, 4, 4),
        (128, 64, 64, 4, 4),
        (128, 128, 64, 8, 3),
        (128, 256, 32, 8, 3),
        (128, 256, 64, 8, 3),
        (256, 128, 64, 8, 3),
    ]
    return [
        triton.Config(
            {"BM": bm, "BN": bn, "BK": bk, "GROUP_M": group},
            num_warps=warps, num_stages=stages,
        )
        for bm, bn, bk, warps, stages in choices
        for group in (1, 8)
    ]


def _prune(configs, named_args, **kwargs):
    k = named_args["K"]
    if k <= 128:
        # A short reduction needs fewer pipeline stages. Keep some wide tiles.
        return [c for c in configs if c.num_stages <= 3]
    return [c for c in configs if c.kwargs["BM"] * c.kwargs["BN"] >= 4096]


def _bench(fn, quantiles):
    # Tune device execution rather than Python dispatch time. Tuning always
    # uses disjoint input/output C, so graph replay cannot compound beta*C.
    return triton.testing.do_bench_cudagraph(fn, rep=30, quantiles=quantiles)


@triton.jit
def _gemm(
    A, B, C_IN, C_OUT,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    alpha, beta,
    BETA_ZERO: tl.constexpr,
    ALPHA_ONE: tl.constexpr,
    USE_I64: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)

    # Group adjacent M tiles before moving along N, encouraging B reuse in L2.
    if GROUP_M == 1:
        pid_m = pid // num_n
        pid_n = pid % num_n
    else:
        group_width = GROUP_M * num_n
        group_id = pid // group_width
        first_m = group_id * GROUP_M
        group_size = tl.minimum(num_m - first_m, GROUP_M)
        local_pid = pid % group_width
        pid_m = first_m + local_pid % group_size
        pid_n = local_pid // group_size

    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)

    # 32-bit element-offset arithmetic for ordinary sizes; 64-bit fallback.
    if USE_I64:
        rows = rows.to(tl.int64)
        cols = cols.to(tl.int64)
        ks = ks.to(tl.int64)

    ap = A + rows[:, None] * K + ks[None, :]
    bp = B + ks[:, None] * N + cols[None, :]
    acc = tl.zeros((BM, BN), tl.float32)

    for kb in range(tl.cdiv(K, BK)):
        if M % BM == 0 and K % BK == 0:
            av = tl.load(ap)
        else:
            av = tl.load(
                ap,
                (rows[:, None] < M) & (kb * BK + ks[None, :] < K),
                other=0.0,
            )
        if N % BN == 0 and K % BK == 0:
            bv = tl.load(bp)
        else:
            bv = tl.load(
                bp,
                (cols[None, :] < N) & (kb * BK + ks[:, None] < K),
                other=0.0,
            )
        acc = tl.dot(av, bv, acc)
        ap += BK
        bp += BK * N

    offsets = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)

    if ALPHA_ONE:
        out = acc
    else:
        out = alpha * acc

    if not BETA_ZERO:
        if M % BM == 0 and N % BN == 0:
            old = tl.load(C_IN + offsets).to(tl.float32)
        else:
            old = tl.load(C_IN + offsets, mask, other=0.0).to(tl.float32)
        out = out + beta * old

    if M % BM == 0 and N % BN == 0:
        tl.store(C_OUT + offsets, out.to(tl.float16))
    else:
        tl.store(C_OUT + offsets, out.to(tl.float16), mask)


# Separate autotuner instances avoid reusing measurements across CUDA devices.
_tuners = {}
_selected = {}


def _select(a, b, c, M, N, K, alpha, beta, beta_zero, alpha_one, key):
    use_i64 = max(
        (M + 256) * (K + 64),
        (K + 64) * (N + 256),
        (M + 256) * (N + 256),
    ) >= 2**31

    device = a.device.index
    if device not in _tuners:
        _tuners[device] = triton.autotune(
            configs=_configs(),
            key=["M", "N", "K", "BETA_ZERO", "ALPHA_ONE", "USE_I64"],
            prune_configs_by={"early_config_prune": _prune},
            do_bench=_bench,
        )(_gemm)
    tuner = _tuners[device]

    # No clone/restore in the timed tuning function. Every candidate reads
    # unchanged C and writes a scratch output; normal execution is in-place.
    scratch = torch.empty_like(c)
    grid = lambda meta: (
        triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),
    )
    tuner[grid](
        a, b, c, scratch, M, N, K, alpha, beta,
        BETA_ZERO=beta_zero, ALPHA_ONE=alpha_one, USE_I64=use_i64,
    )
    cfg = tuner.best_config
    launch_grid = (
        triton.cdiv(M, cfg.kwargs["BM"]) * triton.cdiv(N, cfg.kwargs["BN"]),
    )
    meta = dict(
        cfg.kwargs,
        BETA_ZERO=beta_zero, ALPHA_ONE=alpha_one, USE_I64=use_i64,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages,
    )
    # Cache configuration/launcher only, never tensor pointers or results.
    selected = (_gemm[launch_grid], meta)
    _selected[key] = selected
    return selected


def solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
    alpha: float,
    beta: float,
):
    if M == 0 or N == 0:
        return
    beta_zero = beta == 0.0
    alpha_one = alpha == 1.0
    key = (a.device.index, M, N, K, beta_zero, alpha_one)
    selected = _selected.get(key)
    if selected is None:
        selected = _select(
            a, b, c, M, N, K, alpha, beta, beta_zero, alpha_one, key,
        )
    launch, meta = selected
    launch(a, b, c, c, M, N, K, alpha, beta, **meta)


def get_tuned_configs():
    """Return selected metadata keyed by (device, M, N, K, beta_zero, alpha_one)."""
    return {key: dict(meta) for key, (_, meta) in _selected.items()}
