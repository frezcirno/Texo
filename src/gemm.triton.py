import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BM": 32, "BN": 64, "BK": 32},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BM": 64, "BN": 64, "BK": 32},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BM": 64, "BN": 128, "BK": 32},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BM": 128, "BN": 64, "BK": 32},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BM": 128, "BN": 128, "BK": 32},
            num_warps=8, num_stages=3,
        ),
        triton.Config(
            {"BM": 64, "BN": 64, "BK": 64},
            num_warps=4, num_stages=4,
        ),
        triton.Config(
            {"BM": 64, "BN": 128, "BK": 64},
            num_warps=4, num_stages=4,
        ),
    ],
    key=["M", "N", "K", "beta"],
    restore_value=["C"],
)
@triton.jit
def gemm_kernel(
    A, B, C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    alpha, beta,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)

    # 行主序：A[M, K]，B[K, N]，C[M, N]。
    a_ptrs = (
        A
        + rows[:, None].to(tl.int64) * K
        + ks[None, :]
    )
    b_ptrs = (
        B
        + ks[:, None].to(tl.int64) * N
        + cols[None, :]
    )

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k_block in range(tl.cdiv(K, BK)):
        k_valid = k_block * BK + ks < K

        a_tile = tl.load(
            a_ptrs,
            mask=(rows[:, None] < M) & k_valid[None, :],
            other=0.0,
        )
        b_tile = tl.load(
            b_ptrs,
            mask=k_valid[:, None] & (cols[None, :] < N),
            other=0.0,
        )

        # FP16 输入、FP32 累加。
        # 在支持的 NVIDIA GPU 上使用 Tensor Core。
        acc = tl.dot(a_tile, b_tile, acc)

        a_ptrs += BK
        b_ptrs += BK * N

    c_ptrs = (
        C
        + rows[:, None].to(tl.int64) * N
        + cols[None, :]
    )
    c_mask = (rows[:, None] < M) & (cols[None, :] < N)

    result = alpha * acc

    if beta != 0.0:
        c_initial = tl.load(
            c_ptrs,
            mask=c_mask,
            other=0.0,
        ).to(tl.float32)

        result = result + beta * c_initial

    tl.store(
        c_ptrs,
        result.to(tl.float16),
        mask=c_mask,
    )


# a, b, c are tensors on the GPU
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

    grid = lambda meta: (
        triton.cdiv(M, meta["BM"]),
        triton.cdiv(N, meta["BN"]),
    )

    gemm_kernel[grid](
        a, b, c,
        M, N, K,
        float(alpha), float(beta),
    )