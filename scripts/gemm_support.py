"""Shared helpers for the optional Triton checks and GEMM comparison."""

import ctypes
import hashlib
import importlib.util
from pathlib import Path
import sys

try:
    import torch
    import triton
except ImportError as error:
    raise SystemExit(
        "GEMM Python targets require PyTorch with CUDA and Triton. "
        "Set make PYTHON=/path/to/an/environment/bin/python. "
        f"Import failed: {error}"
    ) from error


ROOT = Path(__file__).resolve().parents[1]


def load_triton(source=None):
    source = Path(source or ROOT / "src/gemm.triton.py").resolve()
    name = "texo_gemm_triton_" + hashlib.sha256(str(source).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def triton_configs(module):
    return module._configs() if hasattr(module, "get_tuned_configs") else module.gemm_kernel.configs


def selected_triton_config(module, m, n, k):
    if hasattr(module, "get_tuned_configs"):
        return module.get_tuned_configs()[(torch.cuda.current_device(), m, n, k, True, True)]
    config = module.gemm_kernel.best_config
    return dict(config.kwargs, num_warps=config.num_warps, num_stages=config.num_stages)


def fixed_triton_solve(module, config):
    """Exercise each version's raw JIT candidate, outside its autotuner."""
    def solve(a, b, c, m, n, k, alpha, beta):
        if m == 0 or n == 0:
            return
        tiles_m, tiles_n = triton.cdiv(m, config.kwargs["BM"]), triton.cdiv(n, config.kwargs["BN"])
        if hasattr(module, "get_tuned_configs"):
            use_i64 = max((m + 256) * (k + 64), (k + 64) * (n + 256),
                          (m + 256) * (n + 256)) >= 2**31
            module._gemm[(tiles_m * tiles_n,)](
                a, b, c, c, m, n, k, float(alpha), float(beta),
                BETA_ZERO=beta == 0, ALPHA_ONE=alpha == 1, USE_I64=use_i64,
                **config.all_kwargs())
        else:
            module.gemm_kernel.fn[(tiles_m, tiles_n)](
                a, b, c, m, n, k, float(alpha), float(beta), **config.all_kwargs())
    return solve


def check_device():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; select it with CUDA_VISIBLE_DEVICES")
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    print(f"GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}; "
          f"Triton={triton.__version__}; torch CUDA={torch.version.cuda}", flush=True)


def guarded_tensor(values, offset=128):
    """Keep an aligned prefix, or deliberately offset a pointer by one half."""
    storage = torch.full((offset + values.numel() + 1,), 123.0,
                         dtype=torch.float16, device="cuda")
    view = storage[offset:offset + values.numel()].view(values.shape)
    view.copy_(values)
    return storage, view


def check_guards(storage, count, offset=128):
    if not bool((storage[:offset] == 123).all()) or not bool((storage[offset + count:] == 123).all()):
        raise AssertionError("output guard overwritten")


def check_close(actual, expected):
    """Bound temporary storage for large matrices; reject NaNs explicitly."""
    actual, expected = actual.reshape(-1), expected.reshape(-1)
    maximum = 0.0
    for start in range(0, actual.numel(), 1 << 20):
        got = actual[start:start + (1 << 20)].float()
        ref = expected[start:start + (1 << 20)].float()
        error = (got - ref).abs()
        maximum = max(maximum, error.max().item())
        valid = torch.isfinite(got) & (error <= 0.01 + 0.01 * ref.abs())
        if not bool(valid.all()):
            bad = (~valid).nonzero()[0].item()
            raise AssertionError(f"element {start + bad}: got={got[bad].item()} "
                                 f"expected={ref[bad].item()}, max_error={maximum}")
    return maximum


class NativeGemm:
    """ctypes adapter for the repository's device-0/default-stream solve ABI."""

    def __init__(self, path):
        self.library = ctypes.CDLL(str(Path(path).resolve()))
        self.solve = self.library.solve
        self.solve.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_float] * 2
        self.solve.restype = None
        self.library.cudaGetLastError.argtypes = []
        self.library.cudaGetLastError.restype = ctypes.c_int

    def prepare(self, a, b, c, m, n, k):
        args = (a.data_ptr(), b.data_ptr(), c.data_ptr(), m, n, k, 1.0, 0.0)
        return lambda: self.solve(*args)

    def check_error(self):
        status = self.library.cudaGetLastError()
        if status:
            raise RuntimeError(f"CUDA launch error {status}")

    def versions(self):
        """Report the libraries actually resolved in this Python process."""
        versions = {}
        for symbol in ("cudaRuntimeGetVersion", "cudaDriverGetVersion"):
            fn = getattr(self.library, symbol)
            fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
            fn.restype = ctypes.c_int
            value = ctypes.c_int()
            if fn(ctypes.byref(value)) == 0:
                versions[symbol] = value.value
        get_property = self.library.cublasGetProperty
        get_property.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        get_property.restype = ctypes.c_int
        for index, name in enumerate(("cublas_major", "cublas_minor", "cublas_patch")):
            value = ctypes.c_int()
            if get_property(index, ctypes.byref(value)) == 0:
                versions[name] = value.value
        return versions
