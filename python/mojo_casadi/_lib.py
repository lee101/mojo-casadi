"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults.
"""

import ctypes
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-casadi.so"

_PTR = ctypes.c_int64
_I = ctypes.c_int32


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    def sig(name, restype, argtypes):
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes

    sig("mc_lu_factor", ctypes.c_int64, [_PTR, _PTR, _PTR, _PTR])
    sig("mc_lu_solve", None, [_PTR, _PTR, _PTR, _PTR, _PTR])
    sig("mc_lu_inverse", None, [_PTR, _PTR, _PTR, _PTR, _PTR])
    sig("mc_det", ctypes.c_double, [_PTR, _PTR, _PTR])
    sig("mc_adj", ctypes.c_double, [_PTR, _PTR, _PTR, _PTR, _PTR])
    sig("mc_ldlt_factor", ctypes.c_int64, [_PTR, _PTR, _PTR, _PTR])
    sig("mc_ldlt_solve", None, [_PTR, _PTR, _PTR, _PTR, _PTR])
    sig("mc_pinv", ctypes.c_int64, [_PTR] * 7)
    sig("mc_gemm", None, [_PTR, _PTR, _PTR, _PTR, _PTR, _PTR,
                          ctypes.c_double, ctypes.c_double])
    sig("mc_gemm_rows", None, [_PTR, _PTR, _PTR, _PTR, _PTR, _PTR,
                               ctypes.c_double, ctypes.c_double, _PTR, _PTR])
    sig("mc_kron", None, [_PTR] * 7)
    sig("mc_norm", ctypes.c_double, [_PTR, _PTR, _PTR, _PTR])
    sig("mc_poly_eval", None, [_PTR] * 5)
    sig("mc_poly_add", ctypes.c_int64, [_PTR] * 5)
    sig("mc_poly_scale", None, [_PTR, _PTR, ctypes.c_double, _PTR])
    sig("mc_poly_mul", ctypes.c_int64, [_PTR] * 5)
    sig("mc_poly_deriv", ctypes.c_int64, [_PTR, _PTR, _PTR])
    sig("mc_poly_deriv_order", ctypes.c_int64, [_PTR, _PTR, _PTR, _PTR])
    return lib


lib = _load()


def _f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def _workers() -> int:
    return int(os.environ.get("MOJO_CASADI_THREADS", "0")) or min(8, os.cpu_count() or 1)


# ---------------------------------------------------------------------------
# LU
# ---------------------------------------------------------------------------


def lu_factor(a) -> tuple:
    """LU with partial pivoting. Returns (LU, piv); piv[i] is the row that was
    swapped into position i."""
    a = _f64(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("lu_factor needs a square matrix")
    n = a.shape[0]
    lu = np.empty((n, n), dtype=np.float64)
    piv = np.empty(n, dtype=np.int32)
    rc = lib.mc_lu_factor(a.ctypes.data, n, lu.ctypes.data, piv.ctypes.data)
    if rc != 0:
        raise np.linalg.LinAlgError("matrix is singular")
    return lu, piv


def lu_solve(lu, piv, b) -> np.ndarray:
    """Solve A x = b given A's LU factors. `b` may be a vector or a matrix."""
    lu = _f64(lu)
    b = _f64(b)
    n = lu.shape[0]
    if b.ndim == 1:
        b = b.reshape(n, 1)
        out = b.copy()
        lib.mc_lu_solve(lu.ctypes.data, piv.ctypes.data, n, 1, out.ctypes.data)
        return out[:, 0]
    # The kernel walks right-hand sides as contiguous columns, so hand it B
    # transposed and flatten that row-major, then transpose the answer back.
    work = np.ascontiguousarray(b.T).copy()
    lib.mc_lu_solve(lu.ctypes.data, piv.ctypes.data, n, b.shape[1],
                    work.ctypes.data)
    return np.ascontiguousarray(work.T)


def solve(a, b) -> np.ndarray:
    """Solve A x = b."""
    return lu_solve(*lu_factor(a), b)


def inv(a) -> np.ndarray:
    """Matrix inverse."""
    a = _f64(a)
    n = a.shape[0]
    lu, piv = lu_factor(a)
    out = np.empty((n, n), dtype=np.float64)
    work = np.empty(n * n, dtype=np.float64)
    lib.mc_lu_inverse(lu.ctypes.data, piv.ctypes.data, n, work.ctypes.data,
                      out.ctypes.data)
    return out


def det(a) -> float:
    """Determinant."""
    a = _f64(a)
    n = a.shape[0]
    lu, piv = lu_factor(a)
    return float(lib.mc_det(lu.ctypes.data, piv.ctypes.data, n))


def adjugate(a) -> np.ndarray:
    """Adjugate (classical adjoint), so that A @ adj(A) == det(A) * I."""
    a = _f64(a)
    n = a.shape[0]
    lu, piv = lu_factor(a)
    out = np.empty((n, n), dtype=np.float64)
    work = np.empty(n * n, dtype=np.float64)
    lib.mc_adj(lu.ctypes.data, piv.ctypes.data, n, work.ctypes.data,
               out.ctypes.data)
    return out


# ---------------------------------------------------------------------------
# LDL^T
# ---------------------------------------------------------------------------


def ldlt(a) -> tuple:
    """L D L^T factorisation of a symmetric matrix. Returns (L, D)."""
    a = _f64(a)
    n = a.shape[0]
    tol = 1e-12 * max(1.0, float(np.abs(a).max()))
    if not np.allclose(a, a.T, rtol=1e-12, atol=tol):
        raise ValueError("ldlt needs a symmetric matrix")
    lower = np.tril(a)
    l = np.empty((n, n), dtype=np.float64)
    d = np.empty(n, dtype=np.float64)
    if lib.mc_ldlt_factor(lower.ctypes.data, n, l.ctypes.data,
                          d.ctypes.data) != 0:
        raise np.linalg.LinAlgError("matrix is singular")
    return l, d


def ldlt_solve(a, b) -> np.ndarray:
    """Solve a symmetric system through L D L^T."""
    l, d = ldlt(a)
    b = _f64(b)
    n = l.shape[0]
    if b.ndim == 1:
        out = b.reshape(n, 1).copy()
        lib.mc_ldlt_solve(l.ctypes.data, d.ctypes.data, n, 1, out.ctypes.data)
        return out[:, 0]
    work = np.ascontiguousarray(b.T).copy()
    lib.mc_ldlt_solve(l.ctypes.data, d.ctypes.data, n, b.shape[1],
                      work.ctypes.data)
    return np.ascontiguousarray(work.T)


# ---------------------------------------------------------------------------
# Pseudoinverse
# ---------------------------------------------------------------------------


def pinv(a) -> np.ndarray:
    """Moore-Penrose pseudoinverse of a full-rank matrix.

    Rank-deficient and badly conditioned inputs are out of scope: this builds
    the Gram matrix and inverts it, which squares the condition number. Use
    `numpy.linalg.pinv` (an SVD) when that matters.
    """
    a = _f64(a)
    if a.ndim != 2:
        raise ValueError("pinv needs a 2-D array")
    m, n = a.shape
    k = min(m, n)
    out = np.empty((n, m), dtype=np.float64)
    work = np.empty(k * k, dtype=np.float64)
    lu = np.empty((k, k), dtype=np.float64)
    piv = np.empty(k, dtype=np.int32)
    if lib.mc_pinv(a.ctypes.data, m, n, out.ctypes.data, work.ctypes.data,
                   lu.ctypes.data, piv.ctypes.data) != 0:
        raise np.linalg.LinAlgError("matrix is rank deficient")
    return out


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------


def mtimes(a, b) -> np.ndarray:
    """Matrix product, split across threads when the product is big enough to
    pay for the fan-out."""
    a = _f64(a)
    b = _f64(b)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"cannot multiply {m}x{k} by {k2}x{n}")
    c = np.zeros((m, n), dtype=np.float64)
    workers = _workers()
    # Below roughly two flops per byte the copy in and out costs more than the
    # fan-out saves, so stay serial for small or skinny products.
    if m * n * k < 1 << 18 or workers <= 1:
        lib.mc_gemm(a.ctypes.data, b.ctypes.data, c.ctypes.data, m, k, n,
                    ctypes.c_double(1.0), ctypes.c_double(0.0))
        return c
    step = (m + workers - 1) // workers

    def part(r):
        lo, hi = r * step, min((r + 1) * step, m)
        if lo < hi:
            lib.mc_gemm_rows(a.ctypes.data, b.ctypes.data, c.ctypes.data, m, k,
                             n, ctypes.c_double(1.0), ctypes.c_double(0.0),
                             lo, hi)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(part, range(workers)))
    return c


def kron(a, b) -> np.ndarray:
    """Kronecker product."""
    a = _f64(a)
    b = _f64(b)
    m, n = a.shape
    p, q = b.shape
    out = np.empty((m * p, n * q), dtype=np.float64)
    lib.mc_kron(a.ctypes.data, m, n, b.ctypes.data, p, q, out.ctypes.data)
    return out


# ---------------------------------------------------------------------------
# Norms and trace
# ---------------------------------------------------------------------------

NORM_1 = 0
NORM_INF = 1
NORM_FRO = 2
TRACE = 3


def norm(a, kind) -> float:
    """kind is one of NORM_1, NORM_INF, NORM_FRO, TRACE."""
    a = _f64(a)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    m, n = a.shape
    return float(lib.mc_norm(a.ctypes.data, m, n, kind))


def norm_1(a) -> float:
    """Sum of absolute values, which is what `casadi.norm_1` returns for a
    dense matrix: CasADi reduces the entries as one flat vector."""
    return norm(a, NORM_1)


def norm_inf(a) -> float:
    """Largest absolute entry, matching `casadi.norm_inf`."""
    return norm(a, NORM_INF)


def norm_fro(a) -> float:
    """Root sum of squares, matching `casadi.norm_fro`."""
    return norm(a, NORM_FRO)


def norm_2(a) -> float:
    """Root sum of squares. CasADi's `norm_2` on a dense matrix is the same
    reduction, not the spectral norm."""
    return norm(a, NORM_FRO)


def trace(a) -> float:
    return norm(a, TRACE)


# ---------------------------------------------------------------------------
# Polynomials
# ---------------------------------------------------------------------------


def _coeffs(p) -> np.ndarray:
    return _f64(p)


def poly_eval(coeffs, x) -> np.ndarray:
    """Evaluate a polynomial by Horner's rule at a scalar or an array."""
    c = _coeffs(coeffs)
    xv = _f64(x).reshape(-1)
    out = np.empty(xv.size, dtype=np.float64)
    lib.mc_poly_eval(c.ctypes.data, c.size - 1, xv.size, xv.ctypes.data,
                     out.ctypes.data)
    return out if np.ndim(x) else out[0]


def poly_add(p, q) -> np.ndarray:
    p = _coeffs(p)
    q = _coeffs(q)
    out = np.empty(max(p.size, q.size), dtype=np.float64)
    deg = lib.mc_poly_add(p.ctypes.data, q.ctypes.data, p.size - 1, q.size - 1,
                          out.ctypes.data)
    return out[:deg + 1]


def poly_scale(p, alpha) -> np.ndarray:
    p = _coeffs(p)
    out = np.empty(p.size, dtype=np.float64)
    lib.mc_poly_scale(p.ctypes.data, p.size - 1, ctypes.c_double(alpha),
                      out.ctypes.data)
    return out


def poly_mul(p, q) -> np.ndarray:
    p = _coeffs(p)
    q = _coeffs(q)
    out = np.zeros(p.size + q.size - 1, dtype=np.float64)
    deg = lib.mc_poly_mul(p.ctypes.data, q.ctypes.data, p.size - 1, q.size - 1,
                          out.ctypes.data)
    return out[:deg + 1]


def poly_deriv(p) -> np.ndarray:
    p = _coeffs(p)
    out = np.zeros(p.size, dtype=np.float64)
    deg = lib.mc_poly_deriv(p.ctypes.data, p.size - 1, out.ctypes.data)
    return out[:deg + 1] if deg >= 0 else out[:1]


def poly_deriv_order(p, k: int) -> np.ndarray:
    """Take the k-th derivative by repeated single differentiation."""
    out = _coeffs(p)
    for _ in range(k):
        out = poly_deriv(out)
    return out
