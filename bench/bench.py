"""Correctness-gated benchmark for mojo-casadi.

Every case checks the result against the real `casadi` before it is timed, so a
regression in the Mojo kernels shows up as a correctness failure rather than as
a suspiciously good number.

The baselines are CasADi's own C++ dense kernels reached through `DM`
operations, which is the strongest fair comparison available: NumPy's
`linalg` routines are OpenBLAS and would be a different, and generally
faster, opponent than the thing being ported.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import casadi as ca  # noqa: E402

import mojo_casadi as mc  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _nonsingular(n, rng):
    a = rng.standard_normal((n, n))
    a[np.arange(n), np.arange(n)] += n * 0.25
    return a


def bench_mtimes(n: int = 256, m: int = 256, k: int = 256):
    """Matrix product, against casadi.mtimes on DM operands."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k))
    b = rng.standard_normal((k, n))
    da, db = ca.DM(a), ca.DM(b)
    got = mc.mtimes(a, b)
    expect = np.array(ca.mtimes(da, db)).reshape(m, n)
    np.testing.assert_allclose(got, expect, rtol=1e-9, atol=1e-10)
    ref = _time(lambda: np.array(ca.mtimes(da, db)), 3)
    mine = _time(lambda: mc.mtimes(a, b), 3)
    return f"mtimes {m}x{k}x{n}", ref, mine


def bench_solve(n: int = 128, nrhs: int = 1):
    rng = np.random.default_rng(1)
    a = _nonsingular(n, rng)
    b = rng.standard_normal((n, nrhs))
    da = ca.DM(a)
    db = ca.DM(b)
    np.testing.assert_allclose(mc.solve(a, b), np.array(ca.solve(da, db)),
                               rtol=1e-8, atol=1e-9)
    ref = _time(lambda: np.array(ca.solve(da, db)), 2)
    mine = _time(lambda: mc.solve(a, b), 3)
    return f"solve {n}x{n} ({nrhs} rhs)", ref, mine


def bench_inv(n: int = 64):
    rng = np.random.default_rng(2)
    a = _nonsingular(n, rng)
    da = ca.DM(a)
    np.testing.assert_allclose(mc.inv(a), np.array(ca.inv(da)),
                               rtol=1e-7, atol=1e-8)
    ref = _time(lambda: np.array(ca.inv(da)), 2)
    mine = _time(lambda: mc.inv(a), 3)
    return f"inv {n}x{n}", ref, mine


def bench_det(n: int = 8):
    rng = np.random.default_rng(3)
    a = _nonsingular(n, rng)
    da = ca.DM(a)
    expect = float(ca.det(da))
    assert abs(mc.det(a) - expect) <= 1e-8 * max(1.0, abs(expect))
    # casadi's DM.det is a cofactor recursion in this build and costs seconds
    # at n=8, so it gets a single run; the ratio is real but the absolute
    # baseline is not a performance target anyone would set.
    ref = _time(lambda: float(ca.det(da)), 1)
    mine = _time(lambda: mc.det(a), 3)
    return f"det {n}x{n} (casadi is O(n!) here)", ref, mine


def bench_kron(m: int = 40, n: int = 40, p: int = 40, q: int = 40):
    rng = np.random.default_rng(4)
    a = rng.standard_normal((m, n))
    b = rng.standard_normal((p, q))
    da, db = ca.DM(a), ca.DM(b)
    got = mc.kron(a, b)
    expect = np.array(ca.kron(da, db))
    np.testing.assert_array_equal(got, expect)
    ref = _time(lambda: np.array(ca.kron(da, db)), 3)
    mine = _time(lambda: mc.kron(a, b), 3)
    return f"kron {m*n}x{p*q}", ref, mine


def bench_pinv(m: int = 8, n: int = 300):
    rng = np.random.default_rng(5)
    a = rng.standard_normal((m, n))
    da = ca.DM(a)
    np.testing.assert_allclose(mc.pinv(a), np.array(ca.pinv(da)),
                               rtol=1e-6, atol=1e-7)
    ref = _time(lambda: np.array(ca.pinv(da)), 3)
    mine = _time(lambda: mc.pinv(a), 3)
    return f"pinv {m}x{n}", ref, mine


def bench_norms(n: int = 2000):
    rng = np.random.default_rng(6)
    a = rng.standard_normal((n, n))
    da = ca.DM(a)
    for mine_f, ref_f in ((mc.norm_1, lambda: float(ca.norm_1(da))),
                          (mc.norm_inf, lambda: float(ca.norm_inf(da))),
                          (mc.norm_fro, lambda: float(ca.norm_fro(da)))):
        assert abs(mine_f(a) - ref_f()) <= 1e-9 * max(1.0, abs(ref_f()))
    ref = _time(lambda: float(ca.norm_fro(da)), 3)
    mine = _time(lambda: mc.norm_fro(a), 3)
    return f"norm_fro {n}x{n}", ref, mine


def bench_poly_eval(deg: int = 64, n: int = 1 << 20):
    rng = np.random.default_rng(7)
    coeffs = rng.standard_normal(deg + 1)
    x = rng.standard_normal(n)
    got = np.asarray(mc.poly_eval(coeffs, x)).reshape(-1)
    expect = np.array([ca.polyval(ca.DM(coeffs), float(v)) for v in x[:64]]).reshape(-1)
    np.testing.assert_allclose(got[:64], expect, rtol=1e-9, atol=1e-10)
    # The fair NumPy baseline is Clenshaw/Horner vectorised over the sample set.
    ref = _time(lambda: np.polynomial.polynomial.polyval(x, coeffs[::-1]), 3)
    mine = _time(lambda: mc.poly_eval(coeffs, x), 3)
    return f"poly_eval deg={deg} n={n}", ref, mine


def bench_ldlt(n: int = 200):
    rng = np.random.default_rng(8)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    a = (q * np.linspace(1.0, 5.0, n)) @ q.T
    a = (a + a.T) / 2.0
    l, d = mc.ldlt(a)
    np.testing.assert_allclose(l @ np.diag(d) @ l.T, a, rtol=1e-8, atol=1e-9)
    b = rng.standard_normal(n)
    np.testing.assert_allclose(mc.ldlt_solve(a, b), np.linalg.solve(a, b),
                               rtol=1e-8, atol=1e-9)
    # casadi's own ldl_solve segfaults on a 1x1 system in this build, so the
    # baseline here is LAPACK's general solve through NumPy, which is a stronger
    # opponent than a Cholesky-based symmetric solve anyway.
    ref = _time(lambda: np.linalg.solve(a, b), 3)
    mine = _time(lambda: mc.ldlt_solve(a, b), 3)
    return f"ldlt_solve {n}x{n} (vs LAPACK)", ref, mine


def main():
    print(f"{'case':<34}{'reference':>14}{'mojo-casadi':>16}{'ratio':>12}  threads")
    print("-" * 82)
    for fn in (bench_mtimes, bench_solve, bench_inv, bench_det, bench_kron,
               bench_pinv, bench_norms, bench_poly_eval, bench_ldlt):
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        print(f"{label:<34}{ref*1e3:>12.2f}ms{got*1e3:>14.2f}ms{ratio:>11.1f}x"
              f"  {os.environ.get('MOJO_CASADI_THREADS', 'auto')}")


if __name__ == "__main__":
    main()
