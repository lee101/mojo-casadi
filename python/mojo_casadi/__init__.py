"""mojo_casadi — the dense numeric core of CasADi, compiled.

CasADi is a symbolic toolbox. Most of what it does — building expression
graphs, differentiating them, generating C code, running IPOPT and the ODE
integrators — lives in its C++ core behind `Function` objects, and none of it is
worth reimplementing in Mojo. What *is* worth compiling is the dense numeric
work that a CasADi caller ends up doing in Python on plain arrays: solving
systems, inverting, determinants and adjugates, the symmetric `L D L^T`
factorisation, the pseudoinverse, matrix products and Kronecker products, the
norms, and polynomial coefficient arithmetic.

Everything here is plain row-major `float64` NumPy in, NumPy out, and every
operation is checked against the real `casadi` in the test suite.

    >>> import numpy as np, mojo_casadi as mc
    >>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
    >>> mc.det(A)
    11.0
    >>> np.allclose(mc.solve(A, np.array([1.0, 2.0])), np.linalg.solve(A, [1.0, 2.0]))
    True
    >>> mc.poly_mul([1.0, 1.0], [1.0, -1.0])
    array([ 1.,  0., -1.])
"""

from ._lib import (  # noqa: F401
    NORM_1,
    NORM_FRO,
    NORM_INF,
    TRACE,
    adjugate,
    det,
    inv,
    kron,
    ldlt,
    ldlt_solve,
    lu_factor,
    lu_solve,
    mtimes,
    norm,
    norm_1,
    norm_2,
    norm_fro,
    norm_inf,
    pinv,
    poly_add,
    poly_deriv,
    poly_deriv_order,
    poly_eval,
    poly_mul,
    poly_scale,
    solve,
    trace,
)

__version__ = "0.1.0"

__all__ = [
    "lu_factor",
    "lu_solve",
    "solve",
    "inv",
    "det",
    "adjugate",
    "ldlt",
    "ldlt_solve",
    "pinv",
    "mtimes",
    "kron",
    "norm",
    "norm_1",
    "norm_2",
    "norm_fro",
    "norm_inf",
    "trace",
    "poly_eval",
    "poly_add",
    "poly_scale",
    "poly_mul",
    "poly_deriv",
    "poly_deriv_order",
    "NORM_1",
    "NORM_INF",
    "NORM_FRO",
    "TRACE",
]
