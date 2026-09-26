"""Dense linear algebra and polynomial kernels for the CasADi numeric core.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.

Matrices are row-major C-contiguous float64. Nothing here is sparse: CasADi's
sparse linear algebra lives in its C++ core behind `Function` objects, and these
are the dense kernels a Python-level caller reaches for.
"""

from std.math import abs, sqrt

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


# ---------------------------------------------------------------------------
# LU with partial pivoting
# ---------------------------------------------------------------------------
# The factorisation keeps the permutation as the sequence of row interchanges,
# so applying the same swaps in the same order twice is the identity. That is
# what lets one routine both apply and undo P without materialising it.


@export("mc_lu_factor")
def mc_lu_factor(a_addr: Int, n: Int, lu_addr: Int, piv_addr: Int) abi("C") -> Int:
    """LU-decompose a row-major n x n matrix into `lu`.

    `piv` receives n row indices. Returns 0 on success and -1 if a pivot column
    is exactly zero, which is the singular case.
    """
    var a = fp(a_addr)
    var lu = fp(lu_addr)
    var piv = ip(piv_addr)
    for i in range(n):
        for j in range(n):
            lu[unsafe_offset=i * n + j] = a[unsafe_offset=i * n + j]
    for k in range(n):
        # Partial pivoting: the largest magnitude entry in the column.
        var best = k
        var bestv = abs(lu[unsafe_offset=k * n + k])
        for i in range(k + 1, n):
            var v = abs(lu[unsafe_offset=i * n + k])
            if v > bestv:
                bestv = v
                best = i
        if bestv == 0.0:
            return -1
        # LAPACK's ipiv: entry k is the row that landed in slot k, or k itself
        # when no interchange happened. Only slot k is ever written, because
        # slot `best` still belongs to a later step of the factorisation.
        piv[unsafe_offset=k] = Int32(best)
        if best != k:
            for j in range(n):
                var t = lu[unsafe_offset=k * n + j]
                lu[unsafe_offset=k * n + j] = lu[unsafe_offset=best * n + j]
                lu[unsafe_offset=best * n + j] = t
        var d = lu[unsafe_offset=k * n + k]
        for i in range(k + 1, n):
            var f = lu[unsafe_offset=i * n + k] / d
            lu[unsafe_offset=i * n + k] = f
            if f != 0.0:
                for j in range(k + 1, n):
                    lu[unsafe_offset=i * n + j] = lu[unsafe_offset=i * n + j] \
                        - f * lu[unsafe_offset=k * n + j]
    return 0


@export("mc_lu_solve")
def mc_lu_solve(lu_addr: Int, piv_addr: Int, n: Int, nrhs: Int,
                b_addr: Int) abi("C"):
    """Solve LU x = P b. The right-hand sides are stored column-major, so a
    caller's row-major m x k matrix can be solved in place by passing k."""
    var lu = fp(lu_addr)
    var piv = ip(piv_addr)
    var b = fp(b_addr)
    for r in range(nrhs):
        var off = r * n
        # Replay the interchanges in the order the factorisation made them,
        # which is how LAPACK's dlaswp undoes ipiv.
        for i in range(n):
            var p = Int(piv[unsafe_offset=i])
            var v = b[unsafe_offset=off + i]
            b[unsafe_offset=off + i] = b[unsafe_offset=off + p]
            b[unsafe_offset=off + p] = v
        # Forward substitution through L, which is unit lower triangular.
        for i in range(1, n):
            var s = b[unsafe_offset=off + i]
            for j in range(i):
                s = s - lu[unsafe_offset=i * n + j] * b[unsafe_offset=off + j]
            b[unsafe_offset=off + i] = s
        # Back substitution through U.
        for i in range(n - 1, -1, -1):
            var s = b[unsafe_offset=off + i]
            for j in range(i + 1, n):
                s = s - lu[unsafe_offset=i * n + j] * b[unsafe_offset=off + j]
            b[unsafe_offset=off + i] = s / lu[unsafe_offset=i * n + i]


@export("mc_lu_inverse")
def mc_lu_inverse(lu_addr: Int, piv_addr: Int, n: Int, work_addr: Int,
                  out_addr: Int) abi("C"):
    """Invert from an existing factorisation by solving n unit systems.

    `work` is n*n scratch. The result is transposed out of the column-major
    solution buffer into the row-major `out`.
    """
    var lu = fp(lu_addr)
    var out = fp(out_addr)
    var w = fp(work_addr)
    for i in range(n * n):
        w[unsafe_offset=i] = 0.0
    for i in range(n):
        w[unsafe_offset=i * n + i] = 1.0
    mc_lu_solve(lu_addr, piv_addr, n, n, work_addr)
    for i in range(n):
        for j in range(n):
            out[unsafe_offset=i * n + j] = w[unsafe_offset=j * n + i]


@export("mc_det")
def mc_det(lu_addr: Int, piv_addr: Int, n: Int) abi("C") -> Float64:
    """Determinant from an LU factorisation: the pivot product, signed by the
    parity of the row interchanges."""
    var lu = fp(lu_addr)
    var piv = ip(piv_addr)
    var d = 1.0
    var swaps = 0
    for i in range(n):
        if Int(piv[unsafe_offset=i]) != i:
            swaps += 1
        d = d * lu[unsafe_offset=i * n + i]
    if swaps % 2 == 1:
        d = -d
    return d


@export("mc_adj")
def mc_adj(lu_addr: Int, piv_addr: Int, n: Int, work_addr: Int,
           out_addr: Int) abi("C") -> Float64:
    """Adjugate, which is det(A) * A^-1, so one factorisation serves both."""
    var out = fp(out_addr)
    mc_lu_inverse(lu_addr, piv_addr, n, work_addr, out_addr)
    var d = mc_det(lu_addr, piv_addr, n)
    for i in range(n * n):
        out[unsafe_offset=i] = out[unsafe_offset=i] * d
    return d


# ---------------------------------------------------------------------------
# LDL^T for symmetric matrices
# ---------------------------------------------------------------------------


@export("mc_ldlt_factor")
def mc_ldlt_factor(a_addr: Int, n: Int, l_addr: Int, d_addr: Int) abi("C") -> Int:
    """L D L^T factorisation. Returns 0, or -1 if a pivot is exactly zero."""
    var a = fp(a_addr)
    var l = fp(l_addr)
    var d = fp(d_addr)
    for i in range(n):
        for j in range(n):
            if i == j:
                l[unsafe_offset=i * n + j] = 1.0
            else:
                l[unsafe_offset=i * n + j] = 0.0
    for j in range(n):
        var s = a[unsafe_offset=j * n + j]
        for k in range(j):
            s = s - l[unsafe_offset=j * n + k] * l[unsafe_offset=j * n + k] * d[unsafe_offset=k]
        d[unsafe_offset=j] = s
        if s == 0.0:
            return -1
        for i in range(j + 1, n):
            var t = a[unsafe_offset=i * n + j]
            for k in range(j):
                t = t - l[unsafe_offset=i * n + k] * l[unsafe_offset=j * n + k] * d[unsafe_offset=k]
            l[unsafe_offset=i * n + j] = t / s
    return 0


@export("mc_ldlt_solve")
def mc_ldlt_solve(l_addr: Int, d_addr: Int, n: Int, nrhs: Int,
                  b_addr: Int) abi("C"):
    """Solve L D L^T x = b for column-major right-hand sides."""
    var l = fp(l_addr)
    var d = fp(d_addr)
    var b = fp(b_addr)
    for r in range(nrhs):
        var off = r * n
        for i in range(n):
            var s = b[unsafe_offset=off + i]
            for j in range(i):
                s = s - l[unsafe_offset=i * n + j] * b[unsafe_offset=off + j]
            b[unsafe_offset=off + i] = s
        for i in range(n):
            b[unsafe_offset=off + i] = b[unsafe_offset=off + i] / d[unsafe_offset=i]
        for i in range(n - 1, -1, -1):
            var s = b[unsafe_offset=off + i]
            for j in range(i + 1, n):
                s = s - l[unsafe_offset=j * n + i] * b[unsafe_offset=off + j]
            b[unsafe_offset=off + i] = s


# ---------------------------------------------------------------------------
# Pseudoinverse
# ---------------------------------------------------------------------------


@export("mc_pinv")
def mc_pinv(a_addr: Int, m: Int, n: Int, out_addr: Int, work_addr: Int,
            lu_addr: Int, piv_addr: Int) abi("C") -> Int:
    """Moore-Penrose pseudoinverse of a full-rank m x n matrix.

    A wide matrix gets the minimum-norm solution A^T (A A^T)^-1, a tall one the
    least-squares solution (A^T A)^-1 A^T. Both come from a single Gram matrix
    and a single factorisation.
    """
    var a = fp(a_addr)
    var out = fp(out_addr)
    var w = fp(work_addr)
    var k = 0
    if m <= n:
        k = m
    else:
        k = n
    # A wide matrix needs A A^T (summing over columns); a tall one needs
    # A^T A (summing over rows). Getting these the wrong way round is the trap:
    # they are different matrices and the result looks plausible either way.
    if m <= n:
        for i in range(k):
            for j in range(k):
                var s = 0.0
                for t in range(n):
                    s += a[unsafe_offset=i * n + t] * a[unsafe_offset=j * n + t]
                w[unsafe_offset=i * k + j] = s
    else:
        for i in range(k):
            for j in range(k):
                var s = 0.0
                for t in range(m):
                    s += a[unsafe_offset=t * n + i] * a[unsafe_offset=t * n + j]
                w[unsafe_offset=i * k + j] = s
    if mc_lu_factor(work_addr, k, lu_addr, piv_addr) != 0:
        return -1
    for i in range(k * k):
        w[unsafe_offset=i] = 0.0
    for i in range(k):
        w[unsafe_offset=i * k + i] = 1.0
    mc_lu_solve(lu_addr, piv_addr, k, k, work_addr)
    # w now holds (gram)^-1, laid out column-major.
    if m <= n:
        # A^T (A A^T)^-1, an n x m result.
        for i in range(n):
            for j in range(m):
                var s = 0.0
                for t in range(m):
                    s += a[unsafe_offset=t * n + i] * w[unsafe_offset=j * m + t]
                out[unsafe_offset=i * m + j] = s
    else:
        for i in range(n):
            for j in range(m):
                var s = 0.0
                for t in range(n):
                    s += w[unsafe_offset=t * n + i] * a[unsafe_offset=j * n + t]
                out[unsafe_offset=i * m + j] = s
    return 0


# ---------------------------------------------------------------------------
# Products and reductions
# ---------------------------------------------------------------------------


@export("mc_gemm_rows")
def mc_gemm_rows(a_addr: Int, b_addr: Int, c_addr: Int, m: Int, k: Int,
                 n: Int, alpha: Float64, beta: Float64, row0: Int,
                 row1: Int) abi("C"):
    """C[rows] = alpha * A[rows] B + beta * C[rows], row-major.

    Row ranges let a caller split a compute-bound product across threads.
    """
    var a = fp(a_addr)
    var b = fp(b_addr)
    var c = fp(c_addr)
    for i in range(row0, row1):
        for j in range(n):
            var s = 0.0
            for t in range(k):
                s += a[unsafe_offset=i * k + t] * b[unsafe_offset=t * n + j]
            c[unsafe_offset=i * n + j] = alpha * s + beta * c[unsafe_offset=i * n + j]


@export("mc_gemm")
def mc_gemm(a_addr: Int, b_addr: Int, c_addr: Int, m: Int, k: Int, n: Int,
            alpha: Float64, beta: Float64) abi("C"):
    """C = alpha * A B + beta * C for row-major m x k, k x n and m x n."""
    mc_gemm_rows(a_addr, b_addr, c_addr, m, k, n, alpha, beta, 0, m)


@export("mc_kron")
def mc_kron(a_addr: Int, m: Int, n: Int, b_addr: Int, p: Int, q: Int,
            out_addr: Int) abi("C"):
    """Kronecker product of an m x n and a p x q matrix, giving (m*p) x (n*q)."""
    var a = fp(a_addr)
    var b = fp(b_addr)
    var o = fp(out_addr)
    for i in range(m):
        for j in range(n):
            var av = a[unsafe_offset=i * n + j]
            var rbase = (i * p) * (n * q) + j * q
            for u in range(p):
                var orow = rbase + u * n * q
                for v in range(q):
                    o[unsafe_offset=orow + v] = av * b[unsafe_offset=u * q + v]


@export("mc_norm")
def mc_norm(a_addr: Int, m: Int, n: Int, kind: Int) abi("C") -> Float64:
    """The four reductions CasADi's norm_1, norm_inf, norm_fro and trace
    perform on a dense matrix. All of them treat the entries as one flat
    vector, not as rows and columns.

    kind 0 is the sum of absolute values, 1 the largest absolute entry, 2 the
    root sum of squares (CasADi's norm_2 and norm_fro agree), 3 the trace.
    """
    var a = fp(a_addr)
    if kind == 0:
        var s = 0.0
        for i in range(m * n):
            s += abs(a[unsafe_offset=i])
        return s
    if kind == 1:
        var best = 0.0
        for i in range(m * n):
            var v = abs(a[unsafe_offset=i])
            if v > best:
                best = v
        return best
    if kind == 2:
        var s = 0.0
        for i in range(m * n):
            var v = a[unsafe_offset=i]
            s += v * v
        return sqrt(s)
    var t = 0.0
    for i in range(min(m, n)):
        t += a[unsafe_offset=i * n + i]
    return t


# ---------------------------------------------------------------------------
# Polynomials
# ---------------------------------------------------------------------------
# Coefficients are in descending order, as in `casadi.polyval` and NumPy, so
# [1, 0, -1] is x^2 - 1.


@export("mc_poly_eval")
def mc_poly_eval(coeff_addr: Int, degree: Int, n: Int, x_addr: Int,
                 out_addr: Int) abi("C"):
    """Evaluate a polynomial at n points by Horner's rule."""
    var c = fp(coeff_addr)
    var x = fp(x_addr)
    var o = fp(out_addr)
    for i in range(n):
        var xi = x[unsafe_offset=i]
        var acc = c[unsafe_offset=0]
        for k in range(1, degree + 1):
            acc = acc * xi + c[unsafe_offset=k]
        o[unsafe_offset=i] = acc


@export("mc_poly_add")
def mc_poly_add(a_addr: Int, b_addr: Int, na: Int, nb: Int,
                out_addr: Int) abi("C") -> Int:
    """Add two coefficient vectors of different degrees. Returns the degree."""
    var a = fp(a_addr)
    var b = fp(b_addr)
    var o = fp(out_addr)
    var deg = na
    if nb > na:
        deg = nb
    # Coefficients run from the highest power down, so the two polynomials are
    # aligned at the constant term, not at the leading term.
    for i in range(deg + 1):
        var s = 0.0
        var from_a = na - (deg - i)
        if from_a >= 0:
            s += a[unsafe_offset=from_a]
        var from_b = nb - (deg - i)
        if from_b >= 0:
            s += b[unsafe_offset=from_b]
        o[unsafe_offset=i] = s
    return deg


@export("mc_poly_scale")
def mc_poly_scale(a_addr: Int, na: Int, alpha: Float64, out_addr: Int) abi("C"):
    var a = fp(a_addr)
    var o = fp(out_addr)
    for i in range(na + 1):
        o[unsafe_offset=i] = alpha * a[unsafe_offset=i]


@export("mc_poly_mul")
def mc_poly_mul(a_addr: Int, b_addr: Int, na: Int, nb: Int,
                out_addr: Int) abi("C") -> Int:
    """Multiply two polynomials by convolution. Returns the degree."""
    var a = fp(a_addr)
    var b = fp(b_addr)
    var o = fp(out_addr)
    for i in range(na + 1):
        for j in range(nb + 1):
            o[unsafe_offset=i + j] = o[unsafe_offset=i + j] \
                + a[unsafe_offset=i] * b[unsafe_offset=j]
    return na + nb


@export("mc_poly_deriv")
def mc_poly_deriv(a_addr: Int, na: Int, out_addr: Int) abi("C") -> Int:
    """Differentiate. A constant differentiates to zero, reported as degree -1
    so the caller can tell that apart from a real degree-0 result."""
    var a = fp(a_addr)
    var o = fp(out_addr)
    if na <= 0:
        o[unsafe_offset=0] = 0.0
        return -1
    for i in range(na):
        o[unsafe_offset=i] = Float64(na - i) * a[unsafe_offset=i]
    return na - 1


@export("mc_poly_deriv_order")
def mc_poly_deriv_order(a_addr: Int, na: Int, k: Int, out_addr: Int) abi("C") -> Int:
    """Take the k-th derivative, reusing the scratch on each pass."""
    var deg = na
    for _ in range(k):
        deg = mc_poly_deriv(a_addr, deg, out_addr)
    return deg
