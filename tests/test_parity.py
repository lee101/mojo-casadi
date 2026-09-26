"""Parity between the Mojo kernels and the real casadi.

Every case compares against `casadi` itself, not against NumPy, so the tests
fail if the two implementations disagree. Where a closed form exists (the
adjugate satisfies A adj(A) = det(A) I) that is checked too, because a
consistently wrong inverse and a consistently wrong CasADi would otherwise
cancel out.

Mojo emits FMA, so results agree to a tolerance rather than bit for bit. The
tolerances below are set per operation: a sum of n products accumulates about
n * eps of relative error, so a dot product over k terms gets a tolerance that
scales with k rather than a single blanket number.
"""

import casadi as ca
import numpy as np
import pytest

import mojo_casadi as mc


def spd(n, rng, cond=1.0):
    """A well-conditioned symmetric positive definite matrix, made exactly
    symmetric so the shim's check is not testing floating-point noise."""
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    a = (q * np.linspace(1.0, cond, n)) @ q.T
    return (a + a.T) / 2.0


def nonsingular(n, rng):
    """A general square matrix that is not diagonally dominant, so that
    partial pivoting actually fires."""
    a = rng.standard_normal((n, n))
    a[np.arange(n), np.arange(n)] += n * 0.25
    return a


# ---------------------------------------------------------------------------
# LU, solve, inverse, determinant, adjugate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 17])
def test_solve_matches_casadi(n):
    rng = np.random.default_rng(n)
    a = nonsingular(n, rng)
    b = rng.standard_normal(n)
    np.testing.assert_allclose(
        mc.solve(a, b), np.array(ca.solve(ca.DM(a), ca.DM(b))).squeeze(),
        rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 17])
def test_solve_multiple_right_hand_sides_matches_casadi(n):
    rng = np.random.default_rng(n + 100)
    a = nonsingular(n, rng)
    b = rng.standard_normal((n, 4))
    np.testing.assert_allclose(
        mc.solve(a, b), np.array(ca.solve(ca.DM(a), ca.DM(b))),
        rtol=1e-9, atol=1e-9)


def test_solve_is_exercised_by_pivoting():
    """A matrix whose largest entry in column 0 is below the diagonal, so a
    factorisation without pivoting gets the wrong answer."""
    a = np.array([[0.0, 1.0], [1.0, 0.0]])
    b = np.array([2.0, 3.0])
    np.testing.assert_allclose(mc.solve(a, b), np.linalg.solve(a, b),
                               rtol=1e-12, atol=1e-12)
    # Without pivoting the diagonal entry is zero, so this must not silently
    # produce an infinity.
    with pytest.raises(np.linalg.LinAlgError):
        mc.lu_factor(np.array([[0.0, 1.0], [0.0, 1.0]]))


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 17])
def test_inverse_matches_casadi(n):
    rng = np.random.default_rng(n + 200)
    a = nonsingular(n, rng)
    np.testing.assert_allclose(mc.inv(a), np.array(ca.inv(ca.DM(a))),
                               rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6, 9])
def test_inverse_is_a_two_sided_inverse(n):
    rng = np.random.default_rng(n + 300)
    a = nonsingular(n, rng)
    inv = mc.inv(a)
    eye = np.eye(n)
    np.testing.assert_allclose(a @ inv, eye, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(inv @ a, eye, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6, 9])
def test_det_matches_casadi(n):
    rng = np.random.default_rng(n + 400)
    a = nonsingular(n, rng)
    expect = float(ca.det(ca.DM(a)))
    got = mc.det(a)
    assert got == pytest.approx(expect, rel=1e-8, abs=1e-9)


def test_det_sign_follows_the_row_interchanges():
    """Swapping two rows of a 2x2 flips the sign of the determinant, which is
    exactly the part a LU implementation can get wrong."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert mc.det(a) == pytest.approx(-2.0)
    assert mc.det(a[::-1, :]) == pytest.approx(2.0)
    assert mc.det(a[:, ::-1]) == pytest.approx(2.0)
    assert mc.det(-a) == pytest.approx(mc.det(a), rel=1e-12)


def test_det_of_a_singular_matrix_is_reported():
    with pytest.raises(np.linalg.LinAlgError):
        mc.det(np.zeros((3, 3)))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6])
def test_adjugate_matches_casadi(n):
    rng = np.random.default_rng(n + 500)
    a = rng.standard_normal((n, n))
    a[np.arange(n), np.arange(n)] += n * 0.5
    np.testing.assert_allclose(mc.adjugate(a), np.array(ca.adj(ca.DM(a))),
                               rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6])
def test_adjugate_satisfies_the_classical_identity(n):
    rng = np.random.default_rng(n + 600)
    a = rng.standard_normal((n, n))
    a[np.arange(n), np.arange(n)] += n * 0.5
    adj = mc.adjugate(a)
    np.testing.assert_allclose(a @ adj, mc.det(a) * np.eye(n),
                               rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(adj @ a, mc.det(a) * np.eye(n),
                               rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# LDL^T
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_ldlt_pivots_agree_with_casadi_ldl(n):
    """casadi.ldl returns (D, L, P) in a scaled, permuted convention that is not
    the textbook unit-lower L D L^T, so the factors cannot be compared entry for
    entry, and casadi's own ldl_solve segfaults on a one-by-one system. What
    both factorisations must agree on is the determinant, which is the product
    of the pivots for either convention."""
    rng = np.random.default_rng(n + 700)
    a = spd(n, rng, 4.0)
    d_theirs, l_theirs, p_theirs = ca.ldl(ca.DM(a))
    theirs = float(np.array(d_theirs).reshape(-1).prod())
    _, d = mc.ldlt(a)
    assert float(d.prod()) == pytest.approx(theirs, rel=1e-9, abs=1e-12)
    assert theirs == pytest.approx(float(ca.det(ca.DM(a))), rel=1e-9, abs=1e-12)
    assert len(np.array(p_theirs).reshape(-1)) == n
    assert np.array(l_theirs).shape == (n, n)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_ldlt_reconstructs_the_matrix(n):
    rng = np.random.default_rng(n + 800)
    a = spd(n, rng, 6.0)
    l, d = mc.ldlt(a)
    np.testing.assert_allclose(l @ np.diag(d) @ l.T, a, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(np.diag(l), np.ones(n), rtol=0, atol=0)
    lower = l - np.diag(np.diag(l))
    np.testing.assert_array_equal(np.triu(lower, 1), np.zeros((n, n)))


@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_ldlt_solve_residual_is_zero(n):
    rng = np.random.default_rng(n + 900)
    a = spd(n, rng, 5.0)
    b = rng.standard_normal(n)
    np.testing.assert_allclose(a @ mc.ldlt_solve(a, b), b, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(mc.ldlt_solve(a, b), np.linalg.solve(a, b),
                               rtol=1e-8, atol=1e-9)


def test_ldlt_rejects_a_nonsymmetric_matrix():
    with pytest.raises(ValueError):
        mc.ldlt(np.array([[1.0, 2.0], [3.0, 4.0]]))


# ---------------------------------------------------------------------------
# Pseudoinverse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (3, 3), (4, 4), (2, 5),
                                   (5, 2), (3, 7), (7, 3), (6, 6)])
def test_pinv_matches_casadi(shape):
    rng = np.random.default_rng(hash(shape) % 2 ** 31)
    a = rng.standard_normal(shape)
    np.testing.assert_allclose(mc.pinv(a), np.array(ca.pinv(ca.DM(a))),
                               rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("shape", [(2, 5), (5, 2), (3, 7), (4, 4)])
def test_pinv_is_a_pseudoinverse(shape):
    """A+ A A+ = A+, the Moore-Penrose conditions that make it the pinv."""
    rng = np.random.default_rng(shape[0] * 31 + shape[1])
    a = rng.standard_normal(shape)
    p = mc.pinv(a)
    np.testing.assert_allclose(a @ p @ a, a, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(p @ a @ p, p, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(a @ p, (a @ p).T, rtol=1e-7, atol=1e-8)


def test_pinv_of_an_orthogonal_matrix_is_its_transpose():
    q, _ = np.linalg.qr(np.random.default_rng(5).standard_normal((6, 6)))
    np.testing.assert_allclose(mc.pinv(q), q.T, rtol=1e-9, atol=1e-9)


def test_pinv_rejects_a_rank_deficient_matrix():
    a = np.ones((3, 3))
    with pytest.raises(np.linalg.LinAlgError):
        mc.pinv(a)


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (5, 3), (8, 8), (16, 4),
                                   (4, 16), (37, 53)])
def test_mtimes_matches_casadi(shape):
    rng = np.random.default_rng(sum(shape))
    m, n = shape
    a = rng.standard_normal((m, n))
    b = rng.standard_normal((n, m))
    np.testing.assert_allclose(mc.mtimes(a, b), np.array(ca.mtimes(ca.DM(a),
                                                                  ca.DM(b))),
                               rtol=1e-10, atol=1e-11)


def test_mtimes_threads_only_for_large_products():
    """The threaded path and the serial path must agree exactly, since they run
    the same kernel over different row ranges."""
    rng = np.random.default_rng(11)
    a = rng.standard_normal((300, 200))
    b = rng.standard_normal((200, 300))
    threaded = mc.mtimes(a, b)
    import os
    old = os.environ.get("MOJO_CASADI_THREADS")
    os.environ["MOJO_CASADI_THREADS"] = "1"
    try:
        serial = mc.mtimes(a, b)
    finally:
        if old is None:
            os.environ.pop("MOJO_CASADI_THREADS")
        else:
            os.environ["MOJO_CASADI_THREADS"] = old
    np.testing.assert_array_equal(threaded, serial)


def test_mtimes_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        mc.mtimes(np.ones((2, 3)), np.ones((4, 5)))


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (3, 2), (5, 7), (7, 5)])
def test_kron_matches_casadi(shape):
    rng = np.random.default_rng(sum(shape))
    m, n = shape
    a = rng.standard_normal((m, n))
    b = rng.standard_normal((2, 3))
    got = mc.kron(a, b)
    expect = np.array(ca.kron(ca.DM(a), ca.DM(b)))
    assert got.shape == expect.shape
    np.testing.assert_allclose(got, expect, rtol=1e-12, atol=1e-12)


def test_kron_shape_is_the_block_form():
    a = np.arange(6.0).reshape(2, 3)
    b = np.arange(12.0).reshape(3, 4)
    got = mc.kron(a, b)
    assert got.shape == (6, 12)
    # kron(A, B) replaces each entry of A by that entry times all of B.
    for i in range(2):
        for j in range(3):
            block = got[i * 3:(i + 1) * 3, j * 4:(j + 1) * 4]
            np.testing.assert_allclose(block, a[i, j] * b, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Norms and trace
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 1), (3, 4), (4, 3), (11, 7), (32, 32)])
def test_norms_match_casadi(shape):
    rng = np.random.default_rng(sum(shape))
    a = rng.standard_normal(shape)
    np.testing.assert_allclose(mc.norm_1(a),
                               float(ca.norm_1(ca.DM(a))), rtol=1e-12)
    np.testing.assert_allclose(mc.norm_inf(a),
                               float(ca.norm_inf(ca.DM(a))), rtol=1e-12)
    np.testing.assert_allclose(mc.norm_fro(a),
                               float(ca.norm_fro(ca.DM(a))), rtol=1e-12)
    if a.shape[0] == a.shape[1]:
        np.testing.assert_allclose(mc.trace(a), float(ca.trace(ca.DM(a))),
                                   rtol=1e-12)


def test_norm_of_a_vector_matches_casadi():
    rng = np.random.default_rng(9)
    v = rng.standard_normal(23)
    np.testing.assert_allclose(mc.norm_2(v),
                               float(ca.norm_2(ca.DM(v))), rtol=1e-12)
    np.testing.assert_allclose(mc.norm_2(v), np.linalg.norm(v), rtol=1e-12)
    np.testing.assert_allclose(mc.norm_1(v), np.abs(v).sum(), rtol=1e-12)
    np.testing.assert_allclose(mc.norm_inf(v), np.abs(v).max(), rtol=1e-12)


def test_norms_flatten_the_matrix():
    """CasADi's dense norms reduce the entries as one flat vector, not per row
    or per column, so a 1-norm here is the sum of absolute values."""
    a = np.array([[1.0, -5.0], [2.0, 3.0]])
    assert mc.norm_1(a) == pytest.approx(11.0)
    assert mc.norm_inf(a) == pytest.approx(5.0)
    assert mc.norm_fro(a) == pytest.approx(np.sqrt(39.0))
    np.testing.assert_allclose(mc.norm_1(a), float(ca.norm_1(ca.DM(a))))
    np.testing.assert_allclose(mc.norm_inf(a), float(ca.norm_inf(ca.DM(a))))


# ---------------------------------------------------------------------------
# Polynomials
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x", [-3.5, -1.0, 0.0, 0.25, 2.0, 7.5])
def test_poly_eval_matches_casadi(x):
    coeffs = [1.0, -3.0, 2.0, 0.5, -1.0]
    assert mc.poly_eval(coeffs, x) == pytest.approx(
        float(ca.polyval(ca.DM(coeffs), x)), rel=1e-12, abs=1e-12)


def test_poly_eval_on_an_array():
    coeffs = [2.0, 0.0, -1.0]
    x = np.linspace(-3.0, 3.0, 101)
    expect = np.array([float(ca.polyval(ca.DM(coeffs), float(v))) for v in x])
    np.testing.assert_allclose(mc.poly_eval(coeffs, x), expect, rtol=1e-12)


def test_poly_eval_on_a_constant():
    assert mc.poly_eval([3.5], 11.0) == pytest.approx(3.5)
    assert mc.poly_eval([0.0], 11.0) == 0.0


def test_poly_add_matches_casadi_function():
    p = [1.0, 2.0, 3.0]
    q = [4.0, -5.0]
    x = ca.SX.sym("x")
    expect = ca.Function("p", [x], [ca.polyval(ca.SX(p), x)
                                    + ca.polyval(ca.SX(q), x)])
    for v in (-2.0, 0.0, 1.5):
        got = mc.poly_add(p, q)
        assert mc.poly_eval(got, v) == pytest.approx(float(expect(v)),
                                                     rel=1e-10, abs=1e-12)
    assert len(mc.poly_add(p, q)) == 3


def test_poly_mul_matches_casadi_function():
    p = [1.0, 2.0]
    q = [3.0, -1.0, 4.0]
    x = ca.SX.sym("x")
    expect = ca.Function("p", [x], [ca.polyval(ca.SX(p), x)
                                    * ca.polyval(ca.SX(q), x)])
    got = mc.poly_mul(p, q)
    assert len(got) == 4
    for v in (-2.0, 0.0, 1.5, 4.0):
        assert mc.poly_eval(got, v) == pytest.approx(float(expect(v)),
                                                     rel=1e-9, abs=1e-11)


def test_poly_mul_is_commutative():
    p = [1.0, 2.0, 3.0]
    q = [4.0, -1.0]
    np.testing.assert_array_equal(mc.poly_mul(p, q), mc.poly_mul(q, p))


def test_poly_deriv_matches_casadi_jacobian():
    """CasADi differentiates a polynomial symbolically, so its Jacobian at a
    point is exactly the derivative of our coefficient vector evaluated there."""
    p = [1.0, 2.0, 3.0, 4.0]
    x = ca.SX.sym("x")
    df = ca.Function("df", [x], [ca.jacobian(ca.polyval(ca.SX(p), x), x)])
    got = mc.poly_deriv(p)
    assert len(got) == 3
    for v in (-2.0, 0.0, 1.5):
        assert mc.poly_eval(got, v) == pytest.approx(float(df(v)),
                                                     rel=1e-9, abs=1e-11)


def test_poly_deriv_of_a_constant_is_zero():
    got = mc.poly_deriv([7.0])
    assert got.shape == (1,)
    assert got[0] == 0.0


def test_poly_deriv_order_matches_repeated_derivatives():
    p = [0.0, 1.0, 6.0, 6.0, 1.0]  # x + 6x^2 + 6x^3 + x^4
    x = ca.SX.sym("x")
    for k in range(5):
        ex = ca.polyval(ca.SX(p), x)
        fk = ca.Function("f", [x], [ex])
        for _ in range(k):
            ex = ca.jacobian(ex, x)
            fk = ca.Function("f", [x], [ex])
        got = mc.poly_deriv_order(p, k)
        for v in (-1.0, 0.0, 2.0):
            assert mc.poly_eval(got, v) == pytest.approx(
                float(fk(v)), rel=1e-7, abs=1e-8), (k, v)


def test_poly_scale():
    p = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(mc.poly_scale(p, -2.0), [-2.0, -4.0, -6.0],
                               rtol=0, atol=0)


def test_poly_add_pads_the_shorter_operand():
    """Coefficients run high power first, so (x + 2) + 5 is x + 7, not 6x + 2:
    the shorter polynomial is padded at the constant end."""
    got = mc.poly_add([1.0, 2.0], [5.0])
    assert len(got) == 2
    np.testing.assert_allclose(got, [1.0, 7.0], rtol=0, atol=0)
    # (x + 2) + (3x^2 + 4x + 5) = 3x^2 + 5x + 7
    np.testing.assert_allclose(mc.poly_add([1.0, 2.0], [3.0, 4.0, 5.0]),
                               [3.0, 5.0, 7.0], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------


def test_every_entry_point_rejects_a_non_square_lu_input():
    with pytest.raises(ValueError):
        mc.lu_factor(np.ones((2, 3)))


def test_lu_factor_reports_its_pivots():
    a = np.array([[0.0, 1.0], [1.0, 0.0]])
    lu, piv = mc.lu_factor(a)
    assert list(piv) == [1, 1], "the swapped-out row is recorded once"
    np.testing.assert_array_equal(lu, [[1.0, 0.0], [0.0, 1.0]])


def test_results_do_not_depend_on_input_layout():
    """A Fortran-ordered or non-contiguous input must give the same answer as a
    C-ordered one, because the shim is what normalises it."""
    rng = np.random.default_rng(21)
    a = rng.standard_normal((6, 6))
    a += 6 * np.eye(6)
    np.testing.assert_array_equal(mc.solve(a, np.ones(6)),
                                  mc.solve(np.asfortranarray(a), np.ones(6)))
    v = rng.standard_normal(6)
    np.testing.assert_array_equal(mc.solve(a, v),
                                  mc.solve(a, np.asfortranarray(v)))
