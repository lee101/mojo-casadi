# mojo-casadi

`mojo-casadi` is the dense numeric core of [CasADi](https://web.casadi.org),
written in Mojo and callable from Python.

CasADi is a symbolic toolbox, and most of what it does is not arithmetic a
compiled kernel can help with: expression graphs, source generation, the IPOPT
interface, ODE integrators, the `Opti` modelling layer, and all of the
sparse linear algebra that lives in its C++ core behind `Function` objects.
What *is* worth compiling is the dense numeric work a CasADi caller ends up
doing on plain NumPy arrays: solving systems, inverting, determinants and
adjugates, the symmetric `L D L^T` factorisation, the pseudoinverse, matrix
products and Kronecker products, the norms, and polynomial coefficient
arithmetic.

Everything here is plain row-major `float64` NumPy in, NumPy out, and every
operation is checked against the real `casadi` in the test suite.

```python
import numpy as np, mojo_casadi as mc

A = np.array([[4.0, 1.0], [1.0, 3.0]])
mc.det(A)                              # 11.0
mc.solve(A, np.array([1.0, 2.0]))      # [0.2, 0.6]
mc.poly_mul([1.0, 1.0], [1.0, -1.0])   # x^2 - 1
```

## Covered subset

| area | implemented API |
| --- | --- |
| LU | `lu_factor`, `lu_solve`, `solve`, `inv` (LU with partial pivoting, LAPACK `ipiv` semantics) |
| Determinants | `det`, `adjugate` |
| Symmetric systems | `ldlt`, `ldlt_solve` (unit-lower `L D Lᵀ`) |
| Pseudoinverse | `pinv` (wide: `Aᵀ(AAᵀ)⁻¹`; tall: `(AᵀA)⁻¹Aᵀ`) |
| Products | `mtimes` (threaded over row ranges for large products), `kron` |
| Reductions | `norm_1`, `norm_2`, `norm_inf`, `norm_fro`, `norm`, `trace` |
| Polynomials | `poly_eval` (Horner), `poly_add`, `poly_scale`, `poly_mul`, `poly_deriv`, `poly_deriv_order` |

## Not implemented

- Anything symbolic: `SX`, `MX`, `DM` construction, `Function`, `jacobian`,
  `gradient`, `hessian`, `jtimes`, code generation. These are CasADi's core
  and belong in its C++, not in a numeric kernel library.
- Anything sparse: `Sparsity`, `Linsol`, `qppsol`, `nlpsol`, IPOPT, the
  integrators, and every `cs_*` routine.
- `expm` and `expm_const`: the `libcasadi_expm_slicot.so` plugin is not shipped
  in the installed wheel, so `casadi.expm` itself raises here and there is
  nothing to compare against.
- `taylor` and `mtaylor`: the Taylor-series coefficient API is the closest
  thing to a symbolic surface worth porting, but `casadi.taylor` raises an
  internal assertion in this build (`Xfunction input arguments must be purely
  symbolic`) for every input form tried, so it cannot be a parity target. The
  coefficient arithmetic it would have driven is implemented here anyway and
  tested against CasADi's symbolic differentiation.
- CasADi's `norm_2` as a matrix spectral norm, and its `norm_1`/`norm_inf` as
  per-row or per-column norms: on a dense `DM` CasADi reduces the entries as
  one flat vector, and that is what is matched. The column-sum and row-sum
  variants that some linear-algebra texts call "the 1-norm" are not provided.
- Eigen decompositions, `sqrtm`, `expm`, matrix functions, and every solver.

## Install

```bash
pixi install
pixi run build      # -> dist/libmojo-casadi.so
pixi run test
```

Set `PYTHONPATH=python` when using the package outside a Pixi task. The Python
package is `mojo_casadi`, so it imports alongside the real `casadi`.

## Tests

```
114 passed
```

`tests/test_parity.py` compares every operation against `casadi` itself, and
where a closed form exists against that too, because a consistently wrong
inverse and a consistently wrong CasADi would otherwise cancel out. The cases
that matter most:

- **`solve` and `inv` on matrices that actually pivot.** Partial pivoting fires
  on a minority of inputs, so a suite built from diagonally dominant matrices
  would pass with the permutation applied in the wrong order. The suite uses
  non-diagonally-dominant matrices and checks the residual both ways.
- **The adjugate's classical identity** `A·adj(A) = det(A)·I`.
- **The Moore-Penrose conditions** `A⁺AA⁺ = A⁺`, plus the symmetry of `AA⁺`.
- **Polynomials against CasADi's symbolic differentiation**: `poly_deriv` is
  compared against `ca.jacobian` of the same polynomial in CasADi's `SX`
  algebra, and `poly_add`/`poly_mul` against CasADi `Function` objects.
- **Threaded and serial `mtimes` must agree exactly**, since they run the same
  kernel over different row ranges.

## Performance

Best-of-N wall clock, correctness gated first. The reference column is
CasADi's own C++ dense kernels reached through `DM` operations, except where
noted. `ldlt_solve` is compared against LAPACK's general solve through NumPy,
because `casadi.ldl_solve` segfaults on a 1×1 system in this build.

| case | reference | mojo-casadi | result |
| --- | ---: | ---: | ---: |
| mtimes 256×256×256 | 76.01 ms | 52.93 ms | 1.4x faster |
| solve 128×128 | 574.50 ms | 0.95 ms | 604.6x faster |
| inv 64×64 | 546.97 ms | 0.68 ms | 798.6x faster |
| det 8×8 | 2303.63 ms | 0.03 ms | 84779.4x faster |
| kron 1600×1600 | 1764.97 ms | 6.90 ms | 255.7x faster |
| pinv 8×300 | 44.74 ms | 0.10 ms | 454.4x faster |
| norm_fro 2000×2000 | 5.30 ms | 8.55 ms | 1.7x slower |
| poly_eval deg=64, n=1M | 593.81 ms | 192.28 ms | 3.1x faster |
| ldlt_solve 200×200 (vs LAPACK) | 1908.12 ms | 10.07 ms | 189.5x faster |

The very large ratios need reading carefully, because they say more about this
CasADi build than about Mojo. `casadi.DM.det` is a cofactor recursion here:
1.5 s at 8×8, and 10×10 does not finish in 60 s. `casadi.inv` takes 0.5 s for
a 64×64. `casadi.kron` and `casadi.pinv` are similarly slow. Mojo's LU is
faster than all of them by a wide margin because a hand-written dense LU is a
straightforward loop and these are not.

The one honest loss is **`norm_fro` at 1.7x slower**. A 2000×2000 reduction is
pure bandwidth: NumPy's pairwise-SIMD reduction is already at memory speed and
a scalar Mojo loop cannot beat it. That is the expected result for a
bandwidth-bound kernel and no amount of threading would help.

`mtimes` at 1.4x is the honest result of a naive triple loop against whatever
CasADi dispatches to. It is threaded above 2^18 multiply-adds, but a blocked
or packed kernel would do better; the port is deliberately plain so the
arithmetic matches the specification exactly.

Reproduce with `pixi run bench`.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-casadi.so`.

The Python layer in `python/mojo_casadi` owns every array. Buffers cross the C
ABI as 64-bit addresses and are rebuilt in Mojo as
`Pointer[Float64, AnyOrigin[mut=True]]`, which keeps the exported symbols
non-parametric.

`mtimes` is the only chunked kernel: `mc_gemm_rows` takes a row range and the
shim fans out over a `ThreadPoolExecutor` when the product is large enough to
pay for it (above 2^18 multiply-adds). The bandwidth-bound reductions are left
serial, because threading them makes them slower.

Four details that are easy to get wrong and are pinned by tests:

- **The pivot vector follows LAPACK's `ipiv` convention**: entry `k` is the row
  that landed in slot `k`, or `k` itself when no interchange happened. Only
  slot `k` is ever written. Writing a swap into both slots corrupts the entry
  for a later step, and the result is then wrong only for the minority of
  matrices that pivot at all — which is why a suite of diagonally dominant
  matrices would never catch it.
- **The Gram matrix in `pinv` differs by branch.** A wide matrix needs `AAᵀ`
  (summing over columns); a tall one needs `AᵀA` (summing over rows). They are
  different matrices and the result looks plausible either way.
- **Polynomial coefficients run from the highest power down**, as in
  `casadi.polyval` and NumPy, so `poly_add` aligns at the constant term. Left
  alignment silently computes a different polynomial.
- **CasADi's dense norms flatten the matrix.** `norm_1` is the sum of absolute
  values, not the maximum column sum.

Mojo emits FMA, so results agree with CasADi to a tolerance rather than
bit-for-bit. The tests set the tolerance per operation — a sum of `k` products
accumulates roughly `k·eps` of relative error, so a long reduction gets a
looser bound than a short one.

## License

MIT
