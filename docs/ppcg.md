# Projected preconditioned CG for tyre contact

The maintained `ppcg` solver addresses the indentation-controlled contact LCP

```text
p >= 0
w = S_c @ p + g >= 0
p.T @ w = 0
```

as the bound-constrained quadratic minimization

```text
min 0.5 * p.T @ S_c @ p + g.T @ p,  subject to p >= 0.
```

It was adapted from the projected preconditioned-CG ideas documented in the
read-only Hcontact reference, while retaining CoFEBEM's prescribed-gap LCP
rather than Hcontact's prescribed-total-load constraint.

Hcontact is a design reference only: CoFEBEM does not import it or require it
at runtime.

## Implementation map

- `cofebem/lcp/solvers/ppcg.py` contains the projected PR+/FR iteration.
- `cofebem/lcp/preconditioners.py` contains
  `SectorSurfaceSpectralPreconditioner`.
- `cofebem/lcp/solve.py` exposes the method as `solve(..., method="ppcg")`.
- `examples/tyre_dihedral_contact.py` constructs the sector preconditioner and
  uses PPCG by default.
- `tests/unit_tests/lcp/test_solvers.py` checks solver behavior and dense/
  operator agreement.
- `tests/unit_tests/lcp/test_preconditioners.py` checks the spectral transform,
  masked positivity, and zero-mode treatment.

## Projected PR+ iteration

At each iterate, `ppcg` evaluates `w = S_c @ p + g` and defines

```text
free = (p > 0) or (w < 0).
```

Thus every penetrating zero-pressure node can enter the free set in the same
iteration. This differs from the older `ccg_v2` face driver, which releases an
active variable after minimizing the current face and consequently performs
many active-set changes on large contact surfaces.

The projected gradient is zero outside `free`. A mask-aware SPD
preconditioner produces `z = B^-1 gradient`; PR+ forms the new search direction
from `z` and restarts whenever the projected free set changes. The quadratic
line search is exact along that direction. The pressure is then projected onto
`p >= 0`, and a local overlap correction introduces any penetrating point that
the projected preconditioned direction left at zero.

More explicitly, let `P_F` be the diagonal projector onto the current free set
and define

\[
r_k=P_Fw_k,\qquad y_k=P_FB^{-1}P_Fr_k,\qquad
\rho_k=r_k^Ty_k.
\]

The default PR+ update is

\[
\beta_k=\max\left(0,
\frac{y_k^T(r_k-r_{k-1})}{\rho_{k-1}}\right),\qquad
d_k=P_F(y_k+\beta_kd_{k-1}).
\]

When the free set changes, or when the overlap correction modifies a projected
step, `beta` is reset to zero. Fletcher--Reeves is also available and replaces
the numerator by `rho_k`, but PR+ is the default because it automatically
discards stale conjugacy after poor nonlinear progress.

For the quadratic objective, the exact step is

\[
\alpha_k=\frac{r_k^Td_k}{d_k^TS_cd_k},\qquad
p_{k+1}=\max(p_k-\alpha_kd_k,0).
\]

Each completed iteration therefore needs two hierarchical operator
applications: one for `w = S_c @ p + g` and one for `S_c @ d`. It never calls
`HMatrix.to_dense()`, `HMatrix.lu()`, or `HMatrix.solve()`.

Convergence uses the same relative natural residual as `ccg_v2`:

```text
norm(min(p, w), inf) / (1 + norm(p, inf) + norm(w, inf)).
```

The result continues to report primal violation, dual violation, and
complementarity independently of that stopping metric.

## Sector-spectral preconditioner

`SectorSurfaceSpectralPreconditioner` expects the tyre's sector-major,
axial-minor ordering. It applies an orthonormal periodic Fourier transform in
the circumferential direction and a cosine transform along the non-periodic
axis. In modal space it multiplies by

```text
sqrt(q_theta**2 + q_x**2 + q_0**2),
```

which approximates the inverse of the elastic compliance's high-frequency
`1 / |q|` symbol. Before and after the transform, the vector is masked to the
current projected free set. The resulting `P B^-1 P` operation remains
symmetric positive definite on that subspace.

The zero mode is deliberately positive. Hcontact can remove it because a
fixed-total-load constraint determines mean pressure separately. CoFEBEM's
indentation-controlled LCP has no such equality constraint, so removing the
constant-pressure mode would make the preconditioner singular and change the
problem being solved.

The preconditioner is approximate: axial nodes need not be uniformly spaced,
and the finite tyre is not an exact periodic half-space. Positive curvature is
still checked on every search direction; loss of SPD is returned as numerical
breakdown.

With `N = n_sectors * n_axial`, one application costs `O(N log N)` and uses
`O(N)` transform storage. The preconditioner does not store an approximation
of `S_c` and does not change H-matrix construction.

When the warning-distance potential zone is enabled, PPCG acts on the reduced
principal compliance. `RestrictedProjectedPreconditioner` scatters the reduced
residual/free mask to the full sector grid, applies the same transform, and
gathers the result. This preserves the full-grid spectral modes and SPD
principal restriction while H-matrix applications and CG vectors remain
candidate-sized. Hcontact's active-set study likewise found that localization
does not replace global spectral preconditioning.

## Usage

```python
from cofebem.lcp import LCP, SectorSurfaceSpectralPreconditioner, solve

preconditioner = SectorSurfaceSpectralPreconditioner(ordering.points)
result = solve(
    LCP(compliance_hmatrix, gap),
    method="ppcg",
    preconditioner=preconditioner,
    tol=1.0e-10,
)
```

The tyre example enables this path by default. Use
`--pcg-preconditioner none` to isolate the projected active-set improvement
and `--contact-solver ccg_v2` to reproduce the previous solver.

The equivalent command-line run is

```bash
conda run -n fenicsx-env python examples/tyre_dihedral_contact.py \
  --axial-divisions 100 \
  --circumferential-divisions 80 \
  --contact-solver ppcg \
  --pcg-preconditioner spectral \
  --pcg-beta-method pr_plus \
  --pcg-zero-mode-factor 1 \
  --regenerate
```

### Tyre PPCG options

| option | default | meaning |
|---|---|---|
| `--contact-solver` | `ppcg` | Select `ppcg`, `ccg_v2`, or legacy `ccg`. |
| `--pcg-preconditioner` | `spectral` | Select the sector-spectral model or `none`. |
| `--pcg-beta-method` | `pr_plus` | Select PR+ or Fletcher--Reeves conjugacy. |
| `--pcg-zero-mode-factor` | `1` | Scale `q_0` relative to the smallest nonzero modal wavenumber. |
| `--tol` | `1e-10` | Relative natural-residual tolerance. |
| `--max-iter` | `10000` | Maximum projected steps. |
| `--warning-distance` | `0.02` | Free-gap threshold for the potential contact zone; `inf` disables restriction. |

`zero-mode-factor=1` is the validated default. It is a preconditioner tuning
parameter and does not alter the LCP operator. Very small values weaken the
constant mode; very large values make the preconditioner closer to uniform
scaling.

## 100×80 tyre benchmark

For the 8,080-unknown case (`axial-divisions=100`,
`circumferential-divisions=80`, H-matrix tolerance `1e-7`, LCP tolerance
`1e-10`), all timings below exclude H-matrix construction:

| method | projected/CG steps | operator applications | solve time |
|---|---:|---:|---:|
| `ccg_v2` | 301, plus 187 face changes | not separately reported | 22.09 s |
| `ppcg`, no preconditioner | 112 | 225 | 6.14 s |
| `ppcg`, sector spectral | 49 | 99 | 2.82 s |

The spectral PPCG result had primal violation `0`, dual violation
`1.80e-8`, and complementarity `4.60e-6`; `ccg_v2` gave `0`, `3.06e-8`, and
`4.50e-6`, respectively. The relative pressure-vector difference was
`2.30e-4` at the common normalized stopping tolerance. These figures are a
recorded validation case, not a grid-independent convergence guarantee.

## Result interpretation

`LCPResult.iterations` counts completed projected steps. The message also
reports operator applications, which is the more useful cost measure for an
H-matrix solve. A successful run should be assessed with all of:

```python
assert result.converged
print(result.residual)          # normalized stopping metric
print(result.primal_violation)  # max(0, -min(p))
print(result.dual_violation)    # max(0, -min(w))
print(result.complementarity)   # norm(p * w, inf)
```

The normalized residual can be small while dimensional dual or
complementarity values appear larger because forces and displacements have
different physical scales. Compare those diagnostics at a common tolerance
and with the same units.

## Assumptions and fallback guidance

Use the sector-spectral preconditioner only when:

- unknowns are ordered by sector and then axial position;
- every sector contains the same axial coordinates;
- sectors are regularly spaced around the global x axis;
- `S_c` is symmetric positive definite, including the H-matrix
  approximation.

The current implementation is serial at the example level. The transforms are
local NumPy/SciPy operations and do not gather a distributed contact vector.

Use unpreconditioned `ppcg` for a non-sector surface or when no suitable SPD
preconditioner is available. Use `ccg_v2` as a regression baseline. If PPCG
returns `numerical_breakdown`, first check H-matrix positive definiteness,
ordering, and the preconditioner's masked positivity; merely disabling the
curvature check is unsafe.

## Validation

Run the maintained numerical suite with

```bash
conda run -n fenicsx-env pytest -q \
  tests/unit_tests/lcp \
  tests/unit_tests/hmatrices \
  tests/unit_tests/test_dihedral_compliance.py
```

The preconditioner tests check masked positivity, retention of the constant
mode, and exact inversion of a matching synthetic spectrum. Solver tests cover
dense and operator LCPs, both beta formulas, malformed preconditioners, and
random SPD comparisons with Lemke. The tyre example supplies the end-to-end
hierarchical/FEniCSx validation.
