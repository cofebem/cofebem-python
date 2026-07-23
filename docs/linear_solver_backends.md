# Compliance-action linear solver backends

The tyre example can solve the same frictionless LCP with four representations
of the normal compliance action

```text
S_c p = B K^{-1} B.T p.
```

`hmatrix` samples a symmetry-reduced compliance and compresses its potential-
zone block. The other three strategies do not construct a compliance matrix:

| strategy | action used by PPCG | main trade-off |
| --- | --- | --- |
| `fe_matrix_free` | full `PREONLY+LU` back-solve | exact, no compliance storage, expensive action |
| `fe_iterative` | full `CG` solve, normally with `GAMG` | lower factor storage, convergence depends on preconditioning |
| `mumps_schur` | solve a factored condensed contact stiffness | fast repeated action, dense selected Schur storage |

## Selected Schur formulation

After splitting the constrained stiffness into eliminated DOFs `r` and a
fixed potential contact set `c`, MUMPS forms

```text
C = K_cc - K_cr K_rr^{-1} K_rc.
```

The required compliance action is the solution of `C u_c = p_c`; it is not
the multiplication `C @ p_c`. The native bridge calls PETSc's public
factor-Schur functions and retains the complete factor so final volume
recovery can use `MatSolve` without a second resident LU.

The installed petsc4py 3.24 interface does not expose these factor routines.
Build the optional CPython bridge against the PETSc already in `fenicsx-env`:

```bash
conda activate fenicsx-env
COFEBEM_BUILD_PETSC_SCHUR=1 \
  python -m pip install --no-build-isolation -e .
```

The Schur index set is the inflation-adjusted union potential zone over the
complete motion history. It is fixed for every PPCG iteration and load step.
A step-local force is zero-padded into that union; inverting a smaller
principal block would not give the same compliance. If certification finds
contact outside the union, the run stops and asks for a larger
`potential_contact.warning_distance` rather than changing the factor mid-run.

The condensed matrix is generally dense. Before factorization the code
estimates three dense scalar arrays, `3*m*m*sizeof(PetscScalar)`, and rejects a
request above `linear_solver.schur_max_memory_gib`. This estimate is a safety
guard rather than an exact MUMPS peak-memory prediction.

For nonzero inflation the exact Schur set is only known after the inflation
solve. The current robust setup therefore makes a temporary full LU, solves
inflation, releases that factor, and then makes the selected-Schur factor.
The reported factorization time and count include both setups. The cost is
amortized over subsequent contact iterations and motion steps.

## Iterative action

`fe_iterative` defaults to `CG+GAMG`. The matrix is marked symmetric positive
definite, GAMG receives mesh coordinates and six elasticity rigid-body
near-nullspace modes, and every operator application starts from zero. Each
inner solve must converge; failure is fatal. Before PPCG starts, two seeded
operator applications probe reciprocity and positive Rayleigh quotients.

The default inner relative tolerance is `1e-10`. A residual-based stopping
rule is not a mathematically exact linear map, so tolerances must remain much
tighter than the outer contact tolerance. Nearly incompressible displacement
elasticity, such as the tyre default `nu=0.48`, can still require many GAMG
iterations. Treat this backend as an experimental low-memory route and compare
its final complementarity and forces with `fe_matrix_free`.

PETSc command-line options with prefix `cofebem_fe_` override JSON defaults.

## Input and comparison

```json
{
  "compliance": {
    "strategy": "mumps_schur",
    "load": null,
    "factor_solver_type": "mumps",
    "memory_map": true
  },
  "linear_solver": {
    "iterative_ksp_type": "cg",
    "iterative_pc_type": "gamg",
    "relative_tolerance": 1e-10,
    "absolute_tolerance": 1e-14,
    "max_iterations": 2000,
    "options_prefix": "cofebem_fe_",
    "schur_factor_type": "lu",
    "schur_max_memory_gib": 4.0
  }
}
```

Run each strategy in a separate process so peak resident memory is comparable.
Then use:

```bash
python examples/compare_tyre_compliance_strategies.py \
  --include hmatrix fe_matrix_free fe_iterative mumps_schur
```

The report checks that archives describe the same gap and candidate set, then
compares timings, solution errors, active sets, operator solves, inner
iterations, and stored or estimated operator data.
