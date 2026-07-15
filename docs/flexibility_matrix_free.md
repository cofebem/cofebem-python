# Flexibility-matrix-free contact

## Formulation

Let `A` be the constrained finite-element stiffness matrix and let `R_K`
inject nodal normal forces on the potential contact zone `K` and extract the
corresponding normal displacements. The exact restricted compliance is

```text
S_KK = R_K @ A^{-1} @ R_K.T.
```

The flexibility-matrix-free strategy never forms `S_KK`. For every operator
application requested by PPCG it performs

```text
f = R_K.T @ p
A @ u = f                  # reusable PETSc PREONLY+LU factorization
S_KK @ p = R_K @ u.
```

`FactorizedComplianceOperator` implements this action and declares symmetry
when force and response DOFs are identical. It reuses PETSc RHS/displacement
vectors, bypasses an exactly zero force vector, and caches the most recent FE
solution. The cache makes full-surface warning-zone verification normally an
extraction from the final PPCG solve rather than another back-solve.

This operator remains an exact action of the discretized FE compliance up to
the direct-solver tolerance. It stores neither the reference-meridian sample
tensor nor H-matrix blocks. The factorized FE stiffness is still stored and is
common to both strategies.

## Tyre usage

```bash
conda run -n fenicsx-env python examples/tyre_dihedral_contact.py \
  --axial-divisions 100 \
  --circumferential-divisions 200 \
  --indentation 1e-2 \
  --warning-distance 2e-2 \
  --compliance-strategy fe_matrix_free
```

The alternative is

```text
--compliance-strategy hmatrix
```

which remains the default. `--load-compliance` and `--sampling-only` apply
only to the H-matrix strategy. Both strategies use the same inflation-adjusted
gap, potential-zone indices, PPCG implementation, restricted sector-spectral
preconditioner, global force scattering, and final FE solve.

Strategy-specific results are saved as

```text
contact_result_hmatrix.npz
contact_result_fe_matrix_free.npz
```

and can be compared with

```bash
python examples/compare_tyre_compliance_strategies.py
```

## 100×200 comparison

Configuration: 100 axial divisions, 200 sectors, 10 mm indentation, 20 mm
warning distance, 1,425 candidate nodes out of 20,200, PPCG with the restricted
sector-spectral preconditioner, H tolerance `1e-7`, LCP tolerance `1e-10`.
The floor is flat at `z=0`; floor generation and pressure post-processing are
common to both strategies and excluded from the strategy timings below.
Only the two mirrored 3 mm disk-edge strips are fixed
(`disk_edge_short_curves_v1`); the adjacent bead surface is free.
These are single serial runs on the same workstation and should be interpreted
as an algorithmic comparison rather than a portable performance claim.

| stage (s) | cold H-matrix | loaded H-matrix | FE matrix-free |
|---|---:|---:|---:|
| LU factorization | 30.496 | 30.491 | 30.306 |
| compliance load/sample | 23.230 | 0.054 | 0 |
| operator build | 1.310 | 1.385 | ~0 |
| PPCG contact solve | 0.947 | 0.886 | 22.505 |
| full-surface verification | 0.680 | 0.651 | ~0 (cached) |
| strategy total | **56.977** | **33.748** | **52.925** |

The cold H-matrix arm performs 202 unit-load FE back-solves before PPCG. The
matrix-free arm performs 200 operator applications: 198 FE back-solves, one
zero-vector bypass, and one cached full-target verification extraction. The
loaded H-matrix arm performs no compliance back-solves during this run.

| result | H-matrix | FE matrix-free |
|---|---:|---:|
| PPCG iterations | 92 | 99 |
| active nodes | 155 | 155 |
| total force | 6596.72947 | 6596.72684 |
| global dual violation | `1.01e-8` | `7.65e-9` |
| complementarity | `1.25e-6` | `5.85e-7` |

Agreement relative to the exact-action matrix-free result:

```text
force relative L2       = 3.48e-5
force relative Linf     = 4.18e-5
clearance absolute Linf = 1.37e-8
active-set differences  = 0
```

The H representation stores 853,108 floating-point entries for the restricted
operator (about 6.5 MiB before Python/block overhead) and the saved global
reference tensor contains 62.3 MiB of raw floating-point data. The matrix-free
strategy stores zero compliance entries. Both retain the PETSc LU factors,
which dominate common memory and factorization time for this mesh.

## Interpretation

For a single cold solve, matrix-free is slightly faster here because 198 PPCG
back-solves cost less than 202 compliance-sampling back-solves plus H-matrix
construction. It also removes all compliance storage and ACA approximation.

Once the sampled tensor exists, the H-matrix path is substantially faster:
the representation-specific H build, solve, and verification take about
2.9 s versus 22.5 s for matrix-free PPCG. Therefore:

- prefer flexibility-matrix-free for one-off contact states, memory-limited
  runs, rapidly changing stiffness, or when no compliance archive exists;
- prefer the H-matrix when many gaps/indentations reuse the same linear
  stiffness and contact ordering, because sampling is amortized and each PPCG
  iteration becomes cheap;
- retain the warning-distance restriction in both cases; it reduces H storage
  and LCP vector work, while matrix-free FE back-solve cost is mostly governed
  by the factorized bulk system rather than candidate count.

The current implementation is serial. Distributed use requires owned/ghost
DOF mappings and explicit PETSc gather/scatter handling before it can claim MPI
support.
