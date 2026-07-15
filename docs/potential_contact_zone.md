# Warning-distance potential contact zone

The tyre surface may contain tens of thousands of normal contact unknowns even
though only a small road-facing patch can contact during one indentation
increment. `examples/tyre_dihedral_contact.py` therefore restricts the
compliance operator and the LCP to a candidate set selected from the current
free gap.

## Restricted problem

After solving the inflation preload, let

```text
g_free = geometric road gap + inflation displacement
K = {i : g_free[i] <= warning_distance}
```

The example builds only the principal compliance operator

```text
S_KK = S_c[K, K]
```

and solves

```text
p_K >= 0
w_K = S_KK @ p_K + g_free[K] >= 0
p_K.T @ w_K = 0.
```

Pressure is identically zero outside `K`. `IndexedEntrySource` maps local ACA
row and column requests back to the global dihedral entry source in the
H-matrix strategy, so neither a global dense compliance nor a global H-matrix
is constructed. In the flexibility-matrix-free strategy, the candidate DOFs
define the injection/extraction maps around a factorized FE solve and no
compliance is stored. The sampled reference-meridian tensor belongs only to
the H-matrix strategy; obtaining it costs `2 * n_axial` PETSc solves,
independently of the number of sectors.

The default warning distance is `0.02` in mesh length units. It is deliberately
generous: too large a value increases H-matrix storage or matrix-free LCP
vector work, whereas too small a value can exclude nodes that should enter
contact. Use `inf` to select the complete surface.

## Full-surface verification and expansion

After a restricted solve, the example certifies the excluded nodes by
evaluating

```text
w_all = S_c[:, K] @ p_K + g_free.
```

For the H-matrix strategy this is evaluated in bounded row chunks directly
from the symmetry entry source. For the flexibility-matrix-free strategy the
full surface displacement is extracted from the cached final FE back-solve.
The rectangular matrix is never stored in either case. Excluded nodes with

```text
w_all < -warning_verification_tol
```

are violations. Their sector/axial halo is added to `K`, the restricted
H-matrix is rebuilt, and the LCP resumes from the scattered previous pressure.
The default limit is five solve/verification rounds. An uncertified final round
raises an error and asks for a larger warning distance instead of silently
accepting penetration or falling back to a global H-matrix.

This follows Hcontact's conservative candidate-set principle but uses the
exact current tyre free gap rather than a prolonged coarse-grid gap. Hcontact's
other key conclusion also carries over: restriction reduces matrix work and
storage, not necessarily conditioning, so the sector spectral preconditioner
remains enabled.

## Restricted spectral preconditioning

The potential zone is generally not a complete rectangular sector-by-axial
grid. `RestrictedProjectedPreconditioner` scatters its residual and projected
free mask into the full regular surface, applies the existing Fourier/cosine
preconditioner, and gathers the candidate entries. Algebraically this is the
principal restriction of the full SPD preconditioner and remains positive on
the restricted free subspace.

The transforms still use full-surface work arrays. Their `O(N log N)` storage
and time are small compared with constructing and repeatedly applying the
global H-matrix, and they preserve the long-wavelength inter-patch modes that
Hcontact found important.

## Command-line use

```bash
conda run -n fenicsx-env python examples/tyre_dihedral_contact.py \
  --axial-divisions 100 \
  --circumferential-divisions 200 \
  --indentation 1e-2 \
  --warning-distance 2e-2
```

Relevant options are:

| option | default | meaning |
|---|---:|---|
| `--warning-distance` | `0.02` | Maximum free gap admitted initially; `inf` selects the full surface. |
| `--warning-halo` | `1` | Periodic-sector/nonperiodic-axial dilation around violations. |
| `--warning-max-rounds` | `5` | Maximum restricted solve and verification rounds. |
| `--warning-verification-tol` | `1e-7` | Allowed negative excluded-node clearance in length units. |

Choose the warning distance from the maximum additional approach expected in
the current increment plus a safety margin. Check the reported candidate
fraction and minimum excluded clearance. A positive excluded clearance well
above the verification tolerance is the desired one-round certificate.

## Outputs

For the H-matrix strategy, `compliance.npz` continues to store the global
reference tensor and global ordered points. It additionally records
`candidate_indices`, warning distance, global/potential unknown counts, and
the final restricted H-matrix statistics. It also records the boundary-
condition identifier, and the tyre loader rejects an archive whose constraint
does not match the current disk-edge clamp. The matrix-free strategy produces
no compliance archive.

`contact_result.npz` and the strategy-specific
`contact_result_{hmatrix,fe_matrix_free}.npz` store full-length `force`, `gap`,
and `clearance` arrays, plus the candidate indices, configuration, timings,
and solve counters. The strategy-specific PVD output contains a scalar
`potential_contact_zone` field equal to one on the final candidate set.

For a restricted run, the reported primal, dual, and complementarity metrics
are computed after scattering pressure to the full surface and performing the
full-target verification. Thus they retain the global contact interpretation.
