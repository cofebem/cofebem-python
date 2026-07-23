# Direct H-matrix construction from tyre symmetry

The dihedral tyre path constructs one scalar normal-contact H-matrix. It does
not construct a tangential contact H-matrix and does not assemble the global
vertical contact compliance before compression.

## Sampled data and indexing

`sample_reference_normal_compliance()` returns

```text
samples[normal_component, source_axial, target_sector, target_axial]
```

The three stored components are `yy`, `yz + zy`, and `zz`. They are exactly
the terms needed by scalar normal contact; the antisymmetric transverse term
is discarded. Legacy complete 2x2 archives remain readable and are compacted
when memory-mapped. Contact unknowns use sector-major order, so global index
`k` represents

```text
sector = k // n_axial
axial  = k % n_axial
```

For a requested matrix entry with target row `i` and source column `j`, let
`delta = (target_sector - source_sector) mod n_sectors` and

```text
q = [sin(source_angle), cos(source_angle)].
```

`DihedralComplianceEntrySource` evaluates

```text
S_c[i, j] = q_y^2 samples[yy] + q_y q_z samples[yz + zy]
            + q_z^2 samples[zz]
```

This rotation is necessary because the road normal is fixed in global `z` as
the tyre meridian rotates about `x`.

## Hierarchical construction and solve

`HMatrix.from_entry_source()` builds the cluster and block-cluster trees from
the ordered contact coordinates. Inadmissible near-field leaves request their
small dense blocks. Each admissible block runs partial ACA and requests only
its selected residual rows and columns. With symmetric storage, only one
triangle is built and reciprocal blocks are applied by transposition.

The contact problem is passed through the maintained API as

```python
result = solve(
    LCP(Sc_h, gap),
    method="ppcg",
    preconditioner=sector_spectral_preconditioner,
)
```

PPCG only invokes `Sc_h @ vector`; it does not call `HMatrix.solve()` or
`HMatrix.to_dense()`. CCG variants also accept operators; dense-only LCP
methods are rejected for operator problems.

For tyre-road runs, `IndexedEntrySource` normally restricts this global entry
oracle to the principal indices selected by the warning-distance potential
contact zone. The cluster tree, block tree, near-field blocks, ACA crosses,
and LCP vectors then all have the candidate size rather than the complete
surface size. A separate chunked full-target evaluation certifies excluded
clearances without storing `S[:, K]`; see
[`potential_contact_zone.md`](potential_contact_zone.md).

## Diagnostics and limitations

The example reports stored H-matrix entries, low-rank and near-field block
counts, source query count, and the largest requested Cartesian block. The
saved `compliance.npz` refers to compact normal samples in
`compliance_samples.npy` and contains these statistics, not a dense `S_c`.
Pass that file through
`--load-compliance results/tyre_dihedral/compliance.npz` to skip repeated PETSc
unit-load sampling on a later run. The hierarchy and ACA factors are rebuilt
from the saved tensor using the current H-matrix command-line settings; no
global dense matrix is materialized.

The current implementation is serial and uses a reusable PETSc LU
factorization for the `2 * n_axial` auxiliary sampling solves. Both directions
are needed to rotate a fixed global-z road force exactly, but they feed one
normal operator rather than separate normal and tangential H-matrices. ACA
tolerance is local to each admissible block and does not by itself guarantee
that the approximation remains positive definite. PPCG detects non-positive
search curvature and reports numerical breakdown if that invariant is lost.

## Optional local symmetry patch

For a graded tyre, physical tag `204` identifies only the regular 60-degree
road-facing surface. `LocalDihedralComplianceEntrySource` uses the same
reference-meridian samples on this open patch, without periodic wraparound,
and obtains the reverse orientation by Maxwell--Betti reciprocity. The
H-matrix, ACA queries and LCP unknowns are restricted to this tag. Tags `201`
and `204` are still combined for pressure integration and for checking that
contact has not escaped the fine patch.

Local regularity does not make the globally condensed FE operator rotationally
invariant: the coarse remainder still affects patch displacement. The example
therefore performs configurable direct-FE column validation and reports the
error. Use the factorized-FE matrix-free strategy when that approximation is
unacceptable.
