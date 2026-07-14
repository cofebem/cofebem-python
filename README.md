# CoFEBEM

CoFEBEM is a research code for solving frictionless contact problems through a
non-intrusive flexibility (compliance) formulation. A conventional finite
element model supplies the response operator on a potential contact boundary;
the contact problem is then solved outside the finite element code as a linear
complementarity problem (LCP).

The current finite element interface targets FEniCSx. The repository also
contains a NumPy H-matrix implementation for compressing the dense boundary
compliance operator and experimental BEM, MFEM, interpolation, and
semi-analytical paths.

Developers: V. A. Yastrebov and Y. Boye.

## Formulation

For contact forces `p`, initial gap `g`, and boundary compliance `S_c`, the
frictionless normal-contact problem is

```text
p >= 0
w = S_c @ p + g >= 0
p.T @ w = 0
```

`S_c` is obtained either by repeatedly solving the constrained linear elastic
system with unit boundary loads or, experimentally, by inverting a condensed
Schur complement. Once `p` is known, a boundary mass solve converts nodal
forces into a FEniCSx traction field and the original finite element problem is
solved normally. The FEM formulation therefore does not need an embedded
contact law.

See [architecture](docs/architecture.md) for the derivation and the exact
current data flow.

## Environment and installation

The working environment is the Conda environment `fenicsx-env`. Activate it
before installing the package in editable mode:

```bash
conda activate fenicsx-env
python -m pip install -e .
```

For development and unit tests:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/unit_tests/hmatrices tests/unit_tests/lcp
```

Keep DOLFINx, PETSc, and MPI supplied by Conda. At the time of this
documentation, the local `fenicsx-env` uses DOLFINx 0.9.0, while the optional
dependency metadata in `pyproject.toml` declares 0.10.0. Assembly APIs used by
this project are version-sensitive, so verify the active environment before
changing the FEniCSx bridge.

## Minimal FEniCSx run

The smallest end-to-end example is
[`cofebem/pipeline_fenicsx_minimal.py`](cofebem/pipeline_fenicsx_minimal.py):

```bash
conda run -n fenicsx-env python cofebem/pipeline_fenicsx_minimal.py
```

It builds a linear-elastic cube, samples the top-surface compliance, solves
contact against a sphere, applies the resolved traction, and writes a PVD
result. Run the current contact examples in serial unless a specific path has
been made MPI-safe.

### Dihedral tyre-road example

`examples/tyre_dihedral_contact.py` generates a full structured hexahedral tyre
from `geo_files/geometry_v2.geo`, with configurable axial and circumferential
divisions. It samples two transverse load directions only on one axial
reference meridian, reconstructs the global vertical compliance using the
tyre's discrete dihedral symmetry only when ACA requests an entry, builds a
symmetric H-matrix directly from those queries, applies the internal inflation
preload by linear superposition, and solves contact against the road plane
with hierarchical matvecs. The default contact solver is projected
preconditioned CG (`ppcg`) with a tyre-sector spectral preconditioner:

```bash
conda run -n fenicsx-env python examples/tyre_dihedral_contact.py \
  --axial-divisions 24 \
  --circumferential-divisions 32 \
  --regenerate
```

Use `--pcg-preconditioner none` to measure the unpreconditioned projected
method, or `--contact-solver ccg_v2` for the previous face-by-face baseline.

By default, the example builds and solves only the part of the tyre whose
inflation-adjusted free gap is within `--warning-distance 0.02` of the road.
It certifies excluded nodes with a chunked full-target evaluation and expands
the zone around any penetration before repeating the restricted solve. Use
`--warning-distance inf` for the previous global-surface path. See the
[potential contact zone](docs/potential_contact_zone.md) for the formulation,
tuning, and output fields.

Use `--sampling-only` to stop after constructing the H-matrix. The saved
`compliance.npz` contains the sampled reference tensor and H-matrix statistics,
not a dense global `S_c`. Reuse those samples without repeating the compliance
solves with:

```bash
conda run -n fenicsx-env python examples/tyre_dihedral_contact.py \
  --axial-divisions 24 --circumferential-divisions 32 \
  --load-compliance results/tyre_dihedral/compliance.npz
```

The current mesh/contact ordering and recorded elastic constants must match the
archive; indentation and inflation pressure may change. Generated meshes,
arrays, and VTK output are written below `results/tyre_dihedral/`. The example
detects when this default mesh has a different density from the requested
divisions and regenerates it automatically. A custom `--mesh` path is only
overwritten when `--regenerate` is explicit. Circumferential and axial division
counts must both be even. For mesh generation alone, edit the two constants in
`examples/generate_tyre_dihedral_mesh.py` and run that short script.

## Main packages

- `cofebem/fenics`: FEniCSx contact adapters and linear-system helpers.
- `cofebem/contact`: compliance sampling and the legacy contact-solver API
  currently used by the FEniCSx adapters.
- `cofebem/hmatrices`: cluster trees, block-cluster trees, ACA/SVD low-rank
  approximations, and H-matrix operations.
- `cofebem/lcp`: validated LCP model, solver results, dispatcher, and modern
  PSOR/PGS, NNLS, Lemke, CCG, and PPCG implementations.
- `cofebem/bodies`: analytical gap functions for rigid indenters.
- `examples`: end-to-end studies, validation scripts, and benchmarks; several
  require local meshes/data and should be treated as research scripts.

See [project structure](docs/project_structure.md) for a maturity-oriented map
and [the FEniCSx workflow](docs/fenicsx_workflow.md) for integration details.

## Current H-matrix status

`cofebem.hmatrices.HMatrix` can be built either from a dense array or from a
`MatrixEntrySource`. In the latter path, inadmissible near-field leaves request
only their local blocks, while admissible leaves use partial ACA requests for
selected rows and columns. `LCP` retains such a matrix operator and the CCG/PPCG
solvers consume its hierarchical matvec directly. `HMatrix.solve()` itself
still uses dense LU and is not the contact-solve path.

The direct entry-source path is exercised end to end by the dihedral tyre
example. The generic FEniCSx `Contact.solve()` adapter remains on its legacy
dense compliance path. See [direct symmetry construction](docs/hmatrix_symmetry.md)
for indexing, complexity, and diagnostics.
The [PPCG solver note](docs/ppcg.md) describes the projected PR+ iteration and
the sector-spectral preconditioner.

## Known scope

The most reliable dependency-light components are `cofebem.hmatrices` and
`cofebem.lcp`; together with the dihedral compliance tests, their focused suite
currently contains 301 passing tests. The
FEniCSx adapters are prototypes with important assumptions: 3D vector CG1-like
spaces, direct PETSc solves, serial-oriented indexing, and use of private
`LinearProblem` attributes. The architecture documentation records these
constraints so that future work can remove them deliberately.
