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
tyre's discrete dihedral symmetry, applies the internal inflation preload by
linear superposition, and solves contact against the road plane:

```bash
conda run -n fenicsx-env python examples/tyre_dihedral_contact.py \
  --axial-divisions 24 \
  --circumferential-divisions 32 \
  --regenerate
```

Use `--sampling-only` to stop after constructing and saving `S_c`. Generated
meshes, compliance arrays, and VTK output are written below
`results/tyre_dihedral/`. For mesh generation alone, edit the two constants in
`examples/generate_tyre_dihedral_mesh.py` and run that short script.

## Main packages

- `cofebem/fenics`: FEniCSx contact adapters and linear-system helpers.
- `cofebem/contact`: compliance sampling and the legacy contact-solver API
  currently used by the FEniCSx adapters.
- `cofebem/hmatrices`: cluster trees, block-cluster trees, ACA/SVD low-rank
  approximations, and H-matrix operations.
- `cofebem/lcp`: validated LCP model, solver results, dispatcher, and modern
  PSOR/PGS, NNLS, Lemke, and CCG implementations.
- `cofebem/bodies`: analytical gap functions for rigid indenters.
- `examples`: end-to-end studies, validation scripts, and benchmarks; several
  require local meshes/data and should be treated as research scripts.

See [project structure](docs/project_structure.md) for a maturity-oriented map
and [the FEniCSx workflow](docs/fenicsx_workflow.md) for integration details.

## Current H-matrix status

`cofebem.hmatrices.HMatrix` currently receives a fully assembled dense matrix,
compresses geometrically admissible blocks, and provides compressed matrix-
vector products. Thus, it reduces storage and matvec cost after compliance
sampling, but it does not yet eliminate the dense sampling stage. Its `solve()`
method also uses a dense LU factorization internally.

The legacy CCG implementation has an H-matrix matvec path, but the main
FEniCSx `Contact.solve()` adapter is still connected to the dense path. Closing
that integration gap is a principal development direction rather than an
already supported end-to-end feature.

## Known scope

The most reliable dependency-light components are `cofebem.hmatrices` and
`cofebem.lcp`; their focused suite currently contains 267 passing tests. The
FEniCSx adapters are prototypes with important assumptions: 3D vector CG1-like
spaces, direct PETSc solves, serial-oriented indexing, and use of private
`LinearProblem` attributes. The architecture documentation records these
constraints so that future work can remove them deliberately.
