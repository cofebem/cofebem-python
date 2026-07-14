# CoFEBEM agent guide

This file applies to the whole repository. It is intended for contributors and
coding agents working on the non-intrusive contact solver, the compliance
operator, the H-matrix implementation, or the FEniCSx interface.

## Start here

Read these files before changing a core numerical path:

- `README.md` for setup, scope, and the supported entry points.
- `docs/architecture.md` for the mathematical formulation and data flow.
- `docs/fenicsx_workflow.md` for the current FEniCSx lifecycle and assumptions.
- `docs/project_structure.md` for module ownership and maturity.
- `ScSPD.md` when a change may affect symmetry or positive definiteness.

The central frictionless contact problem is

```text
p >= 0
w = S_c @ p + g >= 0
p.T @ w = 0
```

Here `p` is a vector of nodal contact forces, `g` is the undeformed gap, and
`S_c` maps those forces to normal boundary displacement. Keep this sign
convention explicit in new APIs and tests.

## Environment

The project is currently exercised in the existing Conda environment
`fenicsx-env`:

```bash
conda activate fenicsx-env
python -m pip install -e .
```

For non-interactive commands, use:

```bash
conda run -n fenicsx-env python <script.py>
```

DOLFINx, PETSc, MPI, and their native dependencies should come from the Conda
environment. Do not casually replace them with pip wheels. On 2026-07-14 the
environment contained Python 3.12.3 and DOLFINx 0.9.0, while the `fenicsx`
extra in `pyproject.toml` declares DOLFINx 0.10.0. Verify the active DOLFINx API
before changing assembly code, and do not assume both versions expose the same
`dolfinx.fem.petsc` helpers.

`pytest` was not installed in `fenicsx-env` at the time this guide was written.
Install the development dependencies before running tests:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/unit_tests/hmatrices tests/unit_tests/lcp
```

The H-matrix and LCP tests are dependency-light and should be run for every
change in those packages. Run a focused FEniCSx example for changes to contact
assembly or degree-of-freedom handling. The repository-wide compile check is
not currently clean: `cofebem/fem/backends/fenics.py` and
`examples/schur_vs_sampling.py` contain known syntax errors in experimental
code. Do not treat those failures as regressions unless those files are in the
scope of the task.

## Authoritative code paths

- `cofebem/hmatrices/`: maintained cluster trees, block partitioning,
  low-rank approximations, and H-matrix operations.
- `cofebem/lcp/`: maintained LCP data model, result type, dispatcher, and
  solvers. Prefer this API in new standalone solver code.
- `cofebem/fenics/contact.py`: current flat, global-z FEniCSx contact adapter.
- `cofebem/fenics/contact_normal.py`: current arbitrary-normal prototype.
- `cofebem/fenics/dihedral_compliance.py`: tyre-sector ordering,
  reference-meridian PETSc LU sampling, and road-normal reconstruction.
- `cofebem/contact/Sc.py` and `Sc_normal.py`: compliance sampling from PETSc.
- `cofebem/contact/lcp_solvers/`: legacy solver API used by the FEniCSx contact
  adapters. Preserve compatibility until the adapters migrate to
  `cofebem.lcp`.
- `cofebem/bodies/`: rigid gap models used by examples.
- `cofebem/mesh/tyre_dihedral_hex.py`: tagged D_n-symmetric hex mesh generator
  for `geo_files/geometry_v2.geo`.
- `cofebem/pipeline_fenicsx_minimal.py`: smallest end-to-end example.

Treat most top-level `cofebem/Sc_*.py`, large files under `cofebem/bem/`, and
many scripts under `examples/` as research experiments. They often execute at
import time, depend on local meshes/data, or duplicate newer modules. Avoid
building a new reusable feature inside one of these scripts.

## Numerical invariants

When changing compliance construction or contact solving, preserve and test:

- Index alignment: `points[i]`, `g[i]`, `p[i]`, and row/column `i` of `S_c`
  must refer to the same contact unknown in the same order.
- Reciprocity: a linear elastic compliance should be symmetric up to the
  assembly/solver tolerance. Measure asymmetry before silently symmetrizing.
- Positive definiteness: CCG and NNLS paths require an SPD operator. The
  structure must be sufficiently constrained to remove rigid-body modes.
- Complementarity: report primal violation, dual violation, and
  complementarity, not only an algorithm-specific stopping metric.
- Units: `S_c` maps nodal force to displacement. The boundary mass solve maps
  nodal force to a traction field; do not mix force and pressure silently.
- H-matrix ordering: the coordinate row order must exactly match the matrix
  order. `symmetric=True` is valid only when the source operator is symmetric.
- Approximation safety: report matvec error and storage, and check whether an
  approximation remains SPD when it will be used by an SPD-only solver.

## FEniCSx constraints

The current adapter is narrower than a general FEniCSx interface:

- `Contact` assumes three-dimensional, vector-valued CG1-like spaces and uses
  the z component as the contact direction.
- Several sampling paths assume interleaved global numbering
  `vertex * tdim + component`. Do not extend that assumption to higher-order,
  blocked, mixed, or parallel layouts. Derive mappings from collapsed
  subspaces or PETSc index sets instead.
- The adapters access private `LinearProblem` members (`_A`, `_a`, `_b`,
  `_L`). These are version-sensitive. Isolate new compatibility code rather
  than spreading more private-member access.
- Current sampling and array extraction are effectively serial. Code that
  claims MPI support must use `mesh.comm`, distinguish owned and ghost DOFs,
  and gather/scatter explicitly. Do not substitute `MPI.COMM_WORLD`
  unconditionally.
- The compliance is built in the contact constructor and reused. Rebuild it
  whenever the mesh, material, Dirichlet conditions, stiffness, contact set,
  function space, or DOF ordering changes. Moving only a rigid indenter may
  reuse it.
- For a tyre rotating about x, a fixed global-z road load is not invariant
  under sector rotation. Reconstruct it from the sampled y/z transverse tensor
  as done in `dihedral_compliance.py`; a scalar cyclic `np.roll` is incorrect.

## H-matrix scope

`HMatrix` supports both an already materialized dense matrix and a
`MatrixEntrySource`. The entry-source path keeps near-field leaf blocks dense
but drives admissible blocks with selected partial-ACA row/column queries. Do
not call `to_dense()`, `lu()`, or `solve()` in a direct hierarchical workflow:
all three intentionally assemble a global dense matrix.

The maintained `cofebem.lcp` CCG solvers accept symmetric matrix operators and
therefore provide the hierarchical contact-solve path. Other maintained LCP
solvers require dense matrices. `examples/tyre_dihedral_contact.py` is the
end-to-end reference; the generic `cofebem.fenics.Contact.solve()` adapter is
still connected to the dense legacy path.

## Change and validation practice

- Keep reusable code import-safe: no mesh loading, plotting, solving, or file
  generation at module import time.
- Add new public numerical code under the maintained packages and expose it
  deliberately through `__init__.py`.
- Prefer small deterministic unit tests with seeded random generators.
- For H-matrices, compare `H @ x` with `A @ x`, report relative error, and
  test dense and symmetric-storage paths.
- For LCP solvers, test the returned status and all three feasibility metrics.
- For FEniCSx changes, use a small mesh and verify contact force sign, gap,
  displacement direction, and DOF/coordinate alignment.
- Put generated VTK/PVD/XDMF, NumPy arrays, plots, and benchmark tables under
  `results/` or another ignored output directory. Do not add generated binary
  artifacts unless they are intentional, small test fixtures.
- Preserve unrelated work in the working tree. Many research scripts are
  active notebooks-in-code; avoid broad formatting or mechanical rewrites.
- Update the relevant file under `docs/` whenever an interface, assumption,
  solver constraint, or supported workflow changes.
