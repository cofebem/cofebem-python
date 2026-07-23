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

`examples/tyre_dihedral_contact.py` generates either a full structured uniform
hexahedral tyre or a graded tetrahedral tyre from
`geo_files/geometry_v2.geo`. The graded mesh preserves an exact 60-degree fine
contact surface, uses two 30-degree transitions, and coarsens axially and
circumferentially around the back side; see the
[graded tyre mesh](docs/graded_tyre_mesh.md). The uniform-mesh H-matrix strategy
uses two auxiliary load directions on one
reference meridian to retain the fixed road-normal direction, contracts them
into three scalar-normal sample fields, and builds one symmetric normal
H-matrix directly from ACA entry queries. An alternative flexibility-matrix-
free strategy applies the exact contact compliance by back-solving the already
factorized FE stiffness during every PPCG operator application. Both apply the
internal inflation preload by linear superposition and solve contact against
the road plane with projected preconditioned CG (`ppcg`). Uniform surfaces use
the tyre-sector spectral preconditioner; graded triangular surfaces use an SPD
nodal-area diagonal preconditioner:

The graded mesh additionally tags its regular fine patch as physical surface
204. Setting `hmatrix.local_symmetry_tag` to 204 builds an open-patch H-matrix
only there and validates selected reconstructed columns against direct FE
solves. This is approximate because the coarse remainder is not rotationally
invariant; matrix-free compliance remains the exact option.

```bash
conda run -n fenicsx-env python examples/tyre_dihedral_contact.py \
  --axial-divisions 24 \
  --circumferential-divisions 32 \
  --regenerate
```

The Dirichlet condition fixes all displacement components only on the two
mirrored shortest (3 mm) disk-edge strips. The adjacent 5 mm bead strips are
free. Existing default meshes with the earlier broad bead clamp are detected
and regenerated automatically.

Use `--pcg-preconditioner none` to measure the unpreconditioned projected
method, or `--contact-solver ccg_v2` for the previous face-by-face baseline.
Use the matrix-free strategy without sampling or storing a compliance:

```bash
conda run -n fenicsx-env python examples/tyre_dihedral_contact.py \
  --axial-divisions 24 --circumferential-divisions 32 \
  --compliance-strategy fe_matrix_free
```

See the [flexibility-matrix-free study](docs/flexibility_matrix_free.md) for
the formulation, timing comparison, storage trade-off, and comparison utility.
Two experimental operator backends are also available: `fe_iterative` uses a
strictly checked PETSc CG solve to reduce direct-factor memory, while
`mumps_schur` factors the condensed stiffness of the fixed motion-union
potential zone for fast repeated actions. See
[linear solver backends](docs/linear_solver_backends.md).

The road can be a regular flat floor or a periodic self-affine rough floor
generated with `rfgen`. Tyre nodes are vertically projected onto its bilinear
height field to form the initial gap. The result contains both
`contact_pressure_stress` and `contact_pressure_force_based`, while a separate
VTU stores the regular floor geometry:

```bash
conda run -n fenicsx-env python examples/tyre_dihedral_contact.py \
  --axial-divisions 24 --circumferential-divisions 32 \
  --floor rough --floor-grid-size 128 --roughness-rms 2e-4
```

See [rough floor contact](docs/rough_floor_contact.md) for the rfgen controls,
gap convention, pressure definitions, outputs, and current vertical-contact
scope.

The tyre can remain fixed while the floor is indented, tilted about global
`OY`, twisted about global `OZ`, and translated in `X/Y`. A JSON key-frame
history is linearly interpolated and solved with one reused PETSc
factorization:

```bash
conda run -n fenicsx-env python examples/tyre_dihedral_contact.py \
  --axial-divisions 100 --circumferential-divisions 200 \
  --motion-file examples/tyre_floor_motion.json
```

See [moving-floor tyre contact](docs/floor_motion.md) for the JSON schema,
rotation convention, outputs, and measured LU-persistence decision.

All mesh, material, floor, roughness, solver, H-matrix, potential-zone,
compliance, execution, and motion parameters can instead be supplied by one
structured input:

```bash
python examples/tyre_dihedral_contact.py -in examples/input.json
```

See [complete tyre-contact input](docs/tyre_contact_input.md) for the schema,
relative-path behavior, and CLI override rules.

The normal-contact operator strategies can be reproduced on three small tyre
meshes with an isolated-process benchmark. It compares dihedral and
non-dihedral H-matrices, direct factorized-$K$ actions, and MUMPS selected
Schur actions for one and ten load states, then writes a LaTeX table plus
timing and peak-memory graphs:

```bash
conda run -n fenicsx-env python \
  examples/benchmark_tyre_contact_strategies.py --regenerate-meshes
```

See the [strategy benchmark](docs/tyre_strategy_benchmark.md) for its exact
configuration, metrics, accuracy audit, and interpretation.

For a clearer amortization study without fixed startup and I/O costs, the
larger three-strategy benchmark measures setup and one online contact state,
then projects one, ten, and 100 repeated states:

```bash
conda run -n fenicsx-env python \
  examples/benchmark_tyre_contact_step_projection.py --regenerate-mesh
```

See [larger contact-step projection](docs/tyre_step_projection.md) for the
timing definition, results, accuracy, and limits of the extrapolation.

The memory-oriented 300-axial case in `examples/sm_input.json` uses 118 fine
divisions over 60 degrees and a coarsening factor of 6. It retains the local
angular resolution of the former 700-sector mesh while reducing the complete
mesh to 232 circumferential meridians. Because local grading breaks global
discrete rotational symmetry, this case uses the exact factorized-FE
compliance strategy.

Uniform H-matrix cases default to memory-mapped compliance samples. All large
cases use mass-lumped stress recovery, avoiding a second direct factorization
during postprocessing. Volume fields are streamed to VTK but are not duplicated
in every step NPZ unless requested.

By default, the example builds and solves only the part of the tyre whose
inflation-adjusted free gap is within `--warning-distance 0.02` of the road.
It certifies excluded nodes with a chunked full-target evaluation and expands
the zone around any penetration before repeating the restricted solve. Use
`--warning-distance inf` for the previous global-surface path. See the
[potential contact zone](docs/potential_contact_zone.md) for the formulation,
tuning, and output fields.

Use `--sampling-only` to stop after constructing the H-matrix. The saved
`compliance.npz` contains metadata and H-matrix statistics and refers to the
memory-mapped reference tensor in `compliance_samples.npy`; neither file is a
dense global `S_c`. Reuse those samples without repeating the compliance solves
with:

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
example. The same example can instead expose the factorized FE system as a
matrix operator and perform no compliance construction. The generic FEniCSx
`Contact.solve()` adapter remains on its legacy dense compliance path. See
[direct symmetry construction](docs/hmatrix_symmetry.md) for indexing,
complexity, and diagnostics.
The [PPCG solver note](docs/ppcg.md) describes the projected PR+ iteration and
the sector-spectral preconditioner.

## Known scope

The most reliable dependency-light components are `cofebem.hmatrices` and
`cofebem.lcp`; together with the dihedral compliance tests, their focused suite
currently contains 320 passing tests. The
FEniCSx adapters are prototypes with important assumptions: 3D vector CG1-like
spaces, direct PETSc solves, serial-oriented indexing, and use of private
`LinearProblem` attributes. The architecture documentation records these
constraints so that future work can remove them deliberately.
