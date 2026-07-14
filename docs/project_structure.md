# CoFEBEM project structure

The repository combines maintained numerical packages with exploratory
research scripts. This map describes the current tree and indicates which
areas should host new reusable work.

```text
cofebem-python/
├── cofebem/
│   ├── hmatrices/            maintained H-matrix package
│   │   └── low_rank_approx/  ACA and truncated-SVD implementations
│   ├── lcp/                  maintained LCP model and solver API
│   │   └── solvers/          PSOR/PGS, NNLS, Lemke, CCG, PPCG
│   ├── fenics/               current FEniCSx contact adapters
│   ├── contact/              compliance builders and legacy contact solvers
│   ├── bodies/               analytical rigid-indenter gap models
│   ├── mesh/                 mesh generation and conversion utilities
│   ├── fem/                  experimental backend-neutral FEM abstraction
│   ├── bem/                  analytical/collocation BEM research code
│   ├── utils/                legacy numerical utilities and prototypes
│   ├── pipeline_fenicsx_minimal.py
│   └── Sc_*.py, bilateral_*.py, ...  standalone research studies
├── examples/                 validation cases and benchmarks
├── tests/
│   ├── unit_tests/hmatrices/ maintained H-matrix unit tests
│   ├── unit_tests/lcp/       maintained LCP unit tests
│   └── integration_tests/    early integration-test prototypes
├── docs/                     architecture and contributor documentation
├── geo_files/, msh_files/    geometry and mesh inputs
├── pyproject.toml            primary package metadata
├── setup.py                  legacy package metadata
└── README.md
```

## Maintained numerical core

### `cofebem/hmatrices`

This is the primary location for hierarchical-matrix work.

- `cluster_tree.py`: geometric binary clustering and leaf ordering.
- `block_cluster_tree.py`: admissibility, recursive block partition, dense and
  low-rank block storage.
- `entry_source.py`: protocol for constructing blocks from implicit entry
  queries without a global dense matrix.
- `hmatrix.py`: construction, matvec, arithmetic, dense reconstruction,
  diagnostics, and visualization.
- `low_rank_approx/`: partial ACA, full ACA, ACA+, and truncated SVD.

The matching unit tests are under `tests/unit_tests/hmatrices`.

### `cofebem/lcp`

This is the preferred API for new complementarity-solver work.

- `problem.py`: immutable validated `LCP(M, q)` for dense matrices or matrix
  operators.
- `result.py`: `LCPResult`, statuses, and feasibility diagnostics.
- `preconditioners.py`: mask-aware sector-surface spectral preconditioner.
- `solve.py`: named solver dispatcher.
- `solvers/`: maintained implementations, including operator-only PPCG.

The matching unit tests are under `tests/unit_tests/lcp`.

## Active integration layer

### `cofebem/fenics`

- `contact.py`: flat z-direction contact, dense compliance sampling, boundary
  mass conversion, and legacy LCP dispatch.
- `contact_normal.py`: varying-normal prototype for curved contact surfaces.
- `linear_problem.py`: helper for a small linear-elastic system.
- `dihedral_compliance.py`: reference-meridian y/z sampling with one reusable
  PETSc LU factorization and an entry source that rotates only requested
  global-z compliance entries.

These files rely on DOLFINx private `LinearProblem` attributes and require
focused testing against the active Conda environment.

### `cofebem/contact`

- `Sc.py`: single-direction compliance sampling.
- `Sc_normal.py`: normal-to-normal compliance sampling.
- `lcp_solvers/`: legacy interfaces used by current adapters and examples.
- `rigid_indenters.py` and other modules: older geometry/contact experiments.

The new and legacy LCP packages overlap. Do not add a third solver interface;
migrate adapters toward `cofebem/lcp` with compatibility tests.

### `cofebem/bodies`

Sphere, cone, and plane modules provide gap functions used by the adapters.
Some files remain experimental: `cylinder_indenter.py` performs file I/O and a
full comparison at import time and should not be treated as a library module.

## Experimental areas

### `cofebem/bem`

Contains direct BEM integration, quadrature, singular integration,
FEM-generated compliance comparisons, and several large executable scripts.
It is valuable research history but is not a uniform importable API. New BEM
features should first identify a small reusable kernel and add tests rather
than extending a monolithic script.

### `cofebem/fem`

Contains an in-progress backend-neutral FEM model and backends for FEniCS,
MFEM, and Z-set. It is not the adapter used by the current FEniCSx contact
pipeline, and `cofebem/fem/backends/fenics.py` currently has a syntax error.

### Top-level research scripts

Files such as `cofebem/Sc_approx*.py`, `Sc_parallel*.py`, `bilateral_contact*.py`,
and `neo_hookean.py` explore interpolation, parallel sampling, bilateral
contact, and nonlinear materials. Many execute immediately, generate output,
or require local data. Extract tested library code before depending on them.

### `examples`

Examples range from useful reference cases to incomplete notebooks-in-code.

- `cofebem/pipeline_fenicsx_minimal.py`: preferred minimal flat-contact
  reference.
- `examples/tyre_contact.py`: arbitrary-normal adapter example.
- `examples/tyre_dihedral_contact.py`: structured hex tyre, direct dihedral
  H-matrix construction, and spectrally preconditioned operator PPCG solve; preferred
  tyre-symmetry reference.
- `examples/generate_tyre_dihedral_mesh.py`: short editable wrapper exposing
  the axial and circumferential division counts.
- `examples/hmat_benchmark.py`: dense compliance and H-matrix compression
  benchmark.
- `examples/hmat_complexity*.py`: synthetic H-matrix scaling studies.
- `examples/hertz_validation.py`, cone/punch scripts: analytical validation
  studies with external meshes/data.
- `examples/schur_vs_sampling.py`: incomplete and currently syntactically
  invalid.

## Documentation ownership

- `README.md`: project entry point, installation, and current capability.
- `docs/architecture.md`: formulation, data flow, component boundaries, and
  limitations.
- `docs/fenicsx_workflow.md`: executable integration lifecycle.
- `docs/hmatrix_symmetry.md`: direct H-matrix construction from the sampled
  tyre meridian and operator contact solve.
- `docs/ppcg.md`: projected preconditioned CG and the tyre-sector spectral
  preconditioner.
- `docs/naming_conventions.md`: naming style.
- `ScSPD.md`: mathematical note on LCP uniqueness and SPD preservation.
- `Interpolation_idea.md`: research note, not an implemented contract.
- `AGENTS.md`: operational instructions and numerical guardrails for future
  changes.
