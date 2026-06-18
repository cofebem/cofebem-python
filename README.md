# CoFEBEM
<p align="center">
<img src="https://github.com/cofebem/cofebem-python/blob/main/logo.png" alt="CoFEBEM logo" style="width:20%; border:0;">
</p>

This package implements the Flexibility method to simulate general contact mechanics problems in infinitesimal strain and linear regime (for the moment). To construct the compliance operator we make use of open-source FEM Libraries, particulary FEniCSx. To overcome the scaling bottleneck of the method we rely on $\mathcal(H)-matrices$ to compress and speedup the linear alegbra.

**Developers:** V. A. Yastrebov, Y. Boye

The currently most reusable parts are:

- LCP problem definitions and solvers (`lemke`, `psor`, `pgs`, `nnls`, `ccg`).
- H-matrix data structures for block clustering and low-rank matrix
  approximation.
- FEM, BEM, mesh, and contact modules used by the example workflows.
- FEniCS/FEniCSx-oriented scripts for constructing contact compliance matrices
  and validating indentation/contact problems.

The repository is still research-code oriented: some examples expect local mesh
or result files, and the heavier FEM/BEM workflows require optional scientific
packages that may be easiest to install in a dedicated conda or containerized
environment.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/cofebem/cofebem-python.git
cd cofebem-python
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e ".[dev]"
```

The package metadata in `pyproject.toml` targets Python 3.12 and includes the
core dependencies:

- `numpy`
- `scipy`
- `matplotlib`
- `numba`
- `fenicsx`
- `meshio`
- `tqdm`

The current top-level package import also loads the mesh module, so install
`meshio` for normal `cofebem.*` imports:

```bash
python -m pip install meshio
```

Some examples and modules also use packages such as `dolfinx`, `ufl`,
`petsc4py`, `mpi4py`, `h5py`, `numpy-stl`, and `pyvista`. Install those only if
you need the corresponding FEM/FEniCSx or visualization workflow.

The project also defines a FEniCSx extra:

```bash
python -m pip install -e ".[fenicsx]"
```

Depending on your platform, FEniCSx/DOLFINx may be easier to install through
conda, Docker, or your HPC environment rather than through plain `pip`.

## Quick Start

### Solve a Linear Complementarity Problem

```python
import numpy as np

from cofebem.lcp import LCP, solve

M = np.array([[2.0, 1.0], [1.0, 2.0]])
q = np.array([-1.0, -1.0])

problem = LCP(M, q)
result = solve(problem, method="lemke")

print(result.z)
print(result.w)
print(result.converged)
```

Available LCP solver names are:

- `lemke`
- `psor`
- `pgs`
- `nnls`
- `ccg`
- `ccg_v2`

### Build an H-Matrix Approximation

```python
import numpy as np

from cofebem.hmatrices import HMatrix

n = 128
pts = np.linspace(0.0, 1.0, n).reshape(-1, 1)
dist = np.abs(pts[:, 0, None] - pts[:, 0][None, :])
A = np.exp(-5.0 * dist) + n * np.eye(n)

H = HMatrix(
    pts,
    A,
    leaf_size=16,
    eta=0.8,
    tol=1e-6,
    lr_approx="aca_partial",
)

x = np.ones(n)
y = H @ x

print(H.stats())
```

Supported low-rank approximation methods include:

- `aca_partial`
- `aca_full`
- `aca_plus`
- `truncated_svd`

## Repository Layout

```text
cofebem/
  bem/                 Boundary element kernels, quadrature, and operators
  bodies/              Rigid/deformable body and indenter definitions
  contact/             Contact compliance and solver experiments
  fem/                 FEM abstractions and backend interfaces
  hmatrices/           Cluster trees, block cluster trees, H-matrices
  lcp/                 LCP problem/result classes and solvers
  mesh/                Mesh utilities and geometry generation helpers
  utils/               Linear algebra, ACA, and matrix approximation helpers

examples/              Validation, benchmark, and contact mechanics scripts
tests/                 Unit and integration tests
docs/                  Project notes and conventions
geo_files/, msh_files/ Geometry and mesh assets
```

## Examples

Example scripts live in `examples/`. They cover topics such as:

- Hertz contact validation.
- Cone, punch, hemisphere, and tyre contact setups.
- H-matrix complexity and benchmarking.
- LCP solver benchmarking.
- FEniCSx contact compliance matrix extraction.

Run an example from the repository root, for example:

```bash
python examples/hmat_complexity2.py
```

Many examples are exploratory scripts and may require mesh files, generated
`.npy` data, or optional FEM/FEniCSx dependencies. If an example imports
`dolfinx`, `petsc4py`, `mpi4py`, or `meshio`, make sure those packages are
available in the active environment.

## Tests

Install the development dependencies and run:

```bash
python -m pytest
```

To run focused test groups:

```bash
python -m pytest tests/unit_tests/lcp
python -m pytest tests/unit_tests/hmatrices
```

The unit tests currently exercise the LCP API, solver behavior, mesh/function
space utilities, and H-matrix construction/operations.


## License

This project is licensed under the BSD 3-Clause License. See `LICENSE` for the
full license text.
