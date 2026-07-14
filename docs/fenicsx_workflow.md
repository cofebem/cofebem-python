# FEniCSx contact workflow

This guide describes the current CoFEBEM integration, not a general contact
API. The smallest executable reference is
`cofebem/pipeline_fenicsx_minimal.py`.

## Preconditions

The current flat adapter expects:

- a three-dimensional DOLFINx mesh;
- a vector-valued, CG1-like displacement space;
- a linear elastic `dolfinx.fem.petsc.LinearProblem`;
- enough Dirichlet constraints to remove all rigid-body modes;
- a tagged potential contact boundary;
- a vector `Function` used as the contact traction field;
- a rigid indenter with `gap(points)`.

The current code is serial-oriented and uses a direct PETSc LU factorization.
Do not assume it is correct for distributed arrays, mixed spaces, higher-order
elements, or non-interleaved DOF layouts.

## 1. Build the bulk problem

Create the mesh, vector function space, linear-elastic bilinear form, body or
Neumann loads, and essential boundary conditions in normal FEniCSx code. The
stiffness must remain constant while the compliance is reused.

Create a traction function and include it on the potential contact boundary:

```python
tc = Function(V)
L = L_external + inner(tc, v) * ds(contact_tag)

problem = LinearProblem(
    a,
    L,
    bcs=bcs,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)
```

The potential contact surface may contain nodes that never enter contact. The
LCP decides the active set.

## 2. Construct the contact adapter

```python
from cofebem.bodies.sphere_indenter import Sphere
from cofebem.fenics.contact import Contact

indenter = Sphere(center=[0.5, 0.5, 1.8], radius=1.0)

contact = Contact(
    mesh=mesh,
    indenter=indenter,
    tc=tc,
    Gamma_c=contact_facets,
    ds=ds,
    Gamma_c_id=contact_tag,
    problem=problem,
    solver="lemke",
)
```

Construction is expensive. It immediately assembles the constrained bulk
matrix, samples the dense contact compliance, assembles the scalar boundary
mass matrix, and factorizes that mass matrix.

The flat adapter collapses displacement component 2 and evaluates gaps at
those scalar-space coordinates. It is therefore a z-normal contact path.

## 3. Resolve and apply contact

```python
contact.solve(max_iter=1000, tol=1e-6)
contact.apply_contact_forces()
uh = problem.solve()
```

`solve()` evaluates the current gap and solves the auxiliary LCP. It stores the
compressive nodal forces in `contact.fc`. `apply_contact_forces()` solves

```text
Mcc * traction_coefficients = -contact.fc
```

and writes the coefficients into `tc`. Calling `problem.solve()` then performs
the ordinary elastic solve with that resolved traction.

Always call `apply_contact_forces()` after `solve()` and before the bulk solve.

## Repeated indenter positions

Moving a rigid indenter changes the gap but not the compliance. For a sequence
of positions:

```python
for position in positions:
    indenter.center = position
    contact.solve(max_iter=1000, tol=1e-6)
    contact.apply_contact_forces()
    problem.solve()
```

This reuse is valid only if the mesh, material, stiffness, essential boundary
conditions, potential contact surface, and function-space ordering remain
fixed. Construct a new `Contact` object otherwise.

## Curved surfaces

Use `cofebem.fenics.contact_normal.Contact_normal` when the response and gap
must be projected along varying surface normals. It:

1. builds the contact vertex set;
2. projects facet normals into a CG1 vector field;
3. samples the normal-to-normal compliance `Snn`;
4. solves the same scalar LCP;
5. maps scalar traction coefficients back to vector traction with the local
   normals.

The indenter must implement `new_gap_n(points, normals, ...)`. This path is a
prototype and retains the same serial/CG1/private-API caveats as the flat path.

## Solver selection

The adapters currently use legacy solvers:

- `solver="lemke"`: direct complementary pivoting; practical for small dense
  contact systems.
- `solver="ccg"`: contact-oriented constrained conjugate gradient; requires an
  SPD compliance and is intended for larger systems.

The maintained `cofebem.lcp` result API is not yet connected to these adapters.
When migrating, preserve the LCP sign convention `M=S_c`, `q=g`, and expose
status plus primal, dual, and complementarity diagnostics.

## Generic-adapter H-matrix use

H-matrix compression is currently manual:

```python
from cofebem.hmatrices import HMatrix

H = HMatrix(
    contact_points,
    dense_Sc,
    leaf_size=64,
    eta=0.8,
    tol=1e-6,
    lr_approx="aca_partial",
    symmetric=True,
)
```

This generic-adapter call requires `dense_Sc` to exist first. The dihedral tyre
path below instead constructs directly from an entry source. Use `H @ x` for compressed
matvecs and `H.stats()` to inspect storage. Validate the global matvec error and
positive definiteness before using the approximation in CCG/PPCG. The main
`Contact.solve()` method does not yet dispatch to the H-matrix path.

## Dihedral tyre workflow

`examples/tyre_dihedral_contact.py` is the reference for a tyre revolved about
the x axis. The mesh generator creates equal circumferential sectors from the
blocked `geometry_v2.geo` cross-section and tags the outer carcass, bead clamp,
and inner surface.

A fixed global-z road force rotates into both y and z components when mapped
to the zero-angle tyre meridian. The example therefore performs two PETSc LU
solves per axial reference node, samples the complete transverse 2x2 response,
and rotates that tensor to construct the global-z compliance. Repeating a
single scalar row with `np.roll`, as in the older z-axis annulus example, is not
valid for this geometry.

```bash
conda run -n fenicsx-env python examples/tyre_dihedral_contact.py \
  --axial-divisions 24 --circumferential-divisions 32 --regenerate
```

The implementation reports sector alignment, reflection parity, sampled
reciprocity, H-matrix storage, and entry-query counts. Symmetric H-matrix
storage enforces reciprocity without constructing or projecting a global
dense matrix. It is currently serial and uses one reusable PETSc
`PREONLY`+`LU` factorization. Internal
pressure is solved first as a free displacement and added to the initial gap;
the final elastic right-hand side superposes inflation and resolved contact
forces.

The default LCP method is `ppcg`. Its projected free set includes positive
pressure nodes and every zero-pressure node with negative clearance, allowing
many active-set violations to be corrected in one iteration. A periodic
Fourier transform around the tyre and a cosine transform along its axis apply
an SPD inverse-compliance spectral model. Use
`--pcg-preconditioner none` or `--contact-solver ccg_v2` for comparisons.

## Diagnostics

For each new problem, check at least:

```python
asymmetry = np.linalg.norm(Sc - Sc.T) / np.linalg.norm(Sc)
eig_min = np.linalg.eigvalsh(0.5 * (Sc + Sc.T))[0]
w = Sc @ p + g
primal_violation = max(0.0, -p.min())
dual_violation = max(0.0, -w.min())
complementarity = np.linalg.norm(p * w, ord=np.inf)
```

Large asymmetry usually indicates inconsistent load/response ordering or DOF
mapping. A non-positive smallest eigenvalue can indicate an unconstrained
rigid mode, a mapping error, insufficient solver accuracy, or an unsafe
approximation.

## Outputs

Examples commonly write PVD/VTK, XDMF, NumPy arrays, plots, and timing tables.
Write generated artifacts below `results/` (or another ignored directory) and
create the directory before running an example. Do not deform
`mesh.geometry.x` in place unless the script is finished with the reference
configuration or has saved a copy for restoration.
