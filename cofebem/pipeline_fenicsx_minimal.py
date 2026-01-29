"""
CoFEBEM minimal FEniCSx pipeline with contact against a spherical indenter.
Solves a linear elasticity problem on a unit cube with:
 - Dirichlet BC at the bottom face (z=0)
 - Contact BC on the top face (z=1) against a spherical indenter

The contact is solved using an auxiliary problem to compute contact tractions,
which are then applied to the main variational problem.

Date: January 2026
"""
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.mesh import create_box, CellType, locate_entities_boundary, meshtags
from dolfinx.fem import functionspace, dirichletbc, locate_dofs_topological, Constant, Function
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import VTKFile
from ufl import Identity, TrialFunction, TestFunction, sym, grad, inner, tr, dx, Measure
from cofebem.bodies.sphere_indenter import Sphere
from cofebem.fenics.contact import Contact

# Mesh & Space
mesh = create_box(MPI.COMM_WORLD, [[0.,0.,0.], [1.,1.,1.]], [10,10,10], CellType.tetrahedron)
V = functionspace(mesh, ("Lagrange", 1, (3,)))

# Material & Variational
E, nu = 1e9, 0.3
lmbda, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
u, v = TrialFunction(V), TestFunction(V)
sigma = lambda w: lmbda*tr(sym(grad(w)))*Identity(3) + 2*mu*sym(grad(w))

# Bilinear and linear forms
a = inner(sigma(u), sym(grad(v))) * dx
L = inner(Constant(mesh, np.zeros(3, dtype=PETSc.ScalarType)), v) * dx

# BCs: Fixed bottom (z=0)
Gamma_u = locate_entities_boundary(mesh, 2, lambda x: np.isclose(x[2], 0.0))
bcs = [dirichletbc(np.array([0.,0.,0.], dtype=PETSc.ScalarType), locate_dofs_topological(V, 2, Gamma_u), V)]

# [1] Potential contact surface (top face z=1)
Gamma_c = locate_entities_boundary(mesh, 2, lambda x: np.isclose(x[2], 1.0))
mt = meshtags(mesh, 2, Gamma_c, np.full(Gamma_c.shape, 1, dtype=np.int32))
ds = Measure("ds", domain=mesh, subdomain_data=mt)
# [2] Predefine contact tractions
tc = Function(V)
L += inner(tc, v) * ds(1)

# Formulate weak problem
problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# [3] Introduce indenter
indenter = Sphere(center=[0.5, 0.5, 1.8], radius=1.0)

# [4] Define contact
contact = Contact(mesh, indenter, tc, Gamma_c, ds, 1, problem, solver="lemke")

# [5]Solve auxiliary contact problem
contact.solve(max_iter=1000, tol=1e-6)

# [6] Apply contact forces to the variational problem
contact.apply_contact_forces()

# Solve the problem with the resolved contact forces
problem.solve()

# Visualize results
problem.u.name = "u"
with VTKFile(mesh.comm, "results/minimal_contact.pvd", "w") as vtk:
    mesh.geometry.x[:,:3] += problem.u.x.array.reshape((-1, 3))
    vtk.write_function(problem.u)
