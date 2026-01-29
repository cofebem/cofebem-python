"""
CoFEBEM minimal FEniCSx pipeline with contact against a spherical indenter.
Solves a linear elasticity problem on a unit cube with:
 - Dirichlet BC at the bottom face (z=0)
 - Neumann BC on the face x=1
 - Contact BC on the top face (z=1) against a spherical indenter
 - Body forces (gravity)

The contact is solved using an auxiliary problem to compute contact tractions,
which are then applied to the main variational problem.

Date: January 2026
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx.mesh as fenics_mesh
import dolfinx.fem as fenics_fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import VTKFile
import ufl
import cofebem.bodies.sphere_indenter as cbem_bodies
import cofebem.fenics.contact as cbem_contact

# Mesh & Space
msh = fenics_mesh.create_box(MPI.COMM_WORLD, [[0.,0.,0.], [1.,1.,1.]], [10,10,10], fenics_mesh.CellType.tetrahedron)
V = fenics_fem.functionspace(msh, ("Lagrange", 1, (3,)))

# Material, Load & Variational
E, nu = 1.e5, 0.3
gravity  = -3.e4
pressure = 3.e3
delta    = -0.2
Lambda, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

sigma = lambda w: Lambda*ufl.tr(ufl.sym(ufl.grad(w)))*ufl.Identity(3) + 2*mu*ufl.sym(ufl.grad(w))
a = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx

# BCs 
# Dirichlet at the bottom (z=0)
Gamma_u = fenics_mesh.locate_entities_boundary(msh, 2, lambda x: np.isclose(x[2], 0.0))
bcs = [fenics_fem.dirichletbc(np.zeros(3, dtype=PETSc.ScalarType), fenics_fem.locate_dofs_topological(V, 2, Gamma_u), V)]

# Mark all boundaries: Neumann (tag=2), Contact (tag=1)
Gamma_f = fenics_mesh.locate_entities_boundary(msh, 2, lambda x: np.isclose(x[0], 1.0))
Gamma_c = fenics_mesh.locate_entities_boundary(msh, 2, lambda x: np.isclose(x[2], 1.0))

# Combine all boundary markers
facet_indices = np.hstack([Gamma_f, Gamma_c])
facet_markers = np.hstack([np.full(Gamma_f.shape, 2, dtype=np.int32), 
                           np.full(Gamma_c.shape, 1, dtype=np.int32)])
mt = fenics_mesh.meshtags(msh, 2, facet_indices, facet_markers)
ds = ufl.Measure("ds", domain=msh, subdomain_data=mt)

# Body forces + Neumann BC + contact tractions
f_body = fenics_fem.Constant(msh, np.array([0., 0., gravity], dtype=PETSc.ScalarType))
tractions = fenics_fem.Constant(msh, np.array([pressure, 0., 0.], dtype=PETSc.ScalarType))
contact_tractions = fenics_fem.Function(V)

# Linear form
L = ufl.inner(f_body, v) * ufl.dx + ufl.inner(tractions, v) * ds(2) + ufl.inner(contact_tractions, v) * ds(1)

# Formulate weak problem
problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# Solve auxiliary contact problem
indenter = cbem_bodies.Sphere(center=[0.5, 0.5, 2.0+delta], radius=1.0)
contact = cbem_contact.Contact(msh, indenter, contact_tractions, Gamma_c, ds, 1, problem, solver="lemke")
contact.solve(max_iter=1000, tol=1e-6)
contact.apply_contact_forces()

# Solve the main problem
problem.solve()

# Visualize
with VTKFile(msh.comm, "results/minimal_contact.pvd", "w") as vtk:
    msh.geometry.x[:,:3] += problem.u.x.array.reshape((-1, 3))
    vtk.write_function(problem.u)