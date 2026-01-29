import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.mesh import locate_entities_boundary, meshtags, create_box, CellType
from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio, VTKFile

from ufl import Identity, Measure, TrialFunction, TestFunction, sym, grad, inner, tr, dx

from cofebem.bodies.sphere_indenter import Sphere
from cofebem.fenics.contact_normal import Contact_normal


# ---------------- Mesh ----------------
mesh, cell_tags, facet_tags = gmshio.read_from_msh(
    "./msh_files/hemisphere2.msh", MPI.COMM_WORLD, 0, gdim=3
)
# L = 1.0
# ncells = 10
# mesh = create_box(
#     MPI.COMM_WORLD,
#     [[0.0, 0.0, 0.0], [L, L, L]],
#     [ncells, ncells, ncells],
#     CellType.tetrahedron,
# )
tdim = mesh.topology.dim
fdim = tdim - 1

# ---------------- Material ----------------
E = 1.0e9
nu = 0.3
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))

# ---------------- Variational forms ----------------
V = functionspace(mesh, ("Lagrange", 1, (tdim,)))
u = TrialFunction(V)
v = TestFunction(V)


def epsilon(w):
    return sym(grad(w))


def sigma(w):
    return lmbda * tr(epsilon(w)) * Identity(tdim) + 2 * mu * epsilon(w)


f_v = Constant(mesh, np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType))
a = inner(sigma(u), epsilon(v)) * dx
L = inner(f_v, v) * dx


# ---------------- Dirichlet BC ----------------
def Gamma_u_locator(x):
    return np.isclose(x[2], 0.0)


Gamma_u = locate_entities_boundary(mesh, fdim, Gamma_u_locator)
Gamma_u_dofs = locate_dofs_topological(V, fdim, Gamma_u)

u0 = np.array([0, 0, 0], dtype=PETSc.ScalarType)
bc = dirichletbc(u0, Gamma_u_dofs, V)
bcs = [bc]


# ---------------- Neumann BC----------------
def Gamma_t_locator(x):
    return np.isclose(x[1], 1.0) & ((x[0] - 0.5) ** 2 + (x[2] - 0.5) ** 2 <= 0.2**2)


Gamma_t = locate_entities_boundary(mesh, fdim, Gamma_t_locator)
Gamma_t_id = 1
Gamma_t_tags = np.full(Gamma_t.shape, Gamma_t_id, dtype=np.int32)

t0 = Constant(mesh, np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType))


# ---------------- Contact BC ----------------
def Gamma_c_locator(x):
    return x[2] > 0.1


Gamma_c = locate_entities_boundary(mesh, fdim, Gamma_c_locator)
Gamma_c_id = 2
Gamma_c_tags = np.full(Gamma_c.shape, Gamma_c_id, dtype=np.int32)

tc = Function(V)

# ---------------------- Setup Neumann and contact contributions to L ----------------
facet_indices = np.hstack([Gamma_t, Gamma_c]).astype(np.int32)
facet_values = np.hstack(
    [
        Gamma_t_tags,
        Gamma_c_tags,
    ]
).astype(np.int32)

order = np.argsort(facet_indices)
facet_indices = facet_indices[order]
facet_values = facet_values[order]

mt = meshtags(mesh, fdim, facet_indices, facet_values)

ds = Measure("ds", domain=mesh, subdomain_data=mt)

L += inner(t0, v) * ds(Gamma_t_id) + inner(tc, v) * ds(Gamma_c_id)

# ---------------- Setup problem ----------------
problem = LinearProblem(
    a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)

problem.u.name = "u"

# ---------------- Setup indentation Scenario ----------------
indenter = Sphere(center=np.array([0.5, 0.5, 1.9]), radius=1.0)

contact = Contact_normal(
    mesh=mesh,
    indenter=indenter,
    tc=tc,
    Gamma_c=Gamma_c,
    ds=ds,
    Gamma_c_id=Gamma_c_id,
    problem=problem,
    solver="lemke",
)

contact.solve(max_iter=1000, tol=1e-6)
contact.apply_contact_forces()
problem.solve()

with VTKFile(
    mesh.comm, f"./results/pipeline_normals/test_fenicsx_normals.pvd", "w"
) as vtk:
    vtk.write_function([problem.u, contact.normal_fn], t=0)

print("Done")
