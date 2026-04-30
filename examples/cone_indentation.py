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

from cofebem.bodies.cone_indenter import Cone
from cofebem.fenics.contact import Contact


# ---------------- Mesh ----------------

nx = 20
ny = 20
nz = 5

l = 1

mesh = create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([l, l, l])],
    [nx, ny, nz],
    CellType.hexahedron,
)
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
    return np.isclose(x[2], 1.0)


Gamma_c = locate_entities_boundary(mesh, fdim, Gamma_c_locator)
Gamma_c_id = 2
Gamma_c_tags = np.full(Gamma_c.shape, Gamma_c_id, dtype=np.int32)

tc = Function(V)
tc.name = "$p_{c}$"

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
delta = 0.3
R = 0.5  # top radius
H = 1.0  # height (top -> apex)

indenter = Cone(
    top_center=np.array([0.5, 0.5, l - delta]),
    top_radius=R,
    height=H,
)

contact = Contact(
    mesh=mesh,
    indenter=indenter,
    tc=tc,
    Gamma_c=Gamma_c,
    ds=ds,
    Gamma_c_id=Gamma_c_id,
    problem=problem,
    solver="lemke",
)

n_frames = 20
xcs = np.linspace(0, l, n_frames)

cone_mesh, _, _ = gmshio.read_from_msh(
    "./msh_files/cone.msh", MPI.COMM_WORLD, 0, gdim=3
)

cone_ref_x = cone_mesh.geometry.x[:, :3].copy()

X = cone_ref_x.copy()
X[:, 0] *= indenter.radius
X[:, 1] *= indenter.radius
X[:, 2] *= indenter.height

V_cone = functionspace(cone_mesh, ("Lagrange", 1))
u_cone = Function(V_cone)
u_cone.name = "indenter"


with VTKFile(mesh.comm, "./results/cone_indent/cone_indent.pvd", "w") as vtk1, VTKFile(
    cone_mesh.comm, "./results/cone_indent/cone.pvd", "w"
) as vtk2:
    for k, xc in enumerate(xcs):
        print(f"Frame {k+1}/{n_frames}: Indenter top_center x = {xc:.3f}")

        indenter.top_center = np.array([xc, 0.5, l + H - delta])

        contact.solve(max_iter=1000, tol=1e-6)
        contact.apply_contact_forces()
        problem.solve()

        vtk1.write_function([problem.u, contact.tc], t=k)

        cone_mesh.geometry.x[:, :3] = X + indenter.top_center
        vtk2.write_mesh(cone_mesh, t=k)
        print("k =", k, "top_center =", indenter.top_center, "apex =", indenter.apex)
        # vtk2.write_function(u_cone, t=k)

print("Done")
