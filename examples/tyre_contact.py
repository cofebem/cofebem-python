import numpy as np

from dolfinx.mesh import locate_entities_boundary, meshtags, create_submesh
from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import LinearProblem
from ufl import (
    Identity,
    Measure,
    FacetNormal,
    TrialFunction,
    TestFunction,
    sym,
    grad,
    inner,
    tr,
    dx,
)
from dolfinx.io import XDMFFile, gmshio

from mpi4py import MPI
from petsc4py import PETSc
from tqdm import tqdm

from dolfinx.io import XDMFFile, VTKFile
import mpi4py.MPI as MPI

from cofebem.bodies.plane_indenter import Plane
from cofebem.fenics.contact_normal import Contact_normal
from cofebem.fenics.contact import Contact

# ------------------helpers----------------


def normalize(v):
    return v / np.linalg.norm(v)


def rotation_matrix_from_a_to_b(a, b):
    a = normalize(a)
    b = normalize(b)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < 1e-14:
        if c > 0:
            return np.eye(3)
        # 180° rotation
        e = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, e)) > 0.9:
            e = np.array([0.0, 1.0, 0.0])
        axis = normalize(np.cross(a, e))
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        return np.eye(3) + 2 * (K @ K)

    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    R = np.eye(3) + K + K @ K * ((1 - c) / (s**2))
    return R


# ---- parameters ----
a0 = 0.20
b0 = 0.10
thickness = 0.03
ox = 0.0
oz = 0.5
theta_cut = np.pi / 6

with XDMFFile(MPI.COMM_WORLD, "tyre_hex.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

tdim = mesh.topology.dim
fdim = tdim - 1

E = 2.5e8
nu = 0.48

lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))


xL = ox - a0 * np.cos(theta_cut)
xR = ox + a0 * np.cos(theta_cut)


xtol = 1e-3
ftol = 5e-3
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
    X = x[0]
    Y = x[1]
    Z = x[2]

    r = np.sqrt(Y * Y + Z * Z)

    a_ref = a0 + 0.5 * thickness
    b_ref = b0 + 0.5 * thickness

    theta = np.arctan2((r - oz) / b_ref, (X - ox) / a_ref)

    theta1 = -np.pi + theta_cut
    theta2 = -theta_cut

    tol = 0.15  # radians (~8.5 degrees)

    def ang_dist(a, b):
        return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))

    return (ang_dist(theta, theta1) < tol) | (ang_dist(theta, theta2) < tol)


Gamma_u = locate_entities_boundary(mesh, fdim, Gamma_u_locator)
Gamma_u_set = set(Gamma_u.tolist())
Gamma_u_dofs = locate_dofs_topological(V, fdim, Gamma_u)

u0 = np.array([0, 0, 0], dtype=PETSc.ScalarType)
bc = dirichletbc(u0, Gamma_u_dofs, V)
bcs = [bc]


# ---------------- Neumann BC----------------
def Gamma_t_locator(x):
    X = x[0]
    Y = x[1]
    Z = x[2]
    r = np.sqrt(Y * Y + Z * Z)

    F = ((X - ox) / a0) ** 2 + ((r - oz) / b0) ** 2 - 1.0
    on_inner = np.abs(F) < ftol

    return on_inner


Gamma_t = locate_entities_boundary(mesh, fdim, Gamma_t_locator)
Gamma_t = np.array([f for f in Gamma_t if f not in Gamma_u_set], dtype=np.int32)
Gamma_t_id = 1
Gamma_t_tags = np.full(Gamma_t.shape, Gamma_t_id, dtype=np.int32)

n = FacetNormal(mesh)

p0 = Constant(mesh, PETSc.ScalarType(1.5e5))

t0 = -p0 * n


# ---------------- Contact BC ----------------
def Gamma_c_locator(x):
    X = x[0]
    Y = x[1]
    Z = x[2]
    r = np.sqrt(Y * Y + Z * Z)

    aout = a0 + thickness
    bout = b0 + thickness
    F = ((X - ox) / aout) ** 2 + ((r - oz) / bout) ** 2 - 1.0
    on_outer = np.abs(F) < ftol

    return on_outer


Gamma_c = locate_entities_boundary(mesh, fdim, Gamma_c_locator)
Gamma_c = np.array([f for f in Gamma_c if f not in Gamma_u_set], dtype=np.int32)
Gamma_c_id = 2
Gamma_c_tags = np.full(Gamma_c.shape, Gamma_c_id, dtype=np.int32)
Gamma_c_dofs = locate_dofs_topological(V, fdim, Gamma_c)

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
uniq, first = np.unique(facet_indices, return_index=True)
facet_indices = facet_indices[first]
facet_values = facet_values[first]

mt = meshtags(mesh, fdim, facet_indices, facet_values)

ds = Measure("ds", domain=mesh, subdomain_data=mt)

L += inner(t0, v) * ds(Gamma_t_id) + inner(tc, v) * ds(Gamma_c_id)


# ---------------- Setup problem ----------------
problem = LinearProblem(
    a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)

problem.u.name = "u"

################################################################################
################################################################################
# eps = 1e-7

# Vn = functionspace(mesh, ("CG", 1, (tdim,)))

# n = FacetNormal(mesh)
# u_ = TrialFunction(Vn)
# v_ = TestFunction(Vn)

# a_ = eps * inner(u_, v_) * dx + inner(u_, v_) * ds(Gamma_c_id)
# L_ = inner(n, v_) * ds(Gamma_c_id)

# normal_fn = Function(Vn)
# normal_fn.name = "contact_normals"

# proj = LinearProblem(
#     a=a_,
#     L=L_,
#     bcs=[],
#     u=normal_fn,
#     petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
# )
# proj.solve()
# normal_fn.x.scatter_forward()

# with XDMFFile(mesh.comm, "./results/tyre/contact_normals.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)
#     xdmf.write_function(normal_fn)


# f_map = mesh.topology.index_map(fdim)
# all_facets = f_map.size_local
# all_values = np.zeros(all_facets, dtype=np.int32)
# all_values[mt.indices] = mt.values

# submesh, entity_map, _, _ = create_submesh(mesh, fdim, facet_indices)

# sub_values = all_values[entity_map]

# sub_mt = meshtags(
#     submesh,
#     submesh.topology.dim,
#     np.arange(len(entity_map), dtype=np.int32),
#     sub_values,
# )

# with XDMFFile(mesh.comm, "./results/tyre/sub_tag.xdmf", "w") as xdmf:
#     xdmf.write_mesh(submesh)
#     xdmf.write_meshtags(sub_mt, submesh.geometry)


# mesh.topology.create_connectivity(fdim, 0)
# f_to_v = mesh.topology.connectivity(fdim, 0)


# def facet_vertices(facets):
#     verts = []
#     for f in facets:
#         verts.extend(f_to_v.links(f))
#     verts = np.unique(verts)
#     return mesh.geometry.x[verts]


# pts_u = facet_vertices(Gamma_u)
# pts_t = facet_vertices(Gamma_t)
# pts_c = facet_vertices(Gamma_c)

# np.savetxt(
#     "./results/tyre/Gamma_u_vertices.csv",
#     pts_u,
#     delimiter=",",
#     header="x,y,z",
#     comments="",
# )

# np.savetxt(
#     "./results/tyre/Gamma_t_vertices.csv",
#     pts_t,
#     delimiter=",",
#     header="x,y,z",
#     comments="",
# )
# np.savetxt(
#     "./results/tyre/Gamma_c_vertices.csv",
#     pts_c,
#     delimiter=",",
#     header="x,y,z",
#     comments="",
# )


# print("Saved surface vertex clouds.")


##########################################################################################
##########################################################################################
center = np.array([0, 0, -0.5])
normal = np.array([0, 0, -1])
indenter = Plane(center, normal)
contact = Contact(mesh, indenter, tc, Gamma_c, ds, Gamma_c_id, problem, solver="lemke")


plane_mesh, _, _ = gmshio.read_from_msh(
    "./msh_files/ground.msh", MPI.COMM_WORLD, 0, gdim=3
)

plane_ref_x = plane_mesh.geometry.x[:, :3].copy()
X = plane_ref_x.copy()

V_plane = functionspace(plane_mesh, ("Lagrange", 1))
u_plane = Function(V_plane)
u_plane.name = "indenter"

z0 = np.array([0.0, 0.0, 1.0])  #
with VTKFile(mesh.comm, "./results/tyre/tyre_contact.pvd", "w") as vtk1, VTKFile(
    plane_mesh.comm, "./results/tyre/plane.pvd", "w"
) as vtk2:

    # ---------- step 0 ----------
    indenter.point = np.array([0.0, 0.0, -0.55])
    indenter.normal = np.array([0.0, 0.0, 1.0])

    contact.solve(max_iter=10000, tol=1e-6)
    contact.apply_contact_forces()
    problem.solve()
    vtk1.write_function([problem.u, contact.tc], t=0)

    R = rotation_matrix_from_a_to_b(z0, indenter.normal)
    plane_mesh.geometry.x[:, :3] = (X @ R.T) + indenter.point
    vtk2.write_function(u_plane, t=0)

    # # ---------- step 1 ----------
    indenter.normal = np.array([1.0, 0.0, 1.0])  # tilted
    indenter.normal = indenter.normal / np.linalg.norm(indenter.normal)  # normalize!

    contact.solve(max_iter=10000, tol=1e-6)
    contact.apply_contact_forces()
    problem.solve()
    vtk1.write_function([problem.u, contact.tc], t=1)

    R = rotation_matrix_from_a_to_b(z0, indenter.normal)
    plane_mesh.geometry.x[:, :3] = (X @ R.T) + indenter.point
    vtk2.write_function(u_plane, t=1)

print("DONE")
