import numpy as np
import time
import matplotlib.pyplot as plt

from dolfinx.mesh import (
    CellType,
    GhostMode,
    create_box,
    locate_entities_boundary,
)
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
    TrialFunction,
    TestFunction,
    sym,
    grad,
    inner,
    tr,
    dx,
)
from dolfinx.io import XDMFFile, VTXWriter, VTKFile, gmshio

from mpi4py import MPI
from petsc4py import PETSc
from tqdm import tqdm

import mpi4py.MPI as MPI

from cofebem.contact.Sc import Sc
from cofebem.contact.rigid_indenters import parabolic
from cofebem.contact.lcp_solvers.ccg import CCG
from cofebem.contact.lcp_solvers.lemke import lemkelcp

# -------------------------------------------------------------------------------------------------------
#  Mesh and material parameters
# -------------------------------------------------------------------------------------------------------

# nx = 20
# ny = 20
# nz = 10

W = 40.0
H = 20.0

# mesh = create_box(
#     MPI.COMM_WORLD,
#     [np.array([0.0, 0.0, 0.0]), np.array([l, l, l])],
#     [nx, ny, nz],
#     CellType.hexahedron,
#     ghost_mode=GhostMode.shared_facet,
# )

# import meshio, sys

# mesh = meshio.read("hertz_cube.msh")

# print("Cell blocks in file:")
# for cb in mesh.cells:  # cb  is a CellBlock
#     print(f"  {cb.type}: {len(cb.data)} elements")
#     if cb.type != "tetra":
#         sys.exit("❌ non-tet cell detected – dolfinx will crash")
# print("✅ only tet4 present – safe for dolfinx")

import meshio

mesh = meshio.read("hertz_cube.msh")

print("Cell blocks present:")
for cb in mesh.cells:  # cb is a CellBlock
    print(f"  {cb.type:7s} : {len(cb.data)} elements")


mesh, cell_tags, facet_tags = gmshio.read_from_msh(
    "hertz_cube.msh", MPI.COMM_WORLD, 0, gdim=3
)


tdim = mesh.topology.dim
fdim = tdim - 1

E = 1.0e9
nu = 0.3

lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))

# -------------------------------------------------------------------------------------------------------
#  LE weak form
# -------------------------------------------------------------------------------------------------------

element_type = "Lagrange"
element_degree = 1

V = functionspace(mesh, (element_type, element_degree, (mesh.geometry.dim,)))

u, v = TrialFunction(V), TestFunction(V)


def epsilon(v):
    return sym(grad(v))


def sigma(u):
    return 2.0 * mu * epsilon(u) + lmbda * tr(epsilon(u)) * Identity(len(u))


def a(u, v):
    return inner(sigma(u), epsilon(v)) * dx


f_v = Constant(mesh, np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType))


def L(v):
    return inner(f_v, v) * dx


# -------------------------------------------------------------------------------------------------------
#  Set boundary conditions
# -------------------------------------------------------------------------------------------------------

tol = 1.0e-5


def Gamma_u_selector(x):
    return np.isclose(x[2], 0, atol=tol)


Gamma_u = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_u_selector)
Iu = locate_dofs_topological(V, entity_dim=fdim, entities=Gamma_u)

u0 = np.array([0, 0, 0], dtype=PETSc.ScalarType)


bc = dirichletbc(
    u0,
    dofs=Iu,
    V=V,
)


# -------------------------------------------------------------------------------------------------------
#  Setup the Problem
# -------------------------------------------------------------------------------------------------------

problem = LinearProblem(
    a=a(u, v), L=L(v), bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)

problem.solve()


# -------------------------------------------------------------------------------------------------------
#  Gamma_c : Contact Region
# -------------------------------------------------------------------------------------------------------

tol = 1.0e-5


# def Gamma_c_selector(x):
#     return (np.isclose(x[2], l, atol=tol)) & ((x[0] - l) ** 2 + (x[1] - l) ** 2 <= 8.0)


def Gamma_c_selector(x):
    return np.isclose(x[2], H, atol=tol)


Gamma_c = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_c_selector)
Ic = locate_dofs_topological(V, fdim, Gamma_c)
Gamma_c_x = mesh.geometry.x[Ic].reshape(-1, tdim)

# -------------------------------------------------------------------------------------------------------
#  Sc construction: Contact compliance matrix
# -------------------------------------------------------------------------------------------------------

# Sc_ = Sc(problem.A, problem.b, tdim, Ic)
# Sc_dense = Sc_.by_sampling()

# Sc_.save(file="./results/cube/Sc.npy")

# -------------------------------------------------------------------------------------------------------
#  Contact Problem
# -------------------------------------------------------------------------------------------------------

Sc_dense = np.load("./results/cube/Sc.npy")

# plt.imshow(Sc_dense, cmap="viridis", aspect="auto")
# # plt.colorbar()
# plt.show()

# delta = 0.2
# R = 0.8

# max_iter = 1000
# tol = 1e-9
# err_type = "nw"

# pfactor = 1e8


# Nframes = 70

# u_fenics = Function(V)
# u_fenics.name = "u"

# p_fenics = Function(V)
# p_fenics.name = "p"


# with VTKFile(mesh.comm, f"./results/cube/cube.pvd", "w") as vtk:
#     vtk.write_mesh(mesh)
#     vtk.write_function([u_fenics, p_fenics], 0)


# ANIMATION = True
# if ANIMATION:
#     x_center = np.linspace(-0.1, 1.1 * W, Nframes)
#     for frame, xc in tqdm(enumerate(x_center)):
#         contact_center = np.array([xc, W / 2])
#         g = (
#             parabolic(
#                 Gamma_c_x[:, 0],
#                 Gamma_c_x[:, 1],
#                 contact_center[0],
#                 contact_center[1],
#                 R,
#                 np.full_like(Gamma_c_x[:, 2], H - delta),
#             )
#             - Gamma_c_x[:, 2]
#         )
#         penetrating_nodes = np.where(g < 0)[0]

#         # p_lemke, _, _ = lemkelcp(Sc_dense, g, maxIter=1000)

#         # if p_lemke is None:
#         #     print("shit")
#         #     continue
#         p_ccg, _, _ = CCG(Sc_dense, err_type, g, max_iter, tol, pfactor).solve()

#         # print(
#         #     f" error = {((np.linalg.norm(p_ccg-p_lemke)/np.linalg.norm(p_lemke)) * 100):.3f} %"
#         # )
#         # Visualization
#         solver_petsc = PETSc.KSP().create(mesh.comm)
#         solver_petsc.setOperators(problem.A)
#         solver_petsc.setType("preonly")
#         solver_petsc.getPC().setType("lu")
#         solver_petsc.setFromOptions()
#         solver_petsc.setUp()

#         b_ = problem.b.copy()
#         u_ = PETSc.Vec().createMPI(b_.getSize(), comm=mesh.comm)

#         b_.set(0)
#         for i, dof in enumerate(Ic):
#             b_.setValue(dof * tdim + 2, p_ccg[i])
#             b_.assemble()

#         solver_petsc.solve(b_, u_)

#         u_fenics.x.array[:] = -u_.array
#         u_fenics.x.scatter_forward()

#         p_fenics.x.array[:] = b_.array
#         p_fenics.x.scatter_forward()

#         vtk.write_function([u_fenics, p_fenics], frame + 1)


# -------------------------------------------------------------------------------------------------------
# Compare with Hertz solution
# -------------------------------------------------------------------------------------------------------

# hx = l / nx
# hy = l / ny
# area_facet = hx * hy
# area_node = area_facet / 4.0

# area_contact_nodes = np.full(len(Gamma_c_x), area_node)


delta = 0.02
R = 100.0

max_iter = 2000
tol = 1e-8
err_type = "nw"

pfactor = 1e8


u_fenics = Function(V)
u_fenics.name = "u"

p_fenics = Function(V)
p_fenics.name = "p"


with VTKFile(mesh.comm, f"./results/cube/cube.pvd", "w") as vtk:
    vtk.write_mesh(mesh, 0.0)
    vtk.write_function([u_fenics, p_fenics], 0.0)


ANIMATION = True
if ANIMATION:

    contact_center = np.array([W / 2, W / 2])
    g = (
        parabolic(
            Gamma_c_x[:, 0],
            Gamma_c_x[:, 1],
            contact_center[0],
            contact_center[1],
            R,
            np.full_like(Gamma_c_x[:, 2], H - delta),
        )
        - Gamma_c_x[:, 2]
    )
    penetrating_nodes = np.where(g < 0)[0]

    p_lemke, _, _ = lemkelcp(Sc_dense, g, max_iter)

    p_ccg, _, _ = CCG(Sc_dense, err_type, g, max_iter, tol, pfactor).solve()

    print(
        f" error = {((np.linalg.norm(p_ccg-p_lemke)/np.linalg.norm(p_lemke)) * 100):.3f} %"
    )

    # Visualization
    solver_petsc = PETSc.KSP().create(mesh.comm)
    solver_petsc.setOperators(problem.A)
    solver_petsc.setType("preonly")
    solver_petsc.getPC().setType("lu")
    solver_petsc.setFromOptions()
    solver_petsc.setUp()

    b_ = problem.b.copy()
    u_ = PETSc.Vec().createMPI(b_.getSize(), comm=mesh.comm)

    b_.set(0)
    for i, dof in enumerate(Ic):
        b_.setValue(dof * tdim + 2, p_lemke[i])
        b_.assemble()

    solver_petsc.solve(b_, u_)

    u_fenics.x.array[:] = -u_.array
    u_fenics.x.scatter_forward()

    p_fenics.x.array[:] = b_.array
    p_fenics.x.scatter_forward()

    vtk.write_function([u_fenics, p_fenics], 1.0)


facet2verts = mesh.topology.connectivity(fdim, 0)

facet_area = np.zeros(len(Gamma_c), dtype=np.float64)
area_node = np.zeros(mesh.geometry.x.shape[0], dtype=np.float64)

for local_i, facet in enumerate(Gamma_c):
    verts = facet2verts.links(facet)
    x0, x1, x2 = mesh.geometry.x[verts]

    area_f = 0.5 * np.linalg.norm(np.cross(x1 - x0, x2 - x0))
    facet_area[local_i] = area_f

    share = area_f / 3.0
    for v in verts:
        area_node[v] += share


x_c = Gamma_c_x


vertex_on_Gc = Ic  # // tdim

p_press = p_lemke / area_node[vertex_on_Gc]


p_num = p_press

E_star = E / (1.0 - nu**2)
a_theo = np.sqrt(R * delta)
P_theo = 4 / 3 * E_star * np.sqrt(R) * delta**1.5
p0_theo = 3 * P_theo / (2 * np.pi * a_theo**2)

p_cut = 0.0001 * p_num.max()


r = np.linalg.norm(x_c[:, :2] - contact_center, axis=1)
contact_nodes = p_num > 0  # p_num > p_cut
a_num = r[contact_nodes].max()


P_num = np.sum(p_num * area_node[vertex_on_Gc])

error_a = np.linalg.norm(a_num - a_theo) / np.linalg.norm(a_theo)
error_P = np.linalg.norm(P_num - P_theo) / np.linalg.norm(P_theo)

print(f"a_num  = {a_num}")
print(f"a_theo = {a_theo}")

print(f"P_num  = {P_num}")
print(f"P_theo = {P_theo}")

print(f"difference on contact patch radius = {(error_a* 100):.3f} %")
print(f"difference on load = {(error_P* 100):.3f} %")


r_contact = r[contact_nodes]
p_contact = p_press[contact_nodes]

nbins = 300
bins = np.linspace(0, a_theo, nbins + 1)
r_mid = 0.5 * (bins[:-1] + bins[1:])
p_avg = np.zeros(nbins)

for i in range(nbins):
    m = (r_contact >= bins[i]) & (r_contact < bins[i + 1])
    p_avg[i] = p_contact[m].mean() if np.any(m) else np.nan

r_th = np.linspace(0, a_theo, 400)
p_th = p0_theo * np.sqrt(1 - (r_th / a_theo) ** 2)

plt.figure(figsize=(5, 4))
plt.plot(r_mid, p_avg, "o", label="FE (binned)")
plt.plot(r_th, p_th, "-", label="Hertz")
plt.xlabel(r"radial distance $r$")
plt.ylabel(r"normal pressure $p(r)$")
plt.title("Pressure distribution on $\Gamma_c$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------------------------------
# a_num vs a_theo as a function of the approach delta
# ------------------------------------------------------------------------------------------------------

a_theos = []
a_nums = []

deltas = np.linspace(0.01, 0.08, 20)

R = 50.0

max_iter = 2000
tol = 1e-8
err_type = "nw"

pfactor = 1e8


contact_center = np.array([W / 2, W / 2])

for delta in deltas:
    g = (
        parabolic(
            Gamma_c_x[:, 0],
            Gamma_c_x[:, 1],
            contact_center[0],
            contact_center[1],
            R,
            np.full_like(Gamma_c_x[:, 2], H - delta),
        )
        - Gamma_c_x[:, 2]
    )
    penetrating_nodes = np.where(g < 0)[0]

    p_lemke, _, _ = lemkelcp(Sc_dense, g, max_iter)

    p_ccg, _, _ = CCG(Sc_dense, err_type, g, max_iter, tol, pfactor).solve()

    print(
        f" error = {((np.linalg.norm(p_ccg-p_lemke)/np.linalg.norm(p_lemke)) * 100):.3f} %"
    )

    p_press = p_lemke / area_node[vertex_on_Gc]

    p_num = p_press

    a_theo = np.sqrt(R * delta)
    p_cut = 0.0001 * p_num.max()

    r = np.linalg.norm(x_c[:, :2] - contact_center, axis=1)
    contact_nodes = p_num > 0  # p_num > p_cut
    a_num = r[contact_nodes].max()

    error_a = np.linalg.norm(a_num - a_theo) / np.linalg.norm(a_theo)

    a_theos.append(a_theo)
    a_nums.append(a_num)

    print(f"a_num  = {a_num}")
    print(f"a_theo = {a_theo}")

    print(f"difference on contact patch radius = {(error_a* 100):.3f} %")


a_theos = np.array(a_theos)
a_nums = np.array(a_nums)


plt.figure(figsize=(5, 4))
plt.plot(deltas, a_theos, "r-", label="a_theo")
plt.plot(deltas, a_nums, "g-", label="a_num")
plt.xlabel(r"approach $\delta$")
plt.ylabel(r"contact patch radius $a$")
plt.title("a_theo vs a_num")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
