import numpy as np
import matplotlib.pyplot as plt
import meshio

from dolfinx.mesh import (
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
from dolfinx.io import VTKFile, gmshio

from mpi4py import MPI
from petsc4py import PETSc

import mpi4py.MPI as MPI

from cofebem.contact.Sc import Sc
from cofebem.contact.rigid_indenters import parabolic
from cofebem.contact.lcp_solvers.ccg import CCG
from cofebem.contact.lcp_solvers.lemke import lemkelcp

# -------------------------------------------------------------------------------------------------------
#  Mesh and material parameters
# -------------------------------------------------------------------------------------------------------
# 0 = 0.20 ; 1 = 0.08 ; 2 = 0.02

W = 40.0
H = 20.0

mesh, cell_tags, facet_tags = gmshio.read_from_msh(
    "hertz_cube_1.msh", MPI.COMM_WORLD, 0, gdim=3
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
#     return (np.isclose(x[2], H, atol=tol)) & ((x[0] - W/2) ** 2 + (x[1] - W/2) ** 2 <= 8.0)


def Gamma_c_selector(x):
    return np.isclose(x[2], H, atol=tol)


Gamma_c = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_c_selector)
Ic = locate_dofs_topological(V, fdim, Gamma_c)
Gamma_c_x = mesh.geometry.x[Ic].reshape(-1, tdim)

# -------------------------------------------------------------------------------------------------------
#  Sc construction: Contact compliance matrix
# -------------------------------------------------------------------------------------------------------

Sc_ = Sc(problem.A, problem.b, tdim, Ic)
Sc_dense = Sc_.by_sampling()

Sc_.save(file="./results/hertz/Sc_1.npy")

# Sc_dense = np.load("./results/hertz/Sc_0.npy")


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


with VTKFile(mesh.comm, f"./results/hertz/hertz1.pvd", "w") as vtk:
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

    # p_ccg, _, _ = CCG(Sc_dense, err_type, g, max_iter, tol, pfactor).solve()

    # print(
    #     f" error = {((np.linalg.norm(p_ccg-p_lemke)/np.linalg.norm(p_lemke)) * 100):.3f} %"
    # )

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


# facet2verts = mesh.topology.connectivity(fdim, 0)

# facet_area = np.zeros(len(Gamma_c), dtype=np.float64)
# area_node = np.zeros(mesh.geometry.x.shape[0], dtype=np.float64)

# for local_i, facet in enumerate(Gamma_c):
#     verts = facet2verts.links(facet)
#     x0, x1, x2 = mesh.geometry.x[verts]

#     area_f = 0.5 * np.linalg.norm(np.cross(x1 - x0, x2 - x0))
#     facet_area[local_i] = area_f

#     share = area_f / 3.0
#     for v in verts:
#         area_node[v] += share


# x_c = Gamma_c_x

# p_press = p_lemke / area_node[Ic]


# p_num = p_press

# E_star = E / (1.0 - nu**2)
# a_theo = np.sqrt(R * delta)
# P_theo = 4 / 3 * E_star * np.sqrt(R) * delta**1.5
# p0_theo = 3 * P_theo / (2 * np.pi * a_theo**2)

# p_cut = 0.0001 * p_num.max()


# r = np.linalg.norm(x_c[:, :2] - contact_center, axis=1)
# contact_nodes = p_num > 0  # p_num > p_cut
# a_num = r[contact_nodes].max()


# P_num = np.sum(p_num * area_node[vertex_on_Gc])

# error_a = np.linalg.norm(a_num - a_theo) / np.linalg.norm(a_theo)
# error_P = np.linalg.norm(P_num - P_theo) / np.linalg.norm(P_theo)

# print(f"a_num  = {a_num}")
# print(f"a_theo = {a_theo}")

# print(f"P_num  = {P_num}")
# print(f"P_theo = {P_theo}")

# print(f"difference on contact patch radius = {(error_a* 100):.3f} %")
# print(f"difference on load = {(error_P* 100):.3f} %")


# r_contact = r[contact_nodes]
# p_contact = p_press[contact_nodes]

# nbins = 300
# bins = np.linspace(0, a_theo, nbins + 1)
# r_mid = 0.5 * (bins[:-1] + bins[1:])
# p_avg = np.zeros(nbins)

# for i in range(nbins):
#     m = (r_contact >= bins[i]) & (r_contact < bins[i + 1])
#     p_avg[i] = p_contact[m].mean() if np.any(m) else np.nan

# r_th = np.linspace(0, a_theo, 400)
# p_th = p0_theo * np.sqrt(1 - (r_th / a_theo) ** 2)

# fig1, ax1 = plt.subplots(figsize=(6, 4))

# ax1.plot(r_mid, p_avg, "o", label="FE (binned)")
# ax1.plot(r_th, p_th, "-", label="Hertz")
# ax1.set_xlabel(r"radial distance $r$")
# ax1.set_ylabel(r"normal pressure $p(r)$")
# ax1.set_title("Pressure distribution on $\Gamma_c$")
# ax1.legend()
# ax1.grid(True)
# plt.tight_layout()

# fig1.savefig("hertz_vs_cofebem.png", format="png")
# plt.show()


# # ------------------------------------------------------------------------------------------------------
# # a_num vs a_theo as a function of the approach delta
# # ------------------------------------------------------------------------------------------------------

# a_theos = []
# a_nums = []

# P_theos = []
# P_nums = []

# deltas = np.linspace(0.01, 0.08, 20)

# R = 50.0

# max_iter = 5000
# tol = 1e-8
# err_type = "nw"

# pfactor = 1e8


# contact_center = np.array([W / 2, W / 2])

# for delta in deltas:
#     g = (
#         parabolic(
#             Gamma_c_x[:, 0],
#             Gamma_c_x[:, 1],
#             contact_center[0],
#             contact_center[1],
#             R,
#             np.full_like(Gamma_c_x[:, 2], H - delta),
#         )
#         - Gamma_c_x[:, 2]
#     )
#     penetrating_nodes = np.where(g < 0)[0]

#     p_lemke, _, _ = lemkelcp(Sc_dense, g, max_iter)

#     # p_ccg, _, _ = CCG(Sc_dense, err_type, g, max_iter, tol, pfactor).solve()

#     # print(
#     #     f" error = {((np.linalg.norm(p_ccg-p_lemke)/np.linalg.norm(p_lemke)) * 100):.3f} %"
#     # )

#     p_press = p_lemke / area_node[vertex_on_Gc]

#     p_num = p_press

#     P_theo = 4 / 3 * E_star * np.sqrt(R) * delta**1.5
#     a_theo = np.sqrt(R * delta)
#     p_cut = 0.0001 * p_num.max()

#     r = np.linalg.norm(x_c[:, :2] - contact_center, axis=1)
#     contact_nodes = p_num > 0  # p_num > p_cut
#     P_num = np.sum(p_num * area_node[vertex_on_Gc])
#     a_num = r[contact_nodes].max()

#     error_a = np.linalg.norm(a_num - a_theo) / np.linalg.norm(a_theo)

#     a_theos.append(a_theo)
#     a_nums.append(a_num)

#     P_theos.append(P_theo)
#     P_nums.append(P_num)

#     print(f"a_num  = {a_num}")
#     print(f"a_theo = {a_theo}")

#     print(f"difference on contact patch radius = {(error_a* 100):.3f} %")


# a_theos = np.array(a_theos)
# a_nums = np.array(a_nums)

# fig2, ax2 = plt.subplots(figsize=(6, 4))

# ax2.plot(deltas, a_theos, "r-", label="a_theo")
# ax2.plot(deltas, a_nums, "g-", label="a_num")
# ax2.set_xlabel(r"approach $\delta$")
# ax2.set_ylabel(r"contact patch radius $a$")
# ax2.set_title("a_theo vs a_num")
# ax2.legend()
# ax2.grid(True)
# plt.tight_layout()


# fig2.savefig("a_theo_vs_a_num.png", format="png")
# plt.show()


# P_theos = np.array(P_theos)
# P_nums = np.array(P_nums)

# fig3, ax3 = plt.subplots(figsize=(6, 4))

# ax3.plot(deltas, P_theos, "r-", label="P_theo")
# ax3.plot(deltas, P_nums, "g-", label="P_num")
# ax3.set_xlabel(r"approach $\delta$")
# ax3.set_ylabel(r"Total load $P$")
# ax3.set_title("P_theo vs P_num")
# ax3.legend()
# ax3.grid(True)
# plt.tight_layout()


# fig3.savefig("P_theo_vs_P_num.png", format="png")
# plt.show()
