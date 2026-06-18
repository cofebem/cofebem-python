import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
from dolfinx.io import VTKFile, gmsh

from mpi4py import MPI
from petsc4py import PETSc

import mpi4py.MPI as MPI

from cofebem.contact.Sc import Sc
from cofebem.contact.rigid_indenters import parabolic
from cofebem.contact.lcp_solvers.ccg import CCG
from cofebem.contact.lcp_solvers.lemke import lemkelcp

# Try on the sphere
# L2 error on and t with a refined mesh
# Comparing G,H_full and nodal collation

# ######################## MESH0 #############################################################################
# -------------------------------------------------------------------------------------------------------
#  Mesh and material parameters
# -------------------------------------------------------------------------------------------------------
# 0 = 0.20 ; 1 = 0.08 ; 2 = 0.02

# W = 40.0
# H = 20.0

# mesh0 = gmsh.read_from_msh(
#     "./cofebem/mesh/smart_Hertz4.msh", MPI.COMM_WORLD, 0, gdim=3
# ).mesh

# mesh1 = gmsh.read_from_msh(
#     "hertz_cube_1.msh", MPI.COMM_WORLD, 0, gdim=3
# ).mesh


# tdim = mesh0.topology.dim
# fdim = tdim - 1

# E = 1.0e9
# nu = 0.3

# lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
# mu = E / (2 * (1 + nu))

# -------------------------------------------------------------------------------------------------------
#  LE weak form
# -------------------------------------------------------------------------------------------------------

# element_type = "Lagrange"
# element_degree = 1

# V0 = functionspace(mesh0, (element_type, element_degree, (mesh0.geometry.dim,)))
# # V1 = functionspace(mesh1, (element_type, element_degree, (mesh1.geometry.dim,)))

# u, v = TrialFunction(V0), TestFunction(V0)


# def epsilon(v):
#     return sym(grad(v))


# def sigma(u):
#     return 2.0 * mu * epsilon(u) + lmbda * tr(epsilon(u)) * Identity(len(u))


# def a(u, v):
#     return inner(sigma(u), epsilon(v)) * dx


# f_v = Constant(mesh0, np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType))


# def L(v):
#     return inner(f_v, v) * dx


# # -------------------------------------------------------------------------------------------------------
# #  Set boundary conditions
# # -------------------------------------------------------------------------------------------------------

# tol = 1.0e-5


# def Gamma_u_selector(x):
#     return np.isclose(x[2], 0, atol=tol)


# Gamma_u = locate_entities_boundary(mesh0, dim=fdim, marker=Gamma_u_selector)
# Iu = locate_dofs_topological(V0, entity_dim=fdim, entities=Gamma_u)

# u0 = np.array([0, 0, 0], dtype=PETSc.ScalarType)


# bc = dirichletbc(
#     u0,
#     dofs=Iu,
#     V=V0,
# )


# -------------------------------------------------------------------------------------------------------
#  Setup the Problem
# -------------------------------------------------------------------------------------------------------

# problem = LinearProblem(
#     a=a(u, v), L=L(v), bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
# )

# problem.solve()


# -------------------------------------------------------------------------------------------------------
#  Gamma_c : Contact Region
# -------------------------------------------------------------------------------------------------------

# tol = 1.0e-5


# # def Gamma_c_selector(x):
# #     return (np.isclose(x[2], H, atol=tol)) & ((x[0] - W/2) ** 2 + (x[1] - W/2) ** 2 <= 8.0)


# def Gamma_c_selector(x):
#     return np.isclose(x[2], H, atol=tol)


# Gamma_c0 = locate_entities_boundary(mesh0, dim=fdim, marker=Gamma_c_selector)
# Ic0 = locate_dofs_topological(V0, fdim, Gamma_c0)
# Gamma_c_x0 = mesh0.geometry.x[Ic0].reshape(-1, tdim)

# Gamma_c1 = locate_entities_boundary(mesh1, dim=fdim, marker=Gamma_c_selector)
# Ic1 = locate_dofs_topological(V1, fdim, Gamma_c1)
# Gamma_c_x1 = mesh1.geometry.x[Ic1].reshape(-1, tdim)
# # -------------------------------------------------------------------------------------------------------
# #  Sc construction: Contact compliance matrix
# # -------------------------------------------------------------------------------------------------------

# Sc_ = Sc(problem.A, problem.b, tdim, Ic0)
# Sc_dense = Sc_.by_sampling()

# Sc_.save(file="./results/hertz/Sc_smart4.npy")

# Sc_dense0 = np.load("./results/hertz/Sc_0.npy")
# Sc_dense1 = np.load("./results/hertz/Sc_1.npy")

# Sc_dense0 = np.load("./results/hertz/Sc_smart4.npy"")
# # -------------------------------------------------------------------------------------------------------
# # Compare with Hertz solution
# # -------------------------------------------------------------------------------------------------------

mesh0= gmsh.read_from_msh(
    "./cofebem/mesh/smart_Hertz0.msh", MPI.COMM_WORLD, 0, gdim=3
).mesh

mesh1 = gmsh.read_from_msh(
    "./cofebem/mesh/smart_Hertz1.msh", MPI.COMM_WORLD, 0, gdim=3
).mesh

mesh2 = gmsh.read_from_msh(
    "./cofebem/mesh/smart_Hertz2.msh", MPI.COMM_WORLD, 0, gdim=3
).mesh

mesh3 = gmsh.read_from_msh(
    "./cofebem/mesh/smart_Hertz3.msh", MPI.COMM_WORLD, 0, gdim=3
).mesh

mesh4 = gmsh.read_from_msh(
    "./cofebem/mesh/smart_Hertz4.msh", MPI.COMM_WORLD, 0, gdim=3
).mesh

meshes = [mesh0, mesh1, mesh2, mesh3, mesh4]

Sc_dense0 = np.load("Sc_smart0.npy")

Sc_dense1 = np.load("Sc_smart1.npy")

Sc_dense2 = np.load("Sc_smart2.npy")

Sc_dense3 = np.load("Sc_smart3.npy")

Sc_dense4 = np.load("Sc_smart4.npy")

Scs = [Sc_dense0, Sc_dense1, Sc_dense2, Sc_dense3, Sc_dense4]


def hertz_vs_cofebem(meshes, Scs):

    W = 40.0
    H = 20.0

    E = 1.0e9
    nu = 0.3

    tol1 = 1.0e-5

    tdim = 3
    fdim = 2

    element_type = "Lagrange"
    element_degree = 1

    def Gamma_c_selector(x):
        return np.isclose(x[2], H, atol=tol1)

    delta = 0.08
    R = 100.0

    E_star = E / (1.0 - nu**2)
    a_theo = np.sqrt(R * delta)
    P_theo = 4 / 3 * E_star * np.sqrt(R) * delta**1.5
    p0_theo = 3 * P_theo / (2 * np.pi * a_theo**2)

    r_th = np.linspace(0, a_theo + 0.3, 400)

    def p_hertz(r, p0, a):
        return p0 * np.sqrt(np.clip(1.0 - (r / a) ** 2, 0.0, None))

    p_th = p_hertz(r_th, p0_theo, a_theo)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r_th, p_th, "-", label="Hertz")

    contact_center = np.array([W / 2, W / 2])

    max_iter = 10000

    errors = []
    h_max = np.array([12.0, 9.6, 7.68, 6.144, 4.9152])

    for i, mesh in enumerate(meshes):

        V = functionspace(mesh, (element_type, element_degree, (tdim,)))

        Gamma_c = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_c_selector)
        Ic = locate_dofs_topological(V, fdim, Gamma_c)

        Gamma_c_x = mesh.geometry.x[Ic].reshape(-1, tdim)

        Sc_dense = Scs[i]

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

        p0_ = np.maximum(-g, 0) * 1e16

        p_lemke, _, err_hist = CCG(
            Sc_dense, g, max_iter, 1e-5, p0=p0_
        ).solve()
        # p_lemke, _, _ = lemkelcp(Sc_dense, g, max_iter)
        # for i, (x, y, z) in enumerate(err_hist):
        #     print(f" iter = {i} : {x}, {y}, {z}")
        # # print(err_hist[:3])
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

        p_press = p_lemke / area_node[Ic]
        p_num = p_press

        rc = np.sqrt((Gamma_c_x[:, 0] - 20) ** 2 + (Gamma_c_x[:, 1] - 20) ** 2)

        p_hertz_ = p_hertz(rc, p0_theo, a_theo)

        # eps = 1e-8
        # err_r = np.zeros_like(p_hertz_)
        # mask = np.abs(p_hertz_) > eps
        # err_r[mask] = np.abs(p_num[mask] - p_hertz_[mask]) / np.abs(p_hertz_[mask])
        p_hertz_avg = (np.pi / 4) * p0_theo
        err_r = (np.abs(p_num - p_hertz_) ** 2) / (p_hertz_avg**2)

        p_fenics_num = Function(V)
        p_fenics_num.name = "p_num"

        p_fenics_hertz = Function(V)
        p_fenics_hertz.name = "p_hertz"

        err_fenics = Function(V)
        err_fenics.name = "error_p"

        with VTKFile(mesh.comm, f"hertz_smart{i}.pvd", "w") as vtk:
            vtk.write_mesh(mesh)
            vtk.write_function([p_fenics_num, p_fenics_hertz, err_fenics], 0)

        p_fenics_num.x.array[:] = 0
        p_fenics_num.x.array[3 * Ic + 2] = p_num
        p_fenics_num.x.scatter_forward()

        p_fenics_hertz.x.array[:] = 0
        p_fenics_hertz.x.array[3 * Ic + 2] = p_hertz_
        p_fenics_hertz.x.scatter_forward()

        err_fenics.x.array[:] = 0
        err_fenics.x.array[3 * Ic + 2] = err_r
        err_fenics.x.scatter_forward()

        vtk.write_function([p_fenics_num, p_fenics_hertz, err_fenics], 1)

        r = np.linalg.norm(Gamma_c_x[:, :2] - contact_center, axis=1)
        contact_nodes = p_num > 0
        a_num = r[contact_nodes].max()

        r_contact = r[contact_nodes]
        p_contact = p_press[contact_nodes]

        nbins = 300
        bins = np.linspace(0, a_theo, nbins + 1)
        r_mid = 0.5 * (bins[:-1] + bins[1:])
        p_avg = np.zeros(nbins)

        for i_ in range(nbins):
            m = (r_contact >= bins[i_]) & (r_contact < bins[i_ + 1])
            p_avg[i_] = p_contact[m].mean() if np.any(m) else np.nan

        def p_fit(r, p0):
            return p0 * np.heaviside(a_theo - r, 1) * np.sqrt(1 - (r / a_num) ** 2)

        mask = ~np.isnan(p_avg) & np.isfinite(p_avg)
        p_avg_ = p_avg[mask]
        r_mid_ = r_mid[mask]

        popt, pcov = curve_fit(p_fit, r_mid_, p_avg_)

        p_fitted = p_hertz(r_mid, popt[0], a_num)

        p_ref = p_hertz(r_mid, p0_theo, a_theo)
        err = np.linalg.norm(p_fitted - p_ref) / np.linalg.norm(p_ref)
        errors.append(err)

        r_ = np.linspace(0, a_num, 400)
        p_plot = p_hertz(r_, popt[0], a_num)
        ax.plot(r_, p_plot, "-", alpha=1.0, label=f"cofebem mesh_{i}")

        print(f"Done with plot {i}")

    ax.set_xlabel(r"radial distance $r$")
    ax.set_ylabel(r"normal pressure $p(r)$")
    ax.set_title("Pressure distribution on $\Gamma_c$")
    ax.legend()
    ax.grid(True)

    # fig.savefig("./results/hertz/hertz_vs_cofebem_smart.png", format="png")

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.loglog(h_max, errors, "o-", lw=2)
    slope = (np.log(errors[-1]) - np.log(errors[0])) / (
        np.log(h_max[-1]) - np.log(h_max[0])
    )
    xref = np.array([h_max.min(), h_max.max()])
    yref = errors[-1] * (xref / h_max[-1]) ** slope
    ax1.loglog(xref, yref, "k--", lw=1, label=f"slope ≈ {slope:.2f}")

    ax1.set_xlabel("element size $h_{\max}$")
    ax1.set_ylabel("relative L2 error ‖$p_{num}$ – $p_{Hertz}$‖ / ‖$p_{Hertz}$‖")
    ax1.set_title("Convergence of pressure distribution")
    ax1.grid(True, which="both", ls="--", alpha=0.6)
    ax1.legend()

    # fig1.savefig("./results/hertz/hertz_vs_cofebem_error.png", format="png")

    plt.tight_layout()
    plt.show()


hertz_vs_cofebem(meshes, Scs)

# ===========================================================================================
#
# # hx = l / nx
# # hy = l / ny
# # area_facet = hx * hy
# # area_node = area_facet / 4.0

# # area_contact_nodes = np.full(len(Gamma_c_x), area_node)


# delta = 0.02
# R = 100.0

# max_iter = 10000
# tol = 1e-8
# err_type = "nw"

# pfactor = 1e8


# u_fenics = Function(V0)
# u_fenics.name = "u"

# p_fenics = Function(V0)
# p_fenics.name = "p"


# # with VTKFile(mesh0.comm, f"./results/hertz/hertz_smart_mesh0.pvd", "w") as vtk:
# #     vtk.write_mesh(mesh0, 0.0)
# #     vtk.write_function([u_fenics, p_fenics], 0.0)


# contact_center = np.array([W / 2, W / 2])

# g0 = (
#     parabolic(
#         Gamma_c_x0[:, 0],
#         Gamma_c_x0[:, 1],
#         contact_center[0],
#         contact_center[1],
#         R,
#         np.full_like(Gamma_c_x0[:, 2], H - delta),
#     )
#     - Gamma_c_x0[:, 2]
# )

# penetrating_nodes0 = np.where(g0 < 0)[0]

# p_lemke0, _, _ = lemkelcp(Sc_dense0, g0, max_iter)


# g1 = (
#     parabolic(
#         Gamma_c_x1[:, 0],
#         Gamma_c_x1[:, 1],
#         contact_center[0],
#         contact_center[1],
#         R,
#         np.full_like(Gamma_c_x1[:, 2], H - delta),
#     )
#     - Gamma_c_x1[:, 2]
# )
# penetrating_nodes1 = np.where(g1 < 0)[0]

# p_lemke1, _, _ = lemkelcp(Sc_dense1, g1, max_iter)

# # p_ccg, _, _ = CCG(Sc_dense, err_type, g, max_iter, tol, pfactor).solve()

# # print(
# #     f" error = {((np.linalg.norm(p_ccg-p_lemke)/np.linalg.norm(p_lemke)) * 100):.3f} %"
# # )

# # # Visualization
# solver_petsc = PETSc.KSP().create(mesh0.comm)
# solver_petsc.setOperators(problem.A)
# solver_petsc.setType("preonly")
# solver_petsc.getPC().setType("lu")
# solver_petsc.setFromOptions()
# solver_petsc.setUp()

# b_ = problem.b.copy()
# u_ = PETSc.Vec().createMPI(b_.getSize(), comm=mesh0.comm)

# b_.set(0)
# for i, dof in enumerate(Ic0):
#     b_.setValue(dof * tdim + 2, p_lemke0[i])
#     b_.assemble()

# solver_petsc.solve(b_, u_)

# u_fenics.x.array[:] = -u_.array
# u_fenics.x.scatter_forward()

# p_fenics.x.array[:] = b_.array
# p_fenics.x.scatter_forward()

# vtk.write_function([u_fenics, p_fenics], 1.0)


# facet2verts0 = mesh0.topology.connectivity(fdim, 0)

# facet_area0 = np.zeros(len(Gamma_c0), dtype=np.float64)
# area_node0 = np.zeros(mesh0.geometry.x.shape[0], dtype=np.float64)

# for local_i, facet in enumerate(Gamma_c0):
#     verts = facet2verts0.links(facet)
#     x0, x1, x2 = mesh0.geometry.x[verts]

#     area_f = 0.5 * np.linalg.norm(np.cross(x1 - x0, x2 - x0))
#     facet_area0[local_i] = area_f

#     share = area_f / 3.0
#     for v in verts:
#         area_node0[v] += share


# x_c0 = Gamma_c_x0

# p_press0 = p_lemke0 / area_node0[Ic0]


# p_num0 = p_press0

# E_star = E / (1.0 - nu**2)
# a_theo = np.sqrt(R * delta)
# P_theo = 4 / 3 * E_star * np.sqrt(R) * delta**1.5
# p0_theo = 3 * P_theo / (2 * np.pi * a_theo**2)

# p_cut0 = 0.0001 * p_num0.max()


# r0 = np.linalg.norm(x_c0[:, :2] - contact_center, axis=1)
# contact_nodes = p_num0 > 0  # p_num > p_cut
# a_num0 = r0[contact_nodes].max()


# P_num0 = np.sum(p_num0 * area_node0[Ic0])

# error_a0 = np.linalg.norm(a_num0 - a_theo) / np.linalg.norm(a_theo)
# error_P0 = np.linalg.norm(P_num0 - P_theo) / np.linalg.norm(P_theo)

# # print(f"a_num  = {a_num0}")
# # print(f"a_theo = {a_theo}")

# # print(f"P_num  = {P_num0}")
# # print(f"P_theo = {P_theo}")

# # print(f"difference on contact patch radius = {(error_a0* 100):.3f} %")
# # print(f"difference on load = {(error_P0* 100):.3f} %")


# r_contact0 = r0[contact_nodes]
# p_contact0 = p_press0[contact_nodes]

# nbins = 300
# bins = np.linspace(0, a_theo, nbins + 1)
# r_mid0 = 0.5 * (bins[:-1] + bins[1:])
# p_avg0 = np.zeros(nbins)

# for i in range(nbins):
#     m = (r_contact0 >= bins[i]) & (r_contact0 < bins[i + 1])
#     p_avg0[i] = p_contact0[m].mean() if np.any(m) else np.nan


# ############### mesh1 #######################################
# facet2verts1 = mesh1.topology.connectivity(fdim, 0)

# facet_area1 = np.zeros(len(Gamma_c1), dtype=np.float64)
# area_node1 = np.zeros(mesh1.geometry.x.shape[0], dtype=np.float64)

# for local_i, facet in enumerate(Gamma_c1):
#     verts = facet2verts1.links(facet)
#     x0, x1, x2 = mesh1.geometry.x[verts]

#     area_f = 0.5 * np.linalg.norm(np.cross(x1 - x0, x2 - x0))
#     facet_area1[local_i] = area_f

#     share = area_f / 3.0
#     for v in verts:
#         area_node1[v] += share


# x_c1 = Gamma_c_x1

# p_press1 = p_lemke1 / area_node1[Ic1]


# p_num1 = p_press1

# p_cut1 = 0.0001 * p_num1.max()


# r1 = np.linalg.norm(x_c1[:, :2] - contact_center, axis=1)
# contact_nodes1 = p_num1 > 0  # p_num > p_cut
# a_num1 = r1[contact_nodes1].max()


# P_num1 = np.sum(p_num1 * area_node1[Ic1])

# # error_a0 = np.linalg.norm(a_num0 - a_theo) / np.linalg.norm(a_theo)
# # error_P0 = np.linalg.norm(P_num0 - P_theo) / np.linalg.norm(P_theo)

# # print(f"a_num  = {a_num0}")
# # print(f"a_theo = {a_theo}")

# # print(f"P_num  = {P_num0}")
# # print(f"P_theo = {P_theo}")

# # print(f"difference on contact patch radius = {(error_a0* 100):.3f} %")
# # print(f"difference on load = {(error_P0* 100):.3f} %")


# r_contact1 = r1[contact_nodes1]
# p_contact1 = p_press1[contact_nodes1]

# nbins = 300
# bins = np.linspace(0, a_theo, nbins + 1)
# r_mid1 = 0.5 * (bins[:-1] + bins[1:])
# p_avg1 = np.zeros(nbins)

# for i in range(nbins):
#     m = (r_contact1 >= bins[i]) & (r_contact1 < bins[i + 1])
#     p_avg1[i] = p_contact1[m].mean() if np.any(m) else np.nan


# r_th = np.linspace(0, a_theo, 400)
# p_th = p0_theo * np.sqrt(1 - (r_th / a_theo) ** 2)


# def press_(r, p0):
#     return p0 * np.heaviside(a_theo - r, 1) * np.sqrt(1 - (r / a_num0) ** 2)


# # press = np.vectorize(press_)

# mask0 = ~np.isnan(p_avg0) & np.isfinite(p_avg0)
# p_avg0_ = p_avg0[mask0]
# r_mid0_ = r_mid0[mask0]

# # mask1 = ~np.isnan(p_avg1) & np.isfinite(p_avg1)
# # p_avg1_ = p_avg1[mask1]
# # r_mid1_ = r_mid0[mask1]

# popt0, pcov0 = curve_fit(press_, r_mid0_, p_avg0_)
# # popt1, pcov1 = curve_fit(press_, r_mid1_, p_avg1_)


# def p_avg_fit0(r):
#     return popt0[0] * np.sqrt(1 - (r / a_num0) ** 2)


# def p_avg_fit1(r):
#     return popt1[0] * np.sqrt(1 - (r / popt1[1]) ** 2)


# fig1, ax1 = plt.subplots(figsize=(6, 4))

# ax1.plot(r_mid0, p_avg0, "go", label="FE mesh0 (binned)")
# ax1.plot(r_mid0, p_avg1, "ro", alpha=0.5, label="FE mesh1 (binned)")
# ax1.plot(r_mid0, p_avg_fit0(r_mid0), "r-", alpha=0.5, label="FE mesh1 (fit)")
# ax1.plot(r_th, p_th, "-", label="Hertz")
# ax1.set_xlabel(r"radial distance $r$")
# ax1.set_ylabel(r"normal pressure $p(r)$")
# ax1.set_title("Pressure distribution on $\Gamma_c$")
# ax1.legend()
# ax1.grid(True)
# plt.tight_layout()

# fig1.savefig("./results/hertz/hertz_vs_cofebem_smartmesh0.png", format="png")
# plt.show()


# ------------------------------------------------------------------------------------------------------
# a_num vs a_theo as a function of the approach delta
# ------------------------------------------------------------------------------------------------------

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

#     p_press = p_lemke / area_node[Ic]

#     p_num = p_press

#     P_theo = 4 / 3 * E_star * np.sqrt(R) * delta**1.5
#     a_theo = np.sqrt(R * delta)
#     p_cut = 0.0001 * p_num.max()

#     r = np.linalg.norm(x_c[:, :2] - contact_center, axis=1)
#     contact_nodes = p_num > 0  # p_num > p_cut
#     P_num = np.sum(p_num * area_node[Ic])
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


# fig2.savefig("./results/hertz/a_theo_vs_a_num_0.png", format="png")
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


# fig3.savefig("./results/hertz/P_theo_vs_P_num_0.png", format="png")
# plt.show()
