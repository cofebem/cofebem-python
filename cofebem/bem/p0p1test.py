import numpy as np

from dolfinx.mesh import (
    CellType,
    GhostMode,
    create_box,
    locate_entities_boundary,
    meshtags,
)
from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
    form,
    dirichletbc,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import LinearProblem, assemble_matrix
from ufl import (
    Identity,
    Measure,
    FunctionSpace,
    TrialFunction,
    TestFunction,
    sym,
    grad,
    inner,
    tr,
    dx,
)
from dolfinx.io import VTKFile, XDMFFile, gmshio

from mpi4py import MPI
from petsc4py import PETSc
from tqdm import tqdm

import mpi4py.MPI as MPI

from cofebem.contact.Sc import Sc
from cofebem.contact.rigid_indenters import parabolic, flat
from cofebem.contact.lcp_solvers.ccg import CCG
from cofebem.contact.lcp_solvers.lemke import lemkelcp


# -------------------------------------------------------------------------------------------------------
#  Mesh and material parameters
# -------------------------------------------------------------------------------------------------------

R = 1.0  # radius of hemisphere


E = 1.0e9
nu = 0.3

E_star = E / (1 - nu**2)

R_indenter = 10.0
delta = 0.05  # indentation depth

lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))

errs_p1p1 = []
errs_p1p0 = []
errs_p0p0 = []


def p_analytic(X):
    x = X[:, 0]
    y = X[:, 1]

    a = np.sqrt(R * delta)  # contact radius
    p0 = 2 * E_star * a / (np.pi * R)  # max contact pressure

    r2 = x**2 + y**2
    p = np.zeros(X.shape[0], dtype=float)

    inside = r2 <= a**2
    p[inside] = p0 * np.sqrt(1.0 - r2[inside] / a**2)

    return p


def err_relative(p, p_ref):
    a = np.sqrt(R * delta)  # contact radius
    p0 = 2 * E_star * a / (np.pi * R)
    return np.linalg.norm(p - p_ref) / p0


for i in range(6):

    # -------------------------------------------------------------------------------------------------------
    #  Mesh
    # -------------------------------------------------------------------------------------------------------

    mesh, cell_tags, facet_tags = gmshio.read_from_msh(
        f"msh_files/hemisphere{i}.msh", MPI.COMM_WORLD, 0, gdim=3
    )

    tdim = mesh.topology.dim
    fdim = tdim - 1

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
        a=a(u, v),
        L=L(v),
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    problem.solve()

    # -------------------------------------------------------------------------------------------------------
    #  Gamma_c : Contact Region
    # -------------------------------------------------------------------------------------------------------

    tol = 1.0e-5

    def Gamma_c_selector(x):
        return np.isclose(np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2), R, atol=tol)

    Gamma_c = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_c_selector)

    mesh.topology.create_connectivity(fdim, 0)

    Ic = locate_dofs_topological(V, fdim, Gamma_c)
    x_Ic = mesh.geometry.x[Ic].reshape(-1, tdim)

    # -------------------------------------------------------------------------------------------------------
    #  Sc construction: Contact compliance matrix
    # -------------------------------------------------------------------------------------------------------

    Sc_ = Sc(problem.A, problem.b, tdim, Ic)
    Sc_p1p1 = Sc_.by_sampling()

    # -------------------------------------------------------------------------------------------------------
    #  Contact Problem P1P1
    # -------------------------------------------------------------------------------------------------------
    mesh.topology.create_connectivity(fdim, 0)
    facet_to_vertices = mesh.topology.connectivity(fdim, 0)

    Ne = len(Gamma_c)
    Nc = len(Ic)

    vertex_to_local = {v: i for i, v in enumerate(Ic)}

    Pi_u_to_p = np.zeros((Ne, Nc), dtype=np.float64)
    A_elem = np.zeros(Ne, dtype=np.float64)  # one area per contact facet
    A_node = np.zeros(
        mesh.geometry.x.shape[0], dtype=np.float64
    )  # lumped area per contact node

    for e_idx, f in enumerate(Gamma_c):
        vs = facet_to_vertices.links(f)

        coords = mesh.geometry.x[vs]  # shape (nv, 3)
        v0, v1, v2 = coords

        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        A_elem[e_idx] = area

        for v in vs:
            A_node[v] += area / 3

    for e_idx, f in enumerate(Gamma_c):
        vs = facet_to_vertices.links(f)

        local_ids = [vertex_to_local[v] for v in vs if v in vertex_to_local]
        nv = len(local_ids)

        w = 1.0 / nv
        for lid in local_ids:
            Pi_u_to_p[e_idx, lid] = w

    max_iter = 10000
    tolerance = 1e-5
    error_type = "nw"

    pfactor = 1e8

    # u_fenics = Function(V)
    # u_fenics.name = "u"

    # p_fenics = Function(V)
    # p_fenics.name = "p"

    # with VTKFile(mesh.comm, f"./results/hemisphereP1P1{i}.pvd", "w") as vtk:
    #     vtk.write_mesh(mesh)
    #     vtk.write_function([u_fenics, p_fenics], 0)

    contact_center = np.array([0.0, 0.0])
    gap_p1p1 = (
        parabolic(
            x_Ic[:, 0],
            x_Ic[:, 1],
            contact_center[0],
            contact_center[1],
            R_indenter,
            np.ones_like(x_Ic[:, 2]) - delta,
        )
        - x_Ic[:, 2]
    )
    penetrating_nodes = np.where(gap_p1p1 < 0)[0]

    f_p1, _, _ = CCG(
        Sc_p1p1, error_type, gap_p1p1, max_iter, tolerance, pfactor
    ).solve()

    # f_p1, _, _ = lemkelcp(Sc_p1p1, gap_p1p1, max_iter)

    p_p1 = np.zeros_like(f_p1)
    p_p1 = f_p1 / A_node[Ic]

    p_analytic_p1 = p_analytic(x_Ic)

    err_p1p1 = err_relative(p_p1, p_analytic_p1)
    errs_p1p1.append(err_p1p1)

    print(f"Refinement step {i}: relative error P1P1 = {err_p1p1:.6e}")

    r_nodes = np.sqrt(x_Ic[:, 0] ** 2 + x_Ic[:, 1] ** 2)

    if i == 5:
        r_p1 = r_nodes.copy()
        p_p1_last = p_p1.copy()
        p_analytic_p1_last = p_analytic_p1.copy()

    # solver_petsc = PETSc.KSP().create(mesh.comm)
    # solver_petsc.setOperators(problem.A)
    # solver_petsc.setType("preonly")
    # solver_petsc.getPC().setType("lu")
    # solver_petsc.setFromOptions()
    # solver_petsc.setUp()

    # b_ = problem.b.copy()
    # u_ = PETSc.Vec().createMPI(b_.getSize(), comm=mesh.comm)

    # b_.set(0)
    # for i, dof in enumerate(Ic):
    #     b_.setValue(dof * tdim + 2, p_lemke[i])
    #     b_.assemble()

    # solver_petsc.solve(b_, u_)

    # u_fenics.x.array[:] = -u_.array
    # u_fenics.x.scatter_forward()

    # p_fenics.x.array[:] = b_.array
    # p_fenics.x.scatter_forward()

    # vtk.write_function([u_fenics, p_fenics], 1)

    # -----------------------------------------------------------------------------------------------------------

    Sc_p0p0 = Pi_u_to_p @ Sc_p1p1 @ Pi_u_to_p.T  # (Ne x Ne)

    centers = np.zeros((Ne, 3), dtype=np.float64)
    for e_idx, f in enumerate(Gamma_c):
        vs = facet_to_vertices.links(f)
        coords = mesh.geometry.x[vs]
        centers[e_idx, :] = coords.mean(axis=0)

    USE_3POINT_GAP = True
    GAP_AGGREGATION = "average"

    if not USE_3POINT_GAP:
        z0 = np.full(Ne, 1.0) - delta
        z_ind = parabolic(
            centers[:, 0],
            centers[:, 1],
            contact_center[0],
            contact_center[1],
            R_indenter,
            z0,
        )
        gap_elem = z_ind - centers[:, 2]

    else:
        gap_elem = np.zeros(Ne, dtype=np.float64)
        z0_local = 1.0 - delta

        for e_idx, f in enumerate(Gamma_c):
            vs = facet_to_vertices.links(f)  # vertices of facet f
            coords = mesh.geometry.x[vs]  # shape (3, 3) for triangles

            x_coords = coords[:, 0]
            y_coords = coords[:, 1]
            z_coords = coords[:, 2]

            z_ind_pts = parabolic(
                x_coords,
                y_coords,
                contact_center[0],
                contact_center[1],
                R_indenter,
                np.full(len(x_coords), z0_local),
            )
            gaps_pts = z_ind_pts - z_coords  # gap at each of the 3 points

            if GAP_AGGREGATION == "min":
                gap_elem[e_idx] = gaps_pts.min()
            else:  # "average" (default)
                gap_elem[e_idx] = gaps_pts.mean()

    # f_p0, _, _ = CCG(
    #     Sc_p0p0, error_type, gap_elem, max_iter, tolerance, pfactor
    # ).solve()

    f_p0, _, _ = lemkelcp(Sc_p0p0, gap_elem, max_iter)

    p_p0 = np.zeros_like(f_p0)
    mask_elem = A_elem > 1.0e-14
    p_p0[mask_elem] = f_p0[mask_elem] / A_elem[mask_elem]

    p_analytic_p0 = p_analytic(centers)

    err_p0p0 = err_relative(p_p0, p_analytic_p0)
    errs_p0p0.append(err_p0p0)
    print(f"Refinement step {i}: relative error P0P0 = {err_p0p0:.6e}")

    p_p1p0 = Pi_u_to_p.T @ p_p0
    err_p1p0 = err_relative(p_p1p0, p_analytic_p1)
    errs_p1p0.append(err_p1p0)
    print(f"Refinement step {i}: relative error P1P0 = {err_p1p0:.6e}")

    # solver_petsc = PETSc.KSP().create(mesh.comm)
    # solver_petsc.setOperators(problem.A)
    # solver_petsc.setType("preonly")
    # solver_petsc.getPC().setType("lu")
    # solver_petsc.setFromOptions()
    # solver_petsc.setUp()

    # b = problem.b.copy()
    # b.set(0)

    # for i_node, dof in enumerate(Ic):
    #     b.setValue(dof * tdim + 2, p_nodes[i_node])
    # b.assemble()

    # u = problem.A.createVecRight()
    # solver_petsc.solve(b, u)

    # u_fenics = Function(V)
    # u_fenics.name = "u"

    # p_fenics = Function(V)
    # p_fenics.name = "p"

    # with VTKFile(mesh.comm, f"./results/hemisphereP1P0.pvd", "w") as vtk2:
    #     vtk2.write_mesh(mesh)
    #     vtk2.write_function([u_fenics, p_fenics], 0)

    # u_fenics.x.array[:] = -u.array
    # u_fenics.x.scatter_forward()
    # u_fenics.name = "u"

    # p_fenics.x.array[:] = b_.array
    # p_fenics.x.scatter_forward()

    # vtk2.write_function([u_fenics, p_fenics], 1)

print("All refinement steps done.")

#######################################################################
hs = np.array([1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8])  # , 1 / 15])
errs_p1p1 = np.array(errs_p1p1)
errs_p1p0 = np.array(errs_p1p0)
errs_p0p0 = np.array(errs_p0p0)


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(hs, errs_p1p1, "bo--", label="err P1P1", markersize=6, linewidth=2)
ax.plot(hs, errs_p1p0, "ys--", label="err P1P0", markersize=6, linewidth=2)
ax.plot(hs, errs_p0p0, "gx--", label="err P0P0", markersize=6, linewidth=2)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("h (element size)", fontsize=10)
ax.set_ylabel("relative error", fontsize=10)

ax.set_title("Comparison of traction in P1/P0", fontsize=16)

ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.legend(fontsize=8, loc="upper left")

plt.tight_layout()

plt.show()

#########################################################
idx_sorted = np.argsort(r_p1)
r_sorted = r_p1[idx_sorted]
p_num_sorted = p_p1_last[idx_sorted]
p_analytic_sorted = p_analytic_p1_last[idx_sorted]

fig2, ax2 = plt.subplots(figsize=(6, 4))

ax2.plot(r_sorted, p_analytic_sorted, "-", label="p_analytic (P1 nodes)", linewidth=2)
ax2.plot(r_sorted, p_num_sorted, "o", label="p_P1 (Lemke)", markersize=4)

ax2.set_xlabel("r (radius in contact plane)", fontsize=10)
ax2.set_ylabel("pressure p", fontsize=10)
ax2.set_title("Pressure profile p(r) at P1 contact nodes", fontsize=14)

ax2.grid(True, linestyle="--", linewidth=0.5)
ax2.legend(fontsize=8)

plt.tight_layout()
plt.show()
