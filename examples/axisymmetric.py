import numpy as np
import time
import matplotlib.pyplot as plt

from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
    locate_dofs_geometrical,
)
from dolfinx.fem.petsc import LinearProblem
from ufl import Identity, TrialFunction, TestFunction, sym, grad, inner, tr, dx
from dolfinx.io import XDMFFile

from mpi4py import MPI
from petsc4py import PETSc
from tqdm import tqdm

from dolfinx.io import XDMFFile, VTKFile
import mpi4py.MPI as MPI

from cofebem.mesh.hollow_cylinder import hollow_cylinder
from cofebem.contact.Sc import Sc
from cofebem.contact.rigid_indenters import parabolic
from cofebem.contact.lcp_solvers.ccg import CCG
from cofebem.hmatrices.cluster import HMatrix


# -------------------------------------------------------------------------------------------------------
#  Mesh and material parameters
# -------------------------------------------------------------------------------------------------------
nr = 10
nt = 70
nz = 3

r_inner = 1
r_outer = 5

hollow_cylinder(nr, nt, nz, r_inner, r_outer)

with XDMFFile(MPI.COMM_WORLD, "hollow_cylinder.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

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
tol = 1.0e-2


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
#  Construct S_c for a symmetric domain
# -------------------------------------------------------------------------------------------------------
angle_tol = 1.0e-8
tol = 1.0e-5


def Gamma_c_selector(x):
    return np.isclose(x[2], 1, atol=tol)


Gamma_c = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_c_selector)
Ic = locate_dofs_topological(V, fdim, Gamma_c)


def reference_line_selector(x):
    return (
        np.isclose(x[1], 0.0, atol=tol) & np.isclose(x[2], 1.0, atol=tol) & (x[0] > 0)
    )


Gamma_c_ref = locate_entities_boundary(mesh, fdim, reference_line_selector)

Ic_ref = locate_dofs_geometrical(V, reference_line_selector)


Gamma_c_x = mesh.geometry.x[Ic].reshape(-1, tdim)

angles = (np.arctan2(Gamma_c_x[:, 1], Gamma_c_x[:, 0]) + 2 * np.pi) % (2 * np.pi)
order_angle = np.argsort(angles)
Ic_sorted = Ic[order_angle]

Ic_sorted = Ic_sorted.reshape(nt, nr + 1)

Gamma_c_x_segments = mesh.geometry.x[Ic_sorted].reshape(nt, nr + 1, tdim)
radii = np.sqrt(Gamma_c_x_segments[:, :, 0] ** 2 + Gamma_c_x_segments[:, :, 1] ** 2)

order_radius = np.argsort(radii, axis=1)
Ic_sorted = np.take_along_axis(Ic_sorted, order_radius, axis=1).flatten()


def compute_Sc_ref(A, b, Ic_ref, Ic_sorted, tdim, comm, f_magnitude=1.0):
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("preonly")
    solver.getPC().setType("lu")

    uh = PETSc.Vec().createMPI(b.getSize(), comm=comm)

    dofs_ref = np.array([node * tdim + 2 for node in Ic_ref])
    dofs_contact = np.array([node * tdim + 2 for node in Ic_sorted])

    Sc_ref = np.zeros((len(dofs_ref), len(dofs_contact)))

    for i, dof in tqdm(enumerate(dofs_ref), desc="Computing Sc_ref"):
        b.zeroEntries()
        b.setValue(dof, f_magnitude)
        b.assemble()

        uh = PETSc.Vec().createMPI(b.getSize(), comm=comm)
        solver.solve(b, uh)

        Sc_ref[i, :] = uh.array[dofs_contact] / f_magnitude

    return Sc_ref


def construct_Sc_from_Sc_ref(Sc_ref, nt, Ic):
    n_ref, nc = Sc_ref.shape
    assert nc == n_ref * nt, "Mismatch between segments and contact nodes."

    Sc = np.zeros((nc, nc))

    for i in tqdm(range(nt), desc="Computing Sc by symmetry"):
        row_start = i * n_ref
        row_end = (i + 1) * n_ref

        shift = i * n_ref

        Sc[row_start:row_end, :] = np.roll(Sc_ref, shift=shift, axis=1)

    mapping = {dof: i for i, dof in enumerate(Ic_sorted)}

    perm = np.array([mapping[dof] for dof in Ic])

    Sc_classic_unsorted = Sc[np.ix_(perm, perm)]

    return Sc_classic_unsorted  # Sc  #


sym_start = time.perf_counter()
Sc_ref = compute_Sc_ref(problem.A, problem.b, Ic_ref, Ic_sorted, tdim, mesh.comm)
Sc_dense = construct_Sc_from_Sc_ref(Sc_ref, nt, Ic)

sym_end = time.perf_counter()
sym_duration = sym_end - sym_start


# Plot heatmap
plt.imshow(Sc_dense, cmap="viridis", aspect="auto")
plt.colorbar()
plt.show()

# Sc_hm = HMatrix.from_dense(Sc_dense, Gamma_c_x, Gamma_c, leaf=32, eta=1.7)
# print("stats:", Sc_hm.stats())

# -------------------------------------------------------------------------------------------------------
#  Construct S_c classic
# -------------------------------------------------------------------------------------------------------


# classic_start = time.perf_counter()

# Sc_classic = Sc(
#     problem.A, problem.b, mesh.comm, tdim, Ic, full_Sc=False
# ).by_sampling()

# classic_end = time.perf_counter()

# classic_duration = classic_end - classic_start

# error = np.linalg.norm(Sc_classic - Sc) / np.linalg.norm(Sc_classic)

# print(f"classic duration = {classic_duration}")
# print(f"sym duration = {sym_duration}")
# print(f"error = {error}")


# -------------------------------------------------------------------------------------------------------
#  Complexity Comparison
# -------------------------------------------------------------------------------------------------------

# nts = np.linspace(5, 400, 30)
# dofs = np.linspace(nr * 5, nr * 400, 30)
# symmetry_times = []
# classic_times = []

# for nt in nts:
#     nt = int(nt)
#     # nr = nr * np.pi
#     # nr = int(nr)
#     # print(nr)
#     hollow_cylinder(nr, nt, nz)

#     with XDMFFile(MPI.COMM_WORLD, "hollow_cylinder.xdmf", "r") as xdmf:
#         mesh = xdmf.read_mesh(name="Grid")

#     tdim = mesh.topology.dim
#     fdim = tdim - 1

#     E = 1.0e9
#     nu = 0.3

#     lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
#     mu = E / (2 * (1 + nu))

#     element_type = "Lagrange"
#     element_degree = 1

#     V = functionspace(mesh, (element_type, element_degree, (mesh.geometry.dim,)))

#     u, v = TrialFunction(V), TestFunction(V)

#     def epsilon(v):
#         return sym(grad(v))

#     def sigma(u):
#         return 2.0 * mu * epsilon(u) + lmbda * tr(epsilon(u)) * Identity(len(u))

#     def a(u, v):
#         return inner(sigma(u), epsilon(v)) * dx

#     f_v = Constant(mesh, np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType))

#     def L(v):
#         return inner(f_v, v) * dx

#     tol = 1.0e-5

#     def Gamma_u_selector(x):
#         return np.isclose(x[2], 0, atol=tol)

#     Gamma_u = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_u_selector)
#     Iu = locate_dofs_topological(V, entity_dim=fdim, entities=Gamma_u)

#     u0 = np.array([0, 0, 0], dtype=PETSc.ScalarType)

#     bc = dirichletbc(
#         u0,
#         dofs=Iu,
#         V=V,
#     )

#     problem = LinearProblem(
#         a=a(u, v),
#         L=L(v),
#         bcs=[bc],
#         petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
#     )

#     problem.solve()

#     angle_tol = 1.0e-8
#     tol = 1.0e-8

#     def Gamma_c_selector(x):
#         return np.isclose(x[2], 1, atol=tol)

#     Gamma_c = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_c_selector)
#     Ic = locate_dofs_topological(V, fdim, Gamma_c)

#     def reference_line_selector(x):
#         return (
#             np.isclose(x[1], 0.0, atol=tol)
#             & np.isclose(x[2], 1.0, atol=tol)
#             & (x[0] > 0)
#         )

#     Gamma_c_ref = locate_entities_boundary(mesh, fdim, reference_line_selector)

#     Ic_ref = locate_dofs_geometrical(V, reference_line_selector)

#     Gamma_c_x = mesh.geometry.x[Ic]
#     Gamma_c_x = Gamma_c_x.reshape(-1, tdim)
#     angles = (np.arctan2(Gamma_c_x[:, 1], Gamma_c_x[:, 0]) + 2 * np.pi) % (2 * np.pi)
#     order_angle = np.argsort(angles)
#     Ic_sorted = Ic[order_angle]

#     Ic_sorted = Ic_sorted.reshape(nt, nr + 1)

#     Gamma_c_x_segments = mesh.geometry.x[Ic_sorted].reshape(nt, nr + 1, tdim)
#     radii = np.sqrt(Gamma_c_x_segments[:, :, 0] ** 2 + Gamma_c_x_segments[:, :, 1] ** 2)

#     order_radius = np.argsort(radii, axis=1)
#     Ic_sorted = np.take_along_axis(Ic_sorted, order_radius, axis=1).flatten()

#     sym_start = time.perf_counter()
#     Sc_ref = compute_Sc_ref(problem.A, problem.b, Ic_ref, Ic_sorted, tdim, mesh.comm)
#     Sc_ = construct_Sc_from_Sc_ref(Sc_ref, nt, Ic)

#     sym_end = time.perf_counter()
#     sym_duration = sym_end - sym_start

#     classic_start = time.perf_counter()

#     Sc_classic = Sc(
#         problem.A, problem.b, mesh.comm, tdim, Ic_sorted, full_Sc=False
#     ).by_sampling()

#     classic_end = time.perf_counter()

#     classic_duration = classic_end - classic_start

#     error = np.linalg.norm(Sc_classic - Sc) / np.linalg.norm(Sc_classic)

#     symmetry_times.append(sym_duration)
#     classic_times.append(classic_duration)

#     print(f"classic duration = {classic_duration}")
#     print(f"sym duration = {sym_duration}")
#     print(f"error = {error}")
#     print(f"-------------------------------------------------------")

# x_lin = np.linspace(100, 1600, 10)
# # Shifted reference power curves to start at the same point as the first data point
# symmetry_times = np.asarray(symmetry_times)
# classic_times = np.asarray(classic_times)
# shift_value = symmetry_times[0]  # classic_times[0]
# power_1 = x_lin / x_lin[0] * shift_value
# power_2 = (x_lin / x_lin[0]) ** 2 * shift_value

# # power_1 = x_lin / x_lin[0] * shift_value
# # power_2 = (x_lin / x_lin[0]) ** 2 * shift_value

# # Create the figure and axis
# fig, ax = plt.subplots(figsize=(4, 3))

# # Plot the data
# ax.plot(dofs, classic_times, "o-", label="Classic Sc", markersize=6, linewidth=2)
# ax.plot(
#     dofs,
#     symmetry_times,
#     "s-",
#     label="Sc enhaced by Symmetry",
#     markersize=6,
#     linewidth=2,
# )

# ax.plot(x_lin, power_1, "--", color="black")  # label="O(N)")
# ax.plot(x_lin, power_2, "-.", color="black")  # label="O(N²)")  # Annotate power curves

# ax.text(
#     dofs[-1],
#     power_1[-1],
#     "O(N)",
#     fontsize=8,
#     color="black",
#     verticalalignment="bottom",
#     horizontalalignment="right",
# )
# ax.text(
#     dofs[-1],
#     power_2[-1],
#     "O(N²)",
#     fontsize=8,
#     color="black",
#     verticalalignment="bottom",
#     horizontalalignment="right",
# )

# # Logarithmic scale for better readability
# ax.set_xscale("log")
# ax.set_yscale("log")

# # Labels and title
# ax.set_xlabel("Degrees of Freedom (DoFs)", fontsize=8)
# ax.set_ylabel("CPU Time (s)", fontsize=8)

# ax.set_title("Comparison of Classic and Symmetry Methods for Sc", fontsize=16)

# # Grid and legend
# ax.grid(True, which="both", linestyle="--", linewidth=0.5)
# ax.legend(fontsize=8, loc="upper left")

# # Improve layout
# plt.tight_layout()

# fig.savefig("Sc_by_symmetry.png", format="png")

# # Show the plot
# plt.show()


# -------------------------------------------------------------------------------------------------------
#  Contact Problem
# -------------------------------------------------------------------------------------------------------

displ = 0.7
Rindenter = 2.0

max_iter = 1000
tolerance = 1e-5
error_type = "nw"
pfactor = 1e8
Nframes = 70

u_fenics = Function(V)
u_fenics.name = "u"

p_fenics = Function(V)
p_fenics.name = "p"


with VTKFile(mesh.comm, f"./results/axisymmetric/axisymmetric.pvd", "w") as vtk:
    vtk.write_mesh(mesh)
    vtk.write_function([u_fenics, p_fenics], 0)

ANIMATION = True
if ANIMATION == True:
    # x_center = np.linspace(-7.0, 7.0, Nframes)
    theta_c = np.linspace(0.0, 2 * np.pi, Nframes)
    rc = 3.0
    for frame, theta_c_ in tqdm(enumerate(theta_c), desc="Solving Contact", unit="it"):
        contact_center = np.array([rc * np.cos(theta_c_), rc * np.sin(theta_c_)])
        gap = (
            parabolic(
                Gamma_c_x[:, 0],
                Gamma_c_x[:, 1],
                contact_center[0],
                contact_center[1],
                Rindenter,
                np.ones_like(Gamma_c_x[:, 2]) - displ,
            )
            - Gamma_c_x[:, 2]
        )
        penetrating_nodes = np.where(gap < 0)[0]

        p_ccg, _, _ = CCG(Sc_hm, error_type, gap, max_iter, tolerance).solve_hm()
        # p_ccg_hm, _, _ = CCG(Sc_hm, error_type, gap, max_iter, tolerance).solve_hm()

        # error = np.linalg.norm(p_ccg - p_ccg_hm) / np.linalg.norm(p_ccg)

        # print(f"relative error on p = {error}")

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
            b_.setValue(dof * tdim + 2, p_ccg[i])
            b_.assemble()

        solver_petsc.solve(b_, u_)

        u_fenics.x.array[:] = -u_.array
        u_fenics.x.scatter_forward()

        p_fenics.x.array[:] = b_.array
        p_fenics.x.scatter_forward()

        vtk.write_function([u_fenics, p_fenics], frame + 1)
