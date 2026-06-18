import numpy as np
import time
import matplotlib.pyplot as plt

from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
    locate_dofs_geometrical,
    form,
)
from dolfinx.fem.petsc import LinearProblem, assemble_matrix
from ufl import Identity, Measure, TrialFunction, TestFunction, sym, grad, inner, tr, dx
from dolfinx.io import XDMFFile

from mpi4py import MPI
from petsc4py import PETSc
from tqdm import tqdm

from dolfinx.io import XDMFFile, VTKFile, gmsh
import mpi4py.MPI as MPI

from cofebem.mesh.hollow_cylinder import hollow_cylinder
from cofebem.contact.Sc import Sc
from cofebem.contact.rigid_indenters import parabolic
from cofebem.contact.lcp_solvers.ccg import CCG
from cofebem.contact.lcp_solvers.lemke import lemkelcp
from cofebem.hmatrices.hmatrix import HMatrix
from cofebem.bodies.sphere_indenter import Sphere

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
Gamma_u_id = 1
Gamma_u_tags = np.full(Gamma_u.shape, Gamma_u_id, dtype=np.int32)
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
    a=a(u, v), L=L(v), bcs=[bc],petsc_options_prefix ="le", petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
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
Gamma_c_id = 2
Gamma_c_tags = np.full(Gamma_c.shape, Gamma_c_id, dtype=np.int32)


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


Sc_hm = HMatrix(Gamma_c_x, Sc_dense, leaf_size=80, eta=1.7)
print("stats:", Sc_hm.stats())

# -------------------------------------------------------------------------------------------------------
#  Construct S_c classic
# -------------------------------------------------------------------------------------------------------

facet_indices = np.hstack([Gamma_u, Gamma_c]).astype(np.int32)
facet_values = np.hstack(
    [
        Gamma_u_tags,
        Gamma_c_tags,
    ]
).astype(np.int32)

order = np.argsort(facet_indices)
facet_indices = facet_indices[order]
facet_values = facet_values[order]

mt = meshtags(mesh, fdim, facet_indices, facet_values)

ds = Measure("ds", domain=mesh, subdomain_data=mt)
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
#         petsc_options_prefix = "le",  
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
def build_Mcc(V, mesh, ds, Gamma_c, Gamma_c_id):
    comm = mesh.comm
    fdim = mesh.topology.dim - 1

    W, W_to_V = V.sub(2).collapse()

    dofs_c_W = locate_dofs_topological(W, fdim, Gamma_c)
    dofs_c_W = np.asarray(dofs_c_W, dtype=np.int32)

    p = TrialFunction(W)
    q = TestFunction(W)
    m_form = form(inner(p, q) * ds(Gamma_c_id))

    M = assemble_matrix(m_form)
    M.assemble()

    is_c = PETSc.IS().createGeneral(dofs_c_W, comm=comm)
    Mcc = M.createSubMatrix(is_c, is_c)

    ksp_Mcc = PETSc.KSP().create(comm)
    ksp_Mcc.setOperators(Mcc)
    ksp_Mcc.setType("preonly")
    ksp_Mcc.getPC().setType("lu")
    ksp_Mcc.setUp()

    n = len(dofs_c_W)
    rhs_arr = np.zeros(n, dtype=PETSc.ScalarType)
    sol_arr = np.zeros(n, dtype=PETSc.ScalarType)
    rhs = PETSc.Vec().createWithArray(rhs_arr, comm=comm)
    sol = PETSc.Vec().createWithArray(sol_arr, comm=comm)

    return ksp_Mcc, rhs, sol, rhs_arr, sol_arr, dofs_c_W, W_to_V


def fc_to_tc(fc, ksp_Mcc, rhs, sol, rhs_arr, sol_arr, sign=+1.0):
    fc = np.asarray(fc, dtype=PETSc.ScalarType)
    if fc.shape[0] != rhs_arr.shape[0]:
        raise ValueError(
            f"fc has length {fc.shape[0]} but Mcc expects {rhs_arr.shape[0]}"
        )

    rhs_arr[:] = sign * fc
    ksp_Mcc.solve(rhs, sol)
    return sol_arr.copy()


displ = 0.5
Rindenter = 1.0

indenter = Sphere(center=np.array([0, 0, 0]), radius=Rindenter)

sphere_mesh = gmsh.read_from_msh(
    "./msh_files/fine_sphere.msh", MPI.COMM_WORLD, 0, gdim=3
).mesh

sphere_ref_x = sphere_mesh.geometry.x[:, :3].copy()
V_sphere = functionspace(sphere_mesh, ("Lagrange", 1))
u_sphere = Function(V_sphere)
u_sphere.name = "indenter"

max_iter = 1000
tolerance = 1e-5
error_type = "nw"
pfactor = 1e8
Nframes = 70

u_fenics = Function(V)
u_fenics.name = "u"


ksp_Mcc, rhs_Mcc, sol_Mcc, rhs_arr, sol_arr, dofs_c_W, W_to_V = build_Mcc(
    V, mesh, ds, Gamma_c, Gamma_c_id
)

W, W_to_V_full = V.sub(2).collapse()
pW = Function(W)
pW.name = "$p_{c}$"


Vz_Ic = (Ic * tdim + 2).astype(np.int64)
Vz_from_W = np.asarray(W_to_V_full, dtype=np.int64)[dofs_c_W]
pos_in_Ic = {int(d): i for i, d in enumerate(Vz_Ic)}
perm_W_from_Ic = np.array([pos_in_Ic[int(d)] for d in Vz_from_W], dtype=np.int64)


with VTKFile(
    mesh.comm, f"./results/axisymmetric/axisymmetric.pvd", "w"
) as vtk, VTKFile(sphere_mesh.comm, f"./results/axisymmetric/sphere.pvd", "w") as vtk2:

    ANIMATION = True
    if ANIMATION == True:
        # x_center = np.linspace(-7.0, 7.0, Nframes)
        theta_c = np.linspace(0.0, 2 * np.pi, Nframes)
        rc = 3.0
        for frame, theta_c_ in tqdm(
            enumerate(theta_c), desc="Solving Contact", unit="it"
        ):
            contact_center = np.array([rc * np.cos(theta_c_), rc * np.sin(theta_c_)])
            indenter.center = np.array(
                [rc * np.cos(theta_c_), rc * np.sin(theta_c_), 1 + Rindenter - displ]
            )
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
            f_ccg, _, _ = lemkelcp(Sc_dense, gap, max_iter)
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
                b_.setValue(dof * tdim + 2, f_ccg[i])
                b_.assemble()

            solver_petsc.solve(b_, u_)

            u_fenics.x.array[:] = -u_.array
            u_fenics.x.scatter_forward()

            f_ccg_W = f_ccg[perm_W_from_Ic]

            p_ccg_W = fc_to_tc(
                f_ccg_W, ksp_Mcc, rhs_Mcc, sol_Mcc, rhs_arr, sol_arr, sign=+1.0
            )

            pW.x.array[:] = 0.0
            pW.x.array[dofs_c_W] = np.abs(p_ccg_W)
            pW.x.scatter_forward()

            sphere_mesh.geometry.x[:, :3] = (
                sphere_ref_x * indenter.radius + indenter.center
            )

            vtk.write_function([u_fenics, pW], frame)
            vtk2.write_function(u_sphere, t=frame)
