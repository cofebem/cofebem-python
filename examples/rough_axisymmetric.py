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

from cofebem.mesh.wrinkly_tyre import rough_hollow_cylinder
from cofebem.contact.lcp_solvers.ccg import CCG
from cofebem.contact.lcp_solvers.lemke import lemkelcp
from cofebem.contact.rigid_indenters import flat

from cofebem.contact.Sc import Sc

from cofebem.bodies.plane_indenter import Plane

# -------------------------------------------------------------------------------------------------------
#  Mesh and material parameters
# -------------------------------------------------------------------------------------------------------

r_inner = 2
r_outer = 5

nr = 40
nt = 40  # int((r_inner + r_outer) * nr * np.pi / (r_outer - r_inner))A=0.12, B=0.07, kr =9, kt = 4
print(nt)
nz = 3

rough_hollow_cylinder(
    nr, nt, nz, r_inner=r_inner, r_outer=r_outer, A=0.06, B=0.07, k_r=14, k_theta=4
)

with XDMFFile(MPI.COMM_WORLD, "rough_hollow_cylinder.xdmf", "r") as xdmf:
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
tol = 1.0e-5


def Gamma_u_selector(x):
    return np.isclose(x[2], 0, atol=tol)


Gamma_u = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_u_selector)
Gamma_u_id = 1
Gamma_u_tags = np.full(Gamma_u.shape, Gamma_u_id, dtype=np.int32)
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
    a=a(u, v), L=L(v), bcs=[bc], petsc_options_prefix="le", petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)

problem.solve()

# -------------------------------------------------------------------------------------------------------
#  Construct S_c for a symmetric domain
# -------------------------------------------------------------------------------------------------------
angle_tol = 1.0e-8
tol = 1.0e-8

# def Gamma_c_selector(x):
#     return np.isclose(x[2], 1, atol=tol)


def Gamma_c_selector(x):
    z = x[2]
    return (z >= 0.8 - tol) & (z <= 10.0 + tol)


Gamma_c = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_c_selector)
Ic = locate_dofs_topological(V, fdim, Gamma_c)


Gamma_c_id = 2
Gamma_c_tags = np.full(Gamma_c.shape, Gamma_c_id, dtype=np.int32)


def reference_line_selector(x):
    return np.isclose(x[1], 0.0, atol=tol) & (x[2] >= 0.95 - tol) & (x[2] <= 10.0 + tol)


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

    return Sc_classic_unsorted


# Sc_ref = compute_Sc_ref(problem.A, problem.b, Ic_ref, Ic_sorted, tdim, mesh.comm)
# Sc = construct_Sc_from_Sc_ref(Sc_ref, nt, Ic)

Sc_ = Sc(problem.A, problem.b, tdim, Ic).by_sampling()


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


# # -------------------------------------------------------------------------------------------------------
# #  Contact Problem
# # -------------------------------------------------------------------------------------------------------
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


displ = 0.3
Rindenter = 10.0

max_iter = 1000
tolerance = 1e-5
error_type = "nw"
pfactor = 1e8
Nframes = 70

u_fenics = Function(V)
u_fenics.name = "u"


indenter = Plane(point=np.array([0, 0, 0]))

plane_mesh = gmsh.read_from_msh(
    "./msh_files/ground.msh", MPI.COMM_WORLD, 0, gdim=3
).mesh

plane_ref_x = plane_mesh.geometry.x[:, :3].copy()
V_plane = functionspace(plane_mesh, ("Lagrange", 1))
u_plane = Function(V_plane)
u_plane.name = "indenter"


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

with VTKFile(mesh.comm, f"./results/rough/rough.pvd", "w") as vtk, VTKFile(
    plane_mesh.comm, f"./results/rough/plane.pvd", "w"
) as vtk2:

    ANIMATION = True
    if ANIMATION:
        # # x_center = np.linspace(-7.0, 7.0, Nframes)
        # theta_c = np.linspace(0.0, 2 * np.pi, Nframes)
        # rc = 3.0
        # for frame, theta_c_ in enumerate(theta_c):
        #     contact_center = np.array([rc * np.cos(theta_c_), rc * np.sin(theta_c_)])
        #     gap = (
        #         flat_indenter(
        #             Gamma_c_x[:, 0],
        #             Gamma_c_x[:, 1],
        #             contact_center[0],
        #             contact_center[1],
        #             Rindenter,
        #             np.ones_like(Gamma_c_x[:, 2]) - displ,
        #         )
        #         - Gamma_c_x[:, 2]
        #     )

        contact_center = np.array([0.0, 0.0])
        z_start = 1.2
        z_end = 1.0 - displ

        z_positions = np.linspace(z_start, z_end, Nframes)

        for frame, z0 in tqdm(
            enumerate(z_positions), desc="Solving contact", unit="it"
        ):
            gap = (
                flat(
                    Gamma_c_x[:, 0],
                    Gamma_c_x[:, 1],
                    contact_center[0],
                    contact_center[1],
                    Rindenter,
                    np.full_like(Gamma_c_x[:, 2], z0),
                )
                - Gamma_c_x[:, 2]
            )
            penetrating_nodes = np.where(gap < 0)[0]

            f_ccg, _, _ = lemkelcp(Sc_, gap, max_iter)

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

            plane_mesh.geometry.x[:, :3] = plane_ref_x + np.array([0, 0, z0])

            vtk.write_function([u_fenics, pW], frame)
            vtk2.write_function(u_plane, t=frame)
