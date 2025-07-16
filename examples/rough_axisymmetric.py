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

from cofebem.mesh.wrinkly_tyre import rough_hollow_cylinder
from cofebem.contact.lcp_solvers.ccg import CCG
from cofebem.contact.rigid_indenters import flat

from cofebem.contact.Sc import Sc

# -------------------------------------------------------------------------------------------------------
#  Mesh and material parameters
# -------------------------------------------------------------------------------------------------------

r_inner = 2
r_outer = 5

nr = 40
nt = 40  # int((r_inner + r_outer) * nr * np.pi / (r_outer - r_inner))
print(nt)
nz = 3

rough_hollow_cylinder(
    nr, nt, nz, r_inner=r_inner, r_outer=r_outer, A=0.12, B=0.07, k_r=20, k_theta=4
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
tol = 1.0e-8

# def Gamma_c_selector(x):
#     return np.isclose(x[2], 1, atol=tol)


def Gamma_c_selector(x):
    z = x[2]
    return (z >= 0.8 - tol) & (z <= 10.0 + tol)


Gamma_c = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_c_selector)
Ic = locate_dofs_topological(V, fdim, Gamma_c)


# def reference_line_selector(x):
#     return np.isclose(x[1], 0.0, atol=tol) & (x[2] >= 0.95 - tol) & (x[2] <= 10.0 + tol)


# Gamma_c_ref = locate_entities_boundary(mesh, fdim, reference_line_selector)

# Ic_ref = locate_dofs_geometrical(V, reference_line_selector)


Gamma_c_x = mesh.geometry.x[Ic].reshape(-1, tdim)

# angles = (np.arctan2(Gamma_c_x[:, 1], Gamma_c_x[:, 0]) + 2 * np.pi) % (2 * np.pi)
# order_angle = np.argsort(angles)
# Ic_sorted = Ic[order_angle]


# Ic_sorted = Ic_sorted.reshape(nt, nr + 1)

# Gamma_c_x_segments = mesh.geometry.x[Ic_sorted].reshape(nt, nr + 1, tdim)
# radii = np.sqrt(Gamma_c_x_segments[:, :, 0] ** 2 + Gamma_c_x_segments[:, :, 1] ** 2)

# order_radius = np.argsort(radii, axis=1)
# Ic_sorted = np.take_along_axis(Ic_sorted, order_radius, axis=1).flatten()


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

# # -------------------------------------------------------------------------------------------------------
# #  Contact Problem
# # -------------------------------------------------------------------------------------------------------

displ = 0.3
Rindenter = 10.0

max_iter = 1000
tolerance = 1e-5
error_type = "nw"
pfactor = 1e8
Nframes = 70

u_fenics = Function(V)
u_fenics.name = "u"

p_fenics = Function(V)
p_fenics.name = "p"


with VTKFile(mesh.comm, f"./results/rough/rough.pvd", "w") as vtk:
    vtk.write_mesh(mesh)
    vtk.write_function([u_fenics, p_fenics], 0)


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

    for frame, z0 in tqdm(enumerate(z_positions), desc="Solving contact", unit="it"):
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

        p_ccg, _, _ = CCG(Sc_, error_type, gap, max_iter, tolerance).solve()

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
