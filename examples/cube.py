import numpy as np

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
from dolfinx.io import VTKFile

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

nx = 20
ny = 20
nz = 10

l = 1

mesh = create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([l, l, l])],
    [nx, ny, nz],
    CellType.hexahedron,
    ghost_mode=GhostMode.shared_facet,
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
    return np.isclose(x[2], l, atol=tol)


Gamma_c = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_c_selector)
Ic = locate_dofs_topological(V, fdim, Gamma_c)
Gamma_c_x = mesh.geometry.x[Ic].reshape(-1, tdim)

# -------------------------------------------------------------------------------------------------------
#  Sc construction: Contact compliance matrix
# -------------------------------------------------------------------------------------------------------

Sc_ = Sc(problem.A, problem.b, tdim, Ic)
Sc_dense = Sc_.by_sampling()

Sc_.save(file="./results/cube/Sc.npy")

# -------------------------------------------------------------------------------------------------------
#  Contact Problem
# -------------------------------------------------------------------------------------------------------

# Sc_dense = np.load("./results/cube/Sc.npy")


delta = 0.2
R = 0.8

max_iter = 1000
tol = 1e-9
err_type = "nw"

pfactor = 1e8


Nframes = 70

u_fenics = Function(V)
u_fenics.name = "u"

p_fenics = Function(V)
p_fenics.name = "p"


with VTKFile(mesh.comm, f"./results/cube/cube.pvd", "w") as vtk:
    vtk.write_mesh(mesh)
    vtk.write_function([u_fenics, p_fenics], 0)


ANIMATION = True
if ANIMATION:
    x_center = np.linspace(-0.1, 1.1 * l, Nframes)
    for frame, xc in tqdm(enumerate(x_center)):
        contact_center = np.array([xc, l / 2])
        g = (
            parabolic(
                Gamma_c_x[:, 0],
                Gamma_c_x[:, 1],
                contact_center[0],
                contact_center[1],
                R,
                np.full_like(Gamma_c_x[:, 2], l - delta),
            )
            - Gamma_c_x[:, 2]
        )
        penetrating_nodes = np.where(g < 0)[0]

        # p_lemke, _, _ = lemkelcp(Sc_dense, g, max_iter)

        p_ccg, _, _ = CCG(Sc_dense, err_type, g, max_iter, tol, pfactor).solve()

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
            b_.setValue(dof * tdim + 2, p_ccg[i])
            b_.assemble()

        solver_petsc.solve(b_, u_)

        u_fenics.x.array[:] = -u_.array
        u_fenics.x.scatter_forward()

        p_fenics.x.array[:] = b_.array
        p_fenics.x.scatter_forward()

        vtk.write_function([u_fenics, p_fenics], frame + 1)
