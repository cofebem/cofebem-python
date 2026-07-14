import numpy as np
import time
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
from dolfinx.io import XDMFFile, gmsh, VTKFile

from mpi4py import MPI
from petsc4py import PETSc
from tqdm import tqdm

import mpi4py.MPI as MPI

from cofebem.contact.Sc import Sc
from cofebem.contact.rigid_indenters import parabolic
from cofebem.contact.lcp_solvers.ccg import CCG
from cofebem.mesh.hemisphere import hemisphere

# -------------------------------------------------------------------------------------------------------
#  Mesh and material parameters
# -------------------------------------------------------------------------------------------------------


mesh = gmsh.read_from_msh(
    "./msh_files/hemisphere5.msh", MPI.COMM_WORLD, 0, gdim=3
).mesh


# nr = 10
# nt = 20
# np_ = 20
# hemisphere(nr, nt, np_)

# with XDMFFile(MPI.COMM_WORLD, "hemisphere.xdmf", "r") as xdmf:
#     mesh = xdmf.read_mesh(name="Grid")

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
    a=a(u, v), L=L(v), bcs=[bc], petsc_options_prefix ="le", petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)

###Contact Solving Setup######
# from cofebem.fenics import Contact

# contact = Contact(mesh1, mesh2, Gamma_c1, Gamma_c2, problem)
# fc = contact.solve()

problem.solve()

# problem.A.assemble()
# problem.b.assemble()

# -------------------------------------------------------------------------------------------------------
#  Gamma_c : Contact Region
# -------------------------------------------------------------------------------------------------------

tol = 1.0e-8


def Gamma_c_selector(x):
    z = x[2]
    return (z >= 0.7 - tol) & (z <= 10.0 + tol)


Gamma_c = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_c_selector)
Ic = locate_dofs_topological(V, fdim, Gamma_c)
Gamma_c_x = mesh.geometry.x[Ic].reshape(-1, tdim)

# -------------------------------------------------------------------------------------------------------
#  Sc construction: Contact compliance matrix
# -------------------------------------------------------------------------------------------------------

Sc_ = Sc(problem.A, problem.b, tdim, Ic)
Sc_dense = Sc_.by_sampling()

plt.imshow(Sc_dense, cmap="inferno")
# plt.colorbar()
plt.show()

# Sc_.save(file="./results/hemisphere/Sc.npy")

# -------------------------------------------------------------------------------------------------------
#  Contact Problem
# -------------------------------------------------------------------------------------------------------

displ = 0.4
Rindenter = 0.4

max_iter = 1000
tolerance = 1e-5
error_type = "nw"

pfactor = 1e8


Nframes = 70

u_fenics = Function(V)
u_fenics.name = "u"

p_fenics = Function(V)
p_fenics.name = "p"

with VTKFile(mesh.comm, f"./results/hemisphere/hemisphere.pvd", "w") as vtk:
    vtk.write_mesh(mesh)
    vtk.write_function([u_fenics, p_fenics], 0)

ANIMATION = False
if ANIMATION:
    x_center = np.linspace(-0.9, 0.9, Nframes)
    for frame, xc in tqdm(enumerate(x_center), desc="Solving Contact", unit="it"):
        contact_center = np.array([xc, 0.0])
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

        p_ccg, _, _ = CCG(
            Sc_dense, error_type, gap, max_iter, tolerance, pfactor
        ).solve()

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
