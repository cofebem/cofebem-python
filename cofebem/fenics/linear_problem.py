import numpy as np
from petsc4py import PETSc

from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import (
    Constant,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import (
    LinearProblem,
    assemble_matrix_mat,
    assemble_vector,
    apply_lifting,
)
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


def build_system(
    mesh,
    E=1.0e9,
    nu=0.3,
    element_type="Lagrange",
    element_order=1,
    Gamma_u_locator=None,
):
    tdim = mesh.topology.dim
    fdim = tdim - 1

    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    V = functionspace(mesh, (element_type, element_order, (tdim,)))
    u = TrialFunction(V)
    v = TestFunction(V)

    def epsilon(w):
        return sym(grad(w))

    def sigma(w):
        return lmbda * tr(epsilon(w)) * Identity(tdim) + 2 * mu * epsilon(w)

    f_v = Constant(mesh, np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType))
    a = inner(sigma(u), epsilon(v)) * dx
    Lform = inner(f_v, v) * dx

    Gamma_u = locate_entities_boundary(mesh, fdim, Gamma_u_locator)
    Gamma_u_dofs = locate_dofs_topological(V, fdim, Gamma_u)
    u0 = np.array([0, 0, 0], dtype=PETSc.ScalarType)
    bc = dirichletbc(u0, Gamma_u_dofs, V)
    bcs = [bc]

    problem = LinearProblem(
        a=a,
        L=Lform,
        bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    problem._A.zeroEntries()
    assemble_matrix_mat(problem._A, problem._a, bcs=problem.bcs)
    problem._A.assemble()

    with problem._b.localForm() as b_loc:
        b_loc.set(0)
    assemble_vector(problem._b, problem._L)

    apply_lifting(problem._b, [problem._a], bcs=[problem.bcs])
    problem._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    for bc in problem.bcs:
        bc.set(problem._b.array_w)

    return V, problem._A, problem._b, problem
