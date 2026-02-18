import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.io import gmshio, VTKFile
from dolfinx.fem import Constant, functionspace, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import (
    LinearProblem,
    assemble_matrix_mat,
    assemble_vector,
    apply_lifting,
)
from ufl import Identity, TrialFunction, TestFunction, sym, grad, inner, tr, dx

import numpy as np
from dolfinx.fem import Function, functionspace, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from ufl import FacetNormal, Measure, TrialFunction, TestFunction, inner, dx


def build_contact_normals(
    mesh,
    contact_vertex_ids: np.ndarray,
    ds_contact,
    eps: float = 1e-8,
):

    gdim = mesh.geometry.dim

    Vn = functionspace(mesh, ("CG", 1, (gdim,)))

    n = FacetNormal(mesh)
    u = TrialFunction(Vn)
    v = TestFunction(Vn)

    a = eps * inner(u, v) * dx + inner(u, v) * ds_contact
    L = inner(n, v) * ds_contact

    normal_field = Function(Vn)
    normal_field.name = "contact_normals"

    LinearProblem(
        a=a,
        L=L,
        u=normal_field,
        bcs=[],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    ).solve()

    normal_field.x.scatter_forward()

    contact_vertex_ids = np.asarray(contact_vertex_ids, dtype=np.int32)
    n_contact = contact_vertex_ids.size

    normals = np.zeros((n_contact, gdim), dtype=np.float64)

    for comp in range(gdim):
        Vc, sub_to_parent = Vn.sub(comp).collapse()
        sub_to_parent = np.asarray(sub_to_parent, dtype=np.int32)

        dofs_sub = locate_dofs_topological(Vc, 0, contact_vertex_ids)
        dofs_sub = np.asarray(dofs_sub, dtype=np.int32)

        parent_dofs = sub_to_parent[dofs_sub]
        normals[:, comp] = normal_field.x.array[parent_dofs]

    norms = np.linalg.norm(normals, axis=1)

    if np.any(norms < 1e-14):
        bad = np.where(norms < 1e-14)[0][:10]
        raise RuntimeError(
            f"Zero/near-zero normal at {bad.size} contact vertices (first indices: {bad}). "
            "Check that ds_contact really corresponds to the contact boundary and that "
            "contact_vertex_ids lie on it."
        )

    normals /= norms[:, None]
    return normals


def build_system(mesh, E=1.0e9, nu=0.3, Gamma_u_locator: function = None):
    tdim = mesh.topology.dim
    fdim = tdim - 1

    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    # ---------------- Variational forms ----------------
    V = functionspace(mesh, ("Lagrange", 1, (tdim,)))
    u = TrialFunction(V)
    v = TestFunction(V)

    def epsilon(w):
        return sym(grad(w))

    def sigma(w):
        return lmbda * tr(epsilon(w)) * Identity(tdim) + 2 * mu * epsilon(w)

    f_v = Constant(mesh, np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType))
    a = inner(sigma(u), epsilon(v)) * dx
    Lform = inner(f_v, v) * dx

    # ---------------- Dirichlet BC ----------------

    Gamma_u = locate_entities_boundary(mesh, fdim, Gamma_u_locator)
    Gamma_u_dofs = locate_dofs_topological(V, fdim, Gamma_u)
    u0 = np.array([0, 0, 0], dtype=PETSc.ScalarType)
    bc = dirichletbc(u0, Gamma_u_dofs, V)
    bcs = [bc]

    # ---------------- Problem setup ----------------
    problem = LinearProblem(
        a=a,
        L=Lform,
        bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    # ---------------- Assemble system ----------------
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

    A = problem._A
    b = problem._b

    return V, A, b, problem


comm = MPI.COMM_WORLD
# ---------------- Meshes ----------------
cube, _, _ = gmshio.read_from_msh("./msh_files/cube_tetra.msh", comm, 0, gdim=3)
hemisphere, _, _ = gmshio.read_from_msh("./msh_files/hemisphere1.msh", comm, 0, gdim=3)

V_cube, A_cube, b_cube, problem_cube = build_system(
    cube,
    E=1.0e9,
    nu=0.3,
    Gamma_u_locator=lambda x: np.isclose(x[2], 0.0),
)

V_hemisphere, A_hemisphere, b_hemisphere, problem_hemisphere = build_system(
    hemisphere,
    E=1.0e9,
    nu=0.3,
    Gamma_u_locator=lambda x: np.isclose(x[2], 1.9),
)


# ---------------- Contact BCs ----------------
def Gamma_c_cube(x):
    return (x[2] > 0.5) & (x[0] > -0.1)


Gamma_c = locate_entities_boundary(cube, cube.topology.dim, Gamma_c_cube)
Gamma_c_id = 2
Gamma_c_tags = np.full(Gamma_c.shape, Gamma_c_id, dtype=np.int32)

facet_indices = np.hstack([Gamma_t, Gamma_c]).astype(np.int32)
facet_values = np.hstack(
    [
        Gamma_t_tags,
        Gamma_c_tags,
    ]
).astype(np.int32)

order = np.argsort(facet_indices)
facet_indices = facet_indices[order]
facet_values = facet_values[order]

mt = meshtags(mesh, fdim, facet_indices, facet_values)

ds = Measure("ds", domain=cube, subdomain_data=facet_tags)
normals_cube, normal_field_cube = build_contact_normals(
    mesh=cube,
    contact_vertex_ids=contact_vertex_ids_cube,  # vertex ids (dim=0 entities)
    ds_contact=ds(contact_tag),
)
