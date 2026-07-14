import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.mesh import (
    locate_entities_boundary,
    create_box,
    CellType,
    entities_to_geometry,
)
from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
    form,
)
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    set_bc,
    create_vector,
)
from ufl import (
    Identity,
    TestFunction,
    TrialFunction,
    grad,
    inner,
    tr,
    det,
    ln,
    dx,
    variable,
    derivative,
)

from cofebem.bodies.sphere_indenter import Sphere
from cofebem.contact.Sc_normal import Sc_normal

# from cofebem.contact.lcp_solvers.lemke import lemkelcp


# ============================================================
# Helpers
# ============================================================


def unique_vertices_from_facets(mesh, facet_ids):
    """
    Return unique geometry vertex ids belonging to the given facets.
    Works well for P1 geometric meshes.
    """
    mesh.topology.create_connectivity(mesh.topology.dim - 1, 0)
    g = entities_to_geometry(mesh, mesh.topology.dim - 1, facet_ids)
    return np.unique(g.reshape(-1)).astype(np.int64)


def contact_vertex_to_vector_dofs(V, vertex_ids):
    """
    For a CG1 vector space on a standard mesh, recover parent dofs attached
    to the given boundary vertices.
    """
    tdim = V.mesh.topology.dim
    dofs_xyz = []
    for c in range(tdim):
        Vc, sub_to_parent = V.sub(c).collapse()
        dofs_c = locate_dofs_topological((V.sub(c), Vc), 0, vertex_ids)[0]
        dofs_xyz.append(dofs_c)

    dofs_xyz = np.stack(dofs_xyz, axis=1)
    return dofs_xyz


def assemble_tangent_and_residual(jacobian_form, residual_form, u, bcs):
    """
    Assemble the current tangent matrix K(u) and residual vector R(u).
    """
    A = assemble_matrix(form(jacobian_form), bcs=bcs)
    A.assemble()

    r = assemble_vector(form(residual_form))
    apply_lifting(r, [form(jacobian_form)], [bcs], x0=[u.x.petsc_vec], alpha=-1.0)
    r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(r, bcs, u.x.petsc_vec, -1.0)

    return A, r


def build_contact_rhs_from_nodal_forces(V, contact_dofs_xyz, fc_xyz):
    """
    Build a global RHS vector from nodal contact force vectors applied at
    contact vector dofs.
    """
    rhs = create_vector(
        form(
            inner(
                Constant(V.mesh, np.zeros(V.mesh.topology.dim, dtype=PETSc.ScalarType)),
                TestFunction(V),
            )
            * dx
        )
    )
    rhs.set(0.0)

    flat_dofs = contact_dofs_xyz.reshape(-1)
    flat_vals = fc_xyz.reshape(-1)
    rhs.setValues(flat_dofs, flat_vals, addv=PETSc.InsertMode.INSERT_VALUES)
    rhs.assemble()
    return rhs


# ============================================================
# Mesh
# ============================================================

Lbox = 1.0
ncells = 5
mesh = create_box(
    MPI.COMM_WORLD,
    [[0.0, 0.0, 0.0], [Lbox, Lbox, Lbox]],
    [ncells * 10, ncells * 10, ncells],
    CellType.hexahedron,
)

tdim = mesh.topology.dim
fdim = tdim - 1


# ============================================================
# Material
# ============================================================

E = 1.0e9
nu = 0.3
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))

V = functionspace(mesh, ("Lagrange", 1, (tdim,)))

u = Function(V)
du = Function(V)  # Newton increment (solution of tangent system)
v = TestFunction(V)
w = TrialFunction(V)

I = Identity(tdim)
F = variable(I + grad(u))
C = F.T * F
J = det(F)

psi = (mu / 2) * (tr(C) - tdim) - mu * ln(J) + (lmbda / 2) * ln(J) ** 2
b = Constant(mesh, np.zeros(tdim, dtype=PETSc.ScalarType))

residual = derivative(psi * dx, u, v) - inner(b, v) * dx
jacobian = derivative(residual, u, w)


# ============================================================
# Boundary conditions
# ============================================================

tol = 1.0e-8


def Gamma_u_locator(x):
    return np.isclose(x[2], 0.0, atol=tol)


Gamma_u = locate_entities_boundary(mesh, fdim, Gamma_u_locator)
dofs_u = locate_dofs_topological(V, fdim, Gamma_u)
u_D = np.zeros(tdim, dtype=PETSc.ScalarType)
bcs = [dirichletbc(u_D, dofs_u, V)]


# ============================================================
# Contact geometry
# ============================================================


def Gamma_c_locator(x):
    return np.isclose(x[2], Lbox, atol=tol)


Gamma_c = locate_entities_boundary(mesh, fdim, Gamma_c_locator)

# Geometry vertices on contact boundary
Ic_vertices = unique_vertices_from_facets(mesh, Gamma_c)

# Vector dofs attached to those vertices
contact_dofs_xyz = contact_vertex_to_vector_dofs(V, Ic_vertices)

# Current coordinates of contact vertices in the reference mesh
Xc = mesh.geometry.x[Ic_vertices]

radius = 0.5
delta = 0.1
sphere = Sphere(
    center=np.array([Lbox / 2, Lbox / 2, Lbox + radius - delta]), radius=radius
)

# Example: fixed normals for a flat top surface in the reference config
normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(Ic_vertices), 1))

# Initial geometric gap on contact vertices
g0 = sphere.gap(Xc)


# ============================================================
# Newton loop driven manually
# ============================================================

max_it = 25
rtol = 1e-8
atol = 1e-8

comm = mesh.comm

for k in range(max_it):
    # --------------------------------------------------------
    # 1. Assemble current tangent and current residual
    # --------------------------------------------------------
    A, R = assemble_tangent_and_residual(jacobian, residual, u, bcs)

    res_norm = R.norm()
    if comm.rank == 0:
        print(f"[Newton {k}] residual norm = {res_norm:.6e}")

    if res_norm < atol:
        if comm.rank == 0:
            print("Converged by absolute residual.")
        break

    # --------------------------------------------------------
    # 2. Update current contact geometry if needed
    # --------------------------------------------------------
    # Current contact points in deformed configuration
    u_array = u.x.array.reshape((-1, tdim))
    xc_current = mesh.geometry.x[Ic_vertices] + u_array[Ic_vertices]

    # Recompute current gap if using total-gap contact
    gk = sphere.gap(xc_current)

    # If normals evolve, update them here.
    # For now, keep them fixed:
    normals_k = normals

    # --------------------------------------------------------
    # 3. Build current tangent compliance Sc^(k)
    # --------------------------------------------------------
    # Sc_normal expects vertex ids (Ic_vertices), not vector dofs.
    sc_builder = Sc_normal(
        A=A,
        b=R,  # only used to duplicate/create PETSc vecs
        Ic=Ic_vertices,
        normals=normals_k,
        tdim=tdim,
        f0=1e9,
    )

    Snn_k = sc_builder.sample_n(show=(comm.rank == 0))

    # --------------------------------------------------------
    # 4. Solve contact problem in incremental or lagged form
    # --------------------------------------------------------
    # Example placeholder:
    #
    # Solve something like:
    #   delta_g = Snn_k @ delta_p + gk
    # with complementarity
    #
    # Here we just put zero incremental contact force as placeholder.
    delta_p = np.zeros(len(Ic_vertices), dtype=PETSc.ScalarType)

    # Example if you later solve an LCP:
    # delta_p, _, _ = lemkelcp(Snn_k, gk)

    # Convert normal nodal forces to xyz nodal forces
    fc_xyz = delta_p[:, None] * normals_k

    # --------------------------------------------------------
    # 5. Build linearized Newton RHS:
    #    K_T du = -R + f_contact
    # --------------------------------------------------------
    rhs_contact = R.duplicate()
    rhs_contact.set(0.0)

    flat_dofs = contact_dofs_xyz.reshape(-1)
    flat_vals = fc_xyz.reshape(-1)
    rhs_contact.setValues(flat_dofs, flat_vals, addv=PETSc.InsertMode.INSERT_VALUES)
    rhs_contact.assemble()

    rhs_total = R.duplicate()
    rhs_total.set(0.0)
    rhs_total.axpy(-1.0, R)  # -R
    rhs_total.axpy(+1.0, rhs_contact)  # + f_contact
    rhs_total.assemble()

    # Re-apply BC treatment to the linearized increment system
    # since du must satisfy homogeneous Dirichlet conditions
    set_bc(rhs_total, bcs)

    # --------------------------------------------------------
    # 6. Solve K_T du = rhs_total
    # --------------------------------------------------------
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.setFromOptions()
    ksp.setUp()

    du.x.petsc_vec.set(0.0)
    ksp.solve(rhs_total, du.x.petsc_vec)
    du.x.scatter_forward()

    du_norm = du.x.petsc_vec.norm()
    if comm.rank == 0:
        print(f"[Newton {k}] ||du|| = {du_norm:.6e}")

    # --------------------------------------------------------
    # 7. Update state
    # --------------------------------------------------------
    u.x.petsc_vec.axpy(1.0, du.x.petsc_vec)
    u.x.scatter_forward()

    if du_norm < rtol:
        if comm.rank == 0:
            print("Converged by increment norm.")
        break

if comm.rank == 0:
    print("DONE")
