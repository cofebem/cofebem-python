import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from tqdm import tqdm

from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.io import gmshio, VTKFile
from dolfinx.fem import (
    Constant,
    Function,
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
    FacetNormal,
    Measure,
)
from dolfinx.mesh import entities_to_geometry

import matplotlib.pyplot as plt


def Sc_n(A, Ic, normals, tdim=3, f0=1e9, show=True):
    comm = MPI.COMM_WORLD
    f0 = float(f0)
    tdim = int(tdim)
    Ic = np.asarray(Ic, dtype=np.int64)
    normals = np.asarray(normals, dtype=float)
    nc = int(Ic.size)

    if normals.shape != (nc, tdim):
        raise ValueError(f"normals must have shape ({nc}, {tdim}).")

    nrm = np.linalg.norm(normals, axis=1)
    if np.any(nrm == 0.0):
        raise ValueError("Some normals have zero norm.")
    n = normals / nrm[:, None]

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.setFromOptions()
    ksp.setUp()

    rhs = A.createVecRight()
    u = A.createVecRight()

    cdofs = np.stack([Ic * tdim + c for c in range(tdim)], axis=1).astype(np.int32)
    cdofs_flat = cdofs.reshape(-1).astype(np.int32)

    uc = PETSc.Vec().createSeq(tdim * nc, comm=PETSc.COMM_SELF)
    is_from = PETSc.IS().createGeneral(cdofs_flat, comm=comm)
    is_to = PETSc.IS().createStride(tdim * nc, first=0, step=1, comm=PETSc.COMM_SELF)
    scat = PETSc.Scatter().create(u, is_from, uc, is_to)

    Sc = np.zeros((nc, nc), dtype=PETSc.ScalarType)

    it = range(nc)
    if show:
        it = tqdm(it, desc="Sampling Snn (normal)", unit="it")

    for j in it:
        rhs.set(0.0)
        rhs.setValues(cdofs[j], f0 * n[j], addv=PETSc.InsertMode.INSERT_VALUES)
        rhs.assemble()

        ksp.solve(rhs, u)

        uc.set(0.0)
        scat.scatter(
            u,
            uc,
            addv=PETSc.InsertMode.INSERT_VALUES,
            mode=PETSc.ScatterMode.FORWARD,
        )

        uc_xyz = uc.getArray(readonly=True).copy().reshape(nc, tdim)
        Sc[:, j] = np.einsum("ij,ij->i", n, uc_xyz) / f0

    return Sc


def _kelvin_tensor_3d(rvec, E, nu):
    r = np.linalg.norm(rvec)
    if r <= 0.0:
        raise ValueError("Kelvin tensor undefined at r=0 without regularization.")
    mu = E / (2.0 * (1.0 + nu))
    I = np.eye(3, dtype=float)
    rr = np.outer(rvec, rvec) / (r * r)
    coeff = 1.0 / (16.0 * np.pi * mu * (1.0 - nu) * r)
    return coeff * ((3.0 - 4.0 * nu) * I + rr)


def nodal_areas_from_facets(mesh, facet_ids, contact_vertex_ids):
    fdim = mesh.topology.dim - 1
    facet_ids = np.asarray(facet_ids, dtype=np.int32)
    contact_vertex_ids = np.asarray(contact_vertex_ids, dtype=np.int32)

    x = mesh.geometry.x
    facet_geom = np.asarray(entities_to_geometry(mesh, fdim, facet_ids), dtype=np.int32)

    v2l = {int(v): i for i, v in enumerate(contact_vertex_ids)}
    a = np.zeros(contact_vertex_ids.size, dtype=float)

    for verts in facet_geom:
        verts = np.asarray(verts, dtype=np.int32)
        if verts.size != 3:
            raise NotImplementedError(
                "This implementation assumes triangular boundary facets."
            )
        x0, x1, x2 = x[verts[0]], x[verts[1]], x[verts[2]]
        area = 0.5 * np.linalg.norm(np.cross(x1 - x0, x2 - x0))
        share = area / 3.0
        for v in verts:
            if int(v) in v2l:
                a[v2l[int(v)]] += share

    return a


def project_point_to_triangle_plane(x, tri):
    x = np.asarray(x, dtype=float).reshape(3)
    tri = np.asarray(tri, dtype=float).reshape(3, 3)

    y0 = tri[0]
    e1 = tri[1] - y0
    e2 = tri[2] - y0

    n = np.cross(e1, e2)
    nn = np.linalg.norm(n)
    if nn < 1e-15:
        raise ValueError("Degenerate triangle in project_point_to_triangle_plane.")

    n_unit = n / nn
    d_signed = np.dot(x - y0, n_unit)
    p = x - d_signed * n_unit

    G = np.array(
        [
            [np.dot(e1, e1), np.dot(e1, e2)],
            [np.dot(e2, e1), np.dot(e2, e2)],
        ],
        dtype=float,
    )
    rhs = np.array(
        [
            np.dot(p - y0, e1),
            np.dot(p - y0, e2),
        ],
        dtype=float,
    )

    xi1, xi2 = np.linalg.solve(G, rhs)
    inside = (xi1 >= -1e-12) and (xi2 >= -1e-12) and (xi1 + xi2 <= 1.0 + 1e-12)
    return p, xi1, xi2, abs(d_signed), inside


def reference_triangle_to_square(xi_star, eta_star):
    xi_star = float(xi_star)
    eta_star = float(eta_star)

    if xi_star < -1e-12 or eta_star < -1e-12 or xi_star + eta_star > 1.0 + 1e-12:
        raise ValueError("Point is outside the reference triangle.")

    u_star = xi_star
    if 1.0 - xi_star < 1e-14:
        v_star = 0.0
    else:
        v_star = eta_star / (1.0 - xi_star)

    v_star = min(max(v_star, 0.0), 1.0)
    return u_star, v_star


def telles_cubic_map_01(zhat, z_star):
    zhat = float(zhat)
    z_star = float(z_star)

    if not (0.0 <= zhat <= 1.0):
        raise ValueError("zhat must be in [0,1].")

    z_star = min(max(z_star, 1e-8), 1.0 - 1e-8)

    s = 2.0 * zhat - 1.0
    a = 2.0 * z_star - 1.0

    t = s + (a - s) * (1.0 - s * s)
    dt_ds = 1.0 - (1.0 - s * s) + (a - s) * (-2.0 * s)

    z = 0.5 * (t + 1.0)
    dz_dzhat = max(abs(dt_ds), 1e-12)

    z = min(max(z, 0.0), 1.0)
    return z, dz_dzhat


def duffy_map_corner(corner, u, v):
    if corner == 0:
        xi1 = u * (1.0 - v)
        xi2 = u * v
        J = u
    elif corner == 1:
        xi1 = 1.0 - u
        xi2 = u * v
        J = u
    else:
        xi1 = u * v
        xi2 = 1.0 - u
        J = u
    return xi1, xi2, J


def _regular_triangle_map(u, v):
    xi1 = u
    xi2 = (1.0 - u) * v
    J = 1.0 - u
    return xi1, xi2, J


def _p1_ref(corner, xi1, xi2):
    if corner == 0:
        return 1.0 - xi1 - xi2
    elif corner == 1:
        return xi1
    else:
        return xi2


def _map_to_triangle(tri, xi1, xi2):
    return tri[0] + xi1 * (tri[1] - tri[0]) + xi2 * (tri[2] - tri[0])


def _triangle_jacobian(tri):
    return np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))


def _triangle_size(tri):
    e01 = np.linalg.norm(tri[1] - tri[0])
    e12 = np.linalg.norm(tri[2] - tri[1])
    e20 = np.linalg.norm(tri[0] - tri[2])
    return max(e01, e12, e20)


def _gauss_legendre_01(nq):
    x, w = np.polynomial.legendre.leggauss(nq)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    return x, w


def incident_triangles_for_vertices(mesh, facet_ids, contact_vertex_ids):
    fdim = mesh.topology.dim - 1
    facet_ids = np.asarray(facet_ids, dtype=np.int32)
    contact_vertex_ids = np.asarray(contact_vertex_ids, dtype=np.int32)

    x = mesh.geometry.x
    facet_geom = np.asarray(entities_to_geometry(mesh, fdim, facet_ids), dtype=np.int32)

    v2l = {int(v): i for i, v in enumerate(contact_vertex_ids)}
    inc = [[] for _ in range(contact_vertex_ids.size)]

    for verts in facet_geom:
        verts = np.asarray(verts, dtype=np.int32)
        if verts.size != 3:
            raise NotImplementedError(
                "This implementation assumes triangular boundary facets."
            )
        tri = x[verts].copy()
        for local_corner, v in enumerate(verts):
            if int(v) in v2l:
                inc[v2l[int(v)]].append((tri, local_corner))

    return inc


def _source_integral_regular(
    x_target,
    n_target,
    tri_s,
    corner_s,
    n_source,
    E,
    nu,
    nq,
):
    xq_s, wq_s = _gauss_legendre_01(nq)
    Js = _triangle_jacobian(tri_s)
    val = 0.0

    for c, us in enumerate(xq_s):
        wus = wq_s[c]
        for d, vs in enumerate(xq_s):
            wvs = wq_s[d]
            xi1_s, xi2_s, Jrs = _regular_triangle_map(us, vs)
            y = _map_to_triangle(tri_s, xi1_s, xi2_s)
            phi_s = _p1_ref(corner_s, xi1_s, xi2_s)

            rvec = x_target - y
            if np.linalg.norm(rvec) < 1e-14:
                continue

            U = _kelvin_tensor_3d(rvec, E, nu)
            val += (n_target @ U @ n_source) * phi_s * Js * Jrs * wus * wvs

    return val


def _source_integral_telles(
    x_target,
    n_target,
    tri_s,
    corner_s,
    n_source,
    E,
    nu,
    nq,
):
    try:
        _, xi1_star, xi2_star, _, inside = project_point_to_triangle_plane(
            x_target, tri_s
        )
        if not inside:
            return _source_integral_regular(
                x_target, n_target, tri_s, corner_s, n_source, E, nu, nq
            )

        u_star, v_star = reference_triangle_to_square(xi1_star, xi2_star)
    except Exception:
        return _source_integral_regular(
            x_target, n_target, tri_s, corner_s, n_source, E, nu, nq
        )

    xq, wq = _gauss_legendre_01(nq)
    Js = _triangle_jacobian(tri_s)
    val = 0.0

    for a, uhat in enumerate(xq):
        wu = wq[a]
        u, Ju = telles_cubic_map_01(uhat, u_star)

        for b, vhat in enumerate(xq):
            wv = wq[b]
            v, Jv = telles_cubic_map_01(vhat, v_star)

            xi1 = u
            xi2 = (1.0 - u) * v
            Jtri = 1.0 - u

            if Jtri <= 0.0 or not np.isfinite(Ju) or not np.isfinite(Jv):
                continue

            y = _map_to_triangle(tri_s, xi1, xi2)
            phi_s = _p1_ref(corner_s, xi1, xi2)

            rvec = x_target - y
            r = np.linalg.norm(rvec)
            if r < 1e-14 or not np.isfinite(r):
                continue

            U = _kelvin_tensor_3d(rvec, E, nu)
            contrib = (n_target @ U @ n_source) * phi_s * Js * Jtri * Ju * Jv * wu * wv

            if np.isfinite(contrib):
                val += contrib

    if not np.isfinite(val):
        return _source_integral_regular(
            x_target, n_target, tri_s, corner_s, n_source, E, nu, nq
        )

    return val


def _source_integral_self_duffy(
    x_target,
    n_target,
    tri_s,
    corner_s,
    n_source,
    E,
    nu,
    nq,
):
    xq_s, wq_s = _gauss_legendre_01(nq)
    Js = _triangle_jacobian(tri_s)
    val = 0.0

    for c, us in enumerate(xq_s):
        wus = wq_s[c]
        for d, vs in enumerate(xq_s):
            wvs = wq_s[d]
            xi1_s, xi2_s, Jds = duffy_map_corner(corner_s, us, vs)
            y = _map_to_triangle(tri_s, xi1_s, xi2_s)
            phi_s = _p1_ref(corner_s, xi1_s, xi2_s)

            rvec = x_target - y
            if np.linalg.norm(rvec) < 1e-14:
                continue

            U = _kelvin_tensor_3d(rvec, E, nu)
            val += (n_target @ U @ n_source) * phi_s * Js * Jds * wus * wvs

    return val


def Sc_Kelvin_source_integrated(
    mesh,
    facet_ids,
    contact_vertex_ids,
    xc,
    normals,
    E,
    nu,
    nq=8,
    nq_self=8,
    dist_threshold=0.5,
    show=True,
):
    contact_vertex_ids = np.asarray(contact_vertex_ids, dtype=np.int32)
    xc = np.asarray(xc, dtype=float)
    normals = np.asarray(normals, dtype=float)

    nc, tdim = xc.shape
    if tdim != 3:
        raise NotImplementedError("Only 3D is implemented.")
    if normals.shape != (nc, tdim):
        raise ValueError(f"normals must have shape ({nc}, {tdim}).")

    nrm = np.linalg.norm(normals, axis=1)
    if np.any(nrm == 0.0):
        raise ValueError("Some normals have zero norm.")
    normals = normals / nrm[:, None]

    areas = nodal_areas_from_facets(mesh, facet_ids, contact_vertex_ids)
    inc = incident_triangles_for_vertices(mesh, facet_ids, contact_vertex_ids)

    Sc = np.zeros((nc, nc), dtype=float)

    it = range(nc)
    if show:
        it = tqdm(it, desc="Building source integrated Kelvin Snn", unit="col")

    for j in it:
        nj = normals[j]
        Aj = areas[j]

        for tri_s, corner_s in inc[j]:
            h_tri = _triangle_size(tri_s)

            for i in range(nc):
                xi = xc[i]
                ni = normals[i]

                if i == j:
                    val = _source_integral_self_duffy(
                        x_target=xi,
                        n_target=ni,
                        tri_s=tri_s,
                        corner_s=corner_s,
                        n_source=nj,
                        E=E,
                        nu=nu,
                        nq=nq_self,
                    )
                else:
                    _, _, _, dist, inside = project_point_to_triangle_plane(xi, tri_s)

                    if inside and dist < dist_threshold * h_tri:
                        val = _source_integral_telles(
                            x_target=xi,
                            n_target=ni,
                            tri_s=tri_s,
                            corner_s=corner_s,
                            n_source=nj,
                            E=E,
                            nu=nu,
                            nq=nq,
                        )
                    else:
                        val = _source_integral_regular(
                            x_target=xi,
                            n_target=ni,
                            tri_s=tri_s,
                            corner_s=corner_s,
                            n_source=nj,
                            E=E,
                            nu=nu,
                            nq=nq,
                        )

                Sc[i, j] += val / Aj

    return Sc


def unique_vertices_from_facets(mesh, facet_ids):
    fdim = mesh.topology.dim - 1
    facet_ids = np.asarray(facet_ids, dtype=np.int32)
    facet_geom = entities_to_geometry(mesh, fdim, facet_ids)
    verts = np.unique(np.asarray(facet_geom, dtype=np.int32).ravel())
    return np.sort(verts).astype(np.int32)


def build_contact_normals(
    mesh,
    contact_vertex_ids,
    ds_contact,
    eps=1e-8,
    save_field=False,
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

    if save_field:
        with VTKFile(mesh.comm, "contact_normals.pvd", "w") as vtk:
            vtk.write_function(normal_field)

    contact_vertex_ids = np.asarray(contact_vertex_ids, dtype=np.int32)
    normals = np.zeros((contact_vertex_ids.size, gdim), dtype=np.float64)

    for comp in range(gdim):
        Vc, sub_to_parent = Vn.sub(comp).collapse()
        sub_to_parent = np.asarray(sub_to_parent, dtype=np.int32)

        dofs_sub = locate_dofs_topological(Vc, 0, contact_vertex_ids)
        dofs_sub = np.asarray(dofs_sub, dtype=np.int32)

        parent_dofs = sub_to_parent[dofs_sub]
        normals[:, comp] = normal_field.x.array[parent_dofs]

    norms = np.linalg.norm(normals, axis=1)
    if np.any(norms < 1e-14):
        raise RuntimeError("Zero/near-zero contact normal detected.")

    normals /= norms[:, None]
    return normals, normal_field


def build_system(mesh, E=1.0e9, nu=0.3, Gamma_u_locator=None):
    tdim = mesh.topology.dim
    fdim = tdim - 1

    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

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


comm = MPI.COMM_WORLD

cube, _, _ = gmshio.read_from_msh("./msh_files/cube_tetra.msh", comm, 0, gdim=3)

cube_V, cube_A, cube_b, cube_problem = build_system(
    cube,
    E=1.0e9,
    nu=0.3,
    Gamma_u_locator=lambda x: np.isclose(x[2], 0.0),
)

cube_tdim = cube.topology.dim
cube_fdim = cube_tdim - 1


def cube_Gamma_c_locator(x):
    return np.isclose(x[2], 1.0, atol=1e-8)


cube_Gamma_c = locate_entities_boundary(cube, cube_fdim, cube_Gamma_c_locator)
cube_Gamma_c_id = 1
cube_Gamma_c_tags = np.full(cube_Gamma_c.shape, cube_Gamma_c_id, dtype=np.int32)

cube_order = np.argsort(cube_Gamma_c)
cube_Gamma_c_facet_indices = cube_Gamma_c[cube_order].astype(np.int32)
cube_Gamma_c_facet_values = cube_Gamma_c_tags[cube_order].astype(np.int32)

cube_mt = meshtags(
    cube, cube_fdim, cube_Gamma_c_facet_indices, cube_Gamma_c_facet_values
)
ds_cube = Measure("ds", domain=cube, subdomain_data=cube_mt)

cube_Ic = unique_vertices_from_facets(cube, cube_Gamma_c)

cube_normals, cube_normal_field = build_contact_normals(
    mesh=cube,
    contact_vertex_ids=cube_Ic,
    ds_contact=ds_cube(cube_Gamma_c_id),
)

xc = cube.geometry.x[cube_Ic]

Sc_FE = Sc_n(cube_A, cube_Ic, cube_normals, cube_tdim, show=True)

Sc_Kelvin_source = Sc_Kelvin_source_integrated(
    mesh=cube,
    facet_ids=cube_Gamma_c,
    contact_vertex_ids=cube_Ic,
    xc=xc,
    normals=cube_normals,
    E=1e9,
    nu=0.3,
    nq=8,
    nq_self=8,
    dist_threshold=0.5,
    show=True,
)

R_source = Sc_FE - Sc_Kelvin_source

_, s_FE, _ = np.linalg.svd(Sc_FE)
_, s_Kelvin_source, _ = np.linalg.svd(Sc_Kelvin_source)
_, s_R_source, _ = np.linalg.svd(R_source)

fig, ax = plt.subplots(figsize=(18, 5), constrained_layout=True)

ax.semilogy(
    np.arange(1, len(s_FE) + 1),
    s_FE,
    marker="o",
    markersize=4,
    linewidth=1.8,
    label=r"$S_c^{FE}$",
)
ax.semilogy(
    np.arange(1, len(s_Kelvin_source) + 1),
    s_Kelvin_source,
    marker="s",
    markersize=4,
    linewidth=1.8,
    label=r"$S_c^{Kelvin,src}$",
)
ax.semilogy(
    np.arange(1, len(s_R_source) + 1),
    s_R_source,
    marker="^",
    markersize=4,
    linewidth=1.8,
    label=r"$S_c^{FE}-S_c^{Kelvin,src}$",
)
ax.set_title("Source integrated Kelvin")
ax.set_xlabel("Singular value index")
ax.set_ylabel("Singular value")
ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.legend()

plt.show()

print("\n--- Frobenius norms ---")
print("||Sc_FE||F                 =", np.linalg.norm(Sc_FE, "fro"))
print("||Sc_Kelvin_source||F      =", np.linalg.norm(Sc_Kelvin_source, "fro"))
print("||R_source||F              =", np.linalg.norm(R_source, "fro"))

print("\n--- Relative residuals ---")
print(
    "source integrated          =",
    np.linalg.norm(R_source, "fro") / np.linalg.norm(Sc_FE, "fro"),
)

print("\n--- Diagonal min/max ---")
print("FE                         :", np.min(np.diag(Sc_FE)), np.max(np.diag(Sc_FE)))
print(
    "Kelvin source integrated   :",
    np.min(np.diag(Sc_Kelvin_source)),
    np.max(np.diag(Sc_Kelvin_source)),
)
