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


def _double_integral_regular(
    tri_t,
    corner_t,
    n_target,
    tri_s,
    corner_s,
    n_source,
    E,
    nu,
    nq_t,
    nq_s,
):
    xq_t, wq_t = _gauss_legendre_01(nq_t)
    xq_s, wq_s = _gauss_legendre_01(nq_s)

    Jt = _triangle_jacobian(tri_t)
    Js = _triangle_jacobian(tri_s)

    val = 0.0
    for a, ut in enumerate(xq_t):
        wut = wq_t[a]
        for b, vt in enumerate(xq_t):
            wvt = wq_t[b]
            xi1_t, xi2_t, Jrt = _regular_triangle_map(ut, vt)
            x = _map_to_triangle(tri_t, xi1_t, xi2_t)
            phi_t = _p1_ref(corner_t, xi1_t, xi2_t)

            for c, us in enumerate(xq_s):
                wus = wq_s[c]
                for d, vs in enumerate(xq_s):
                    wvs = wq_s[d]
                    xi1_s, xi2_s, Jrs = _regular_triangle_map(us, vs)
                    y = _map_to_triangle(tri_s, xi1_s, xi2_s)
                    phi_s = _p1_ref(corner_s, xi1_s, xi2_s)

                    rvec = x - y
                    if np.linalg.norm(rvec) < 1e-14:
                        continue
                    U = _kelvin_tensor_3d(rvec, E, nu)
                    val += (
                        (n_target @ U @ n_source)
                        * phi_t
                        * phi_s
                        * Jt
                        * Js
                        * Jrt
                        * Jrs
                        * wut
                        * wvt
                        * wus
                        * wvs
                    )

    return val


def _double_integral_self_duffy(
    tri,
    corner,
    n_target,
    n_source,
    E,
    nu,
    nq_t,
    nq_s,
):
    xq_t, wq_t = _gauss_legendre_01(nq_t)
    xq_s, wq_s = _gauss_legendre_01(nq_s)

    Jtri = _triangle_jacobian(tri)
    val = 0.0

    for a, ut in enumerate(xq_t):
        wut = wq_t[a]
        for b, vt in enumerate(xq_t):
            wvt = wq_t[b]
            xi1_t, xi2_t, Jdt = duffy_map_corner(corner, ut, vt)
            x = _map_to_triangle(tri, xi1_t, xi2_t)
            phi_t = _p1_ref(corner, xi1_t, xi2_t)

            for c, us in enumerate(xq_s):
                wus = wq_s[c]
                for d, vs in enumerate(xq_s):
                    wvs = wq_s[d]
                    xi1_s, xi2_s, Jds = duffy_map_corner(corner, us, vs)
                    y = _map_to_triangle(tri, xi1_s, xi2_s)
                    phi_s = _p1_ref(corner, xi1_s, xi2_s)

                    rvec = x - y
                    if np.linalg.norm(rvec) < 1e-14:
                        continue

                    U = _kelvin_tensor_3d(rvec, E, nu)
                    val += (
                        (n_target @ U @ n_source)
                        * phi_t
                        * phi_s
                        * Jtri
                        * Jtri
                        * Jdt
                        * Jds
                        * wut
                        * wvt
                        * wus
                        * wvs
                    )
    return val


def Sc_Kelvin_pointwise(xc, normals, E, nu, self_radii=None, diag_scale=1.0, show=True):
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

    if self_radii is None:
        rad = np.zeros(nc, dtype=float)
        for i in range(nc):
            d = np.linalg.norm(xc - xc[i], axis=1)
            d[i] = np.inf
            rad[i] = np.min(d)
    elif np.isscalar(self_radii):
        rad = float(self_radii) * np.ones(nc, dtype=float)
    else:
        rad = np.asarray(self_radii, dtype=float).reshape(nc)

    rad = diag_scale * rad

    Sc = np.zeros((nc, nc), dtype=float)

    it = range(nc)
    if show:
        it = tqdm(it, desc="Building pointwise Kelvin Snn", unit="row")

    for i in it:
        xi = xc[i]
        ni = normals[i]
        for j in range(nc):
            nj = normals[j]
            if i == j:
                rvec = rad[i] * ni
            else:
                rvec = xi - xc[j]
            U = _kelvin_tensor_3d(rvec, E, nu)
            Sc[i, j] = ni @ U @ nj

    return Sc


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
        xj = xc[j]
        nj = normals[j]
        Aj = areas[j]

        for tri_s, corner_s in inc[j]:
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


def Sc_Kelvin_double_integrated(
    mesh,
    facet_ids,
    contact_vertex_ids,
    normals,
    E,
    nu,
    nq=4,
    nq_self=6,
    show=True,
):
    contact_vertex_ids = np.asarray(contact_vertex_ids, dtype=np.int32)
    normals = np.asarray(normals, dtype=float)

    nc, tdim = normals.shape
    if tdim != 3:
        raise NotImplementedError("Only 3D is implemented.")

    nrm = np.linalg.norm(normals, axis=1)
    if np.any(nrm == 0.0):
        raise ValueError("Some normals have zero norm.")
    normals = normals / nrm[:, None]

    areas = nodal_areas_from_facets(mesh, facet_ids, contact_vertex_ids)
    inc = incident_triangles_for_vertices(mesh, facet_ids, contact_vertex_ids)

    Sc = np.zeros((nc, nc), dtype=float)

    it = range(nc)
    if show:
        it = tqdm(it, desc="Building double integrated Kelvin Snn", unit="col")

    for j in it:
        nj = normals[j]
        Aj = areas[j]

        for i in range(nc):
            ni = normals[i]
            Ai = areas[i]
            val_ij = 0.0

            for tri_t, corner_t in inc[i]:
                for tri_s, corner_s in inc[j]:
                    same_tri = np.allclose(tri_t, tri_s)
                    same_corner = corner_t == corner_s

                    if i == j and same_tri and same_corner:
                        val = _double_integral_self_duffy(
                            tri=tri_t,
                            corner=corner_t,
                            n_target=ni,
                            n_source=nj,
                            E=E,
                            nu=nu,
                            nq_t=nq_self,
                            nq_s=nq_self,
                        )
                    else:
                        val = _double_integral_regular(
                            tri_t=tri_t,
                            corner_t=corner_t,
                            n_target=ni,
                            tri_s=tri_s,
                            corner_s=corner_s,
                            n_source=nj,
                            E=E,
                            nu=nu,
                            nq_t=nq,
                            nq_s=nq,
                        )

                    val_ij += val

            Sc[i, j] = val_ij / (Ai * Aj)

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

Sc_FE = Sc_n(cube_A, cube_Ic, cube_normals, cube_tdim, show=True)

xc = cube.geometry.x[cube_Ic]

Sc_Kelvin_point = Sc_Kelvin_pointwise(
    xc=xc,
    normals=cube_normals,
    E=1e9,
    nu=0.3,
    diag_scale=1.0,
    show=True,
)

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
    show=True,
)

Sc_Kelvin_double = Sc_Kelvin_double_integrated(
    mesh=cube,
    facet_ids=cube_Gamma_c,
    contact_vertex_ids=cube_Ic,
    normals=cube_normals,
    E=1e9,
    nu=0.3,
    nq=4,
    nq_self=6,
    show=True,
)

R_point = Sc_FE - Sc_Kelvin_point
R_source = Sc_FE - Sc_Kelvin_source
R_double = Sc_FE - Sc_Kelvin_double

_, s_FE, _ = np.linalg.svd(Sc_FE)
_, s_Kelvin_point, _ = np.linalg.svd(Sc_Kelvin_point)
_, s_Kelvin_source, _ = np.linalg.svd(Sc_Kelvin_source)
_, s_Kelvin_double, _ = np.linalg.svd(Sc_Kelvin_double)
_, s_R_point, _ = np.linalg.svd(R_point)
_, s_R_source, _ = np.linalg.svd(R_source)
_, s_R_double, _ = np.linalg.svd(R_double)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

axes[0].semilogy(
    np.arange(1, len(s_FE) + 1),
    s_FE,
    marker="o",
    markersize=4,
    linewidth=1.8,
    label=r"$S_c^{FE}$",
)
axes[0].semilogy(
    np.arange(1, len(s_Kelvin_point) + 1),
    s_Kelvin_point,
    marker="s",
    markersize=4,
    linewidth=1.8,
    label=r"$S_c^{Kelvin,pt}$",
)
axes[0].semilogy(
    np.arange(1, len(s_R_point) + 1),
    s_R_point,
    marker="^",
    markersize=4,
    linewidth=1.8,
    label=r"$S_c^{FE}-S_c^{Kelvin,pt}$",
)
axes[0].set_title("Pointwise Kelvin")
axes[0].set_xlabel("Singular value index")
axes[0].set_ylabel("Singular value")
axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
axes[0].legend()

axes[1].semilogy(
    np.arange(1, len(s_FE) + 1),
    s_FE,
    marker="o",
    markersize=4,
    linewidth=1.8,
    label=r"$S_c^{FE}$",
)
axes[1].semilogy(
    np.arange(1, len(s_Kelvin_source) + 1),
    s_Kelvin_source,
    marker="s",
    markersize=4,
    linewidth=1.8,
    label=r"$S_c^{Kelvin,src}$",
)
axes[1].semilogy(
    np.arange(1, len(s_R_source) + 1),
    s_R_source,
    marker="^",
    markersize=4,
    linewidth=1.8,
    label=r"$S_c^{FE}-S_c^{Kelvin,src}$",
)
axes[1].set_title("Source integrated Kelvin")
axes[1].set_xlabel("Singular value index")
axes[1].set_ylabel("Singular value")
axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
axes[1].legend()

axes[2].semilogy(
    np.arange(1, len(s_FE) + 1),
    s_FE,
    marker="o",
    markersize=4,
    linewidth=1.8,
    label=r"$S_c^{FE}$",
)
axes[2].semilogy(
    np.arange(1, len(s_Kelvin_double) + 1),
    s_Kelvin_double,
    marker="s",
    markersize=4,
    linewidth=1.8,
    label=r"$S_c^{Kelvin,double}$",
)
axes[2].semilogy(
    np.arange(1, len(s_R_double) + 1),
    s_R_double,
    marker="^",
    markersize=4,
    linewidth=1.8,
    label=r"$S_c^{FE}-S_c^{Kelvin,double}$",
)
axes[2].set_title("Double integrated Kelvin")
axes[2].set_xlabel("Singular value index")
axes[2].set_ylabel("Singular value")
axes[2].grid(True, which="both", linestyle="--", alpha=0.5)
axes[2].legend()

plt.show()

print("\n--- Frobenius norms ---")
print("||Sc_FE||F                 =", np.linalg.norm(Sc_FE, "fro"))
print("||Sc_Kelvin_point||F       =", np.linalg.norm(Sc_Kelvin_point, "fro"))
print("||Sc_Kelvin_source||F      =", np.linalg.norm(Sc_Kelvin_source, "fro"))
print("||Sc_Kelvin_double||F      =", np.linalg.norm(Sc_Kelvin_double, "fro"))
print("||R_point||F               =", np.linalg.norm(R_point, "fro"))
print("||R_source||F              =", np.linalg.norm(R_source, "fro"))
print("||R_double||F              =", np.linalg.norm(R_double, "fro"))

print("\n--- Relative residuals ---")
print(
    "pointwise                  =",
    np.linalg.norm(R_point, "fro") / np.linalg.norm(Sc_FE, "fro"),
)
print(
    "source integrated          =",
    np.linalg.norm(R_source, "fro") / np.linalg.norm(Sc_FE, "fro"),
)
print(
    "double integrated          =",
    np.linalg.norm(R_double, "fro") / np.linalg.norm(Sc_FE, "fro"),
)

print("\n--- Diagonal min/max ---")
print("FE                         :", np.min(np.diag(Sc_FE)), np.max(np.diag(Sc_FE)))
print(
    "Kelvin pointwise           :",
    np.min(np.diag(Sc_Kelvin_point)),
    np.max(np.diag(Sc_Kelvin_point)),
)
print(
    "Kelvin source integrated   :",
    np.min(np.diag(Sc_Kelvin_source)),
    np.max(np.diag(Sc_Kelvin_source)),
)
print(
    "Kelvin double integrated   :",
    np.min(np.diag(Sc_Kelvin_double)),
    np.max(np.diag(Sc_Kelvin_double)),
)
