import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

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

from dolfinx.geometry import bb_tree, create_midpoint_tree, compute_closest_entity
from dolfinx.mesh import entities_to_geometry

from cofebem.contact.Sc_normal import Sc_normal
from cofebem.contact.lcp_solvers.lemke import lemkelcp

import meshio


# ============================================================
# Geometry helpers
# ============================================================


def triangle_plane_projection_barycentric(p, a, b, c, eps=1e-12):

    ab = b - a
    ac = c - a
    ntri = np.cross(ab, ac)
    nn = np.linalg.norm(ntri)
    if nn < eps:
        return None, None, False, None

    ntri = ntri / nn

    # Orthogonal projection onto the plane
    q = p - np.dot(p - a, ntri) * ntri

    # Barycentric coordinates on the plane
    v0 = ab
    v1 = ac
    v2 = q - a

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < eps:
        return q, None, False, ntri

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    bary = np.array([u, v, w], dtype=np.float64)
    inside = np.all(bary >= -eps) and np.all(bary <= 1.0 + eps)

    return q, bary, inside, ntri


def ray_triangle_intersection(
    ray_origin,
    ray_dir,
    a,
    b,
    c,
    eps=1e-12,
):

    e1 = b - a
    e2 = c - a
    pvec = np.cross(ray_dir, e2)
    det = np.dot(e1, pvec)

    if abs(det) < eps:
        return False, np.inf, None, None  # parallel

    inv_det = 1.0 / det
    tvec = ray_origin - a
    u = np.dot(tvec, pvec) * inv_det
    if u < -eps or u > 1.0 + eps:
        return False, np.inf, None, None

    qvec = np.cross(tvec, e1)
    v = np.dot(ray_dir, qvec) * inv_det
    if v < -eps or (u + v) > 1.0 + eps:
        return False, np.inf, None, None

    t = np.dot(e2, qvec) * inv_det
    if t < eps:
        return False, np.inf, None, None

    w0 = 1.0 - u - v
    bary = np.array([w0, u, v], dtype=np.float64)
    xhit = ray_origin + t * ray_dir
    return True, t, bary, xhit


def interpolate_master_normal_from_triangle(
    gnodes,
    bary,
    master_global_to_local,
    master_vertex_normals,
    eps=1e-12,
):

    n_pair = np.zeros(3, dtype=np.float64)
    for a_loc in range(3):
        vg = int(gnodes[a_loc])
        jv = master_global_to_local.get(vg, None)
        if jv is None:
            return None
        n_pair += bary[a_loc] * master_vertex_normals[jv]

    nn = np.linalg.norm(n_pair)
    if nn < eps:
        return None

    return n_pair / nn


def normal_projection_signed_gap(
    slave_points,
    slave_normals,
    target_mesh,
    target_facet_ids,
    master_vertex_ids,
    master_vertex_normals,
    search_data=None,
    outside_value=1.0,
    eps=1e-12,
):

    slave_points = np.asarray(slave_points, dtype=np.float64)
    slave_normals = np.asarray(slave_normals, dtype=np.float64)
    master_vertex_ids = np.asarray(master_vertex_ids, dtype=np.int32)
    master_vertex_normals = np.asarray(master_vertex_normals, dtype=np.float64)

    if search_data is None:
        search_data = build_closest_point_search_data(target_mesh, target_facet_ids)

    tree = search_data["tree"]
    midpoint_tree = search_data["midpoint_tree"]
    facet_geom_dofs = search_data["facet_geom_dofs"]
    facet_to_local = search_data["facet_to_local"]

    master_global_to_local = build_vertex_local_index(master_vertex_ids)
    X = target_mesh.geometry.x
    ns = slave_points.shape[0]

    y = np.zeros((ns, 3), dtype=np.float64)
    facet_pairs = -np.ones(ns, dtype=np.int32)
    bary = np.zeros((ns, 3), dtype=np.float64)
    hit_mask = np.zeros(ns, dtype=bool)
    gap_signed = np.full(ns, outside_value, dtype=np.float64)
    hit_vertices = -np.ones((ns, 3), dtype=np.int32)
    pair_normals = np.zeros((ns, 3), dtype=np.float64)
    dist = np.full(ns, np.inf, dtype=np.float64)

    closest_facets = compute_closest_entity(
        tree, midpoint_tree, target_mesh, slave_points
    )

    for i in range(ns):
        f = int(closest_facets[i])
        if f < 0:
            continue

        jfacet = facet_to_local.get(f, None)
        if jfacet is None:
            continue

        gnodes = np.asarray(facet_geom_dofs[jfacet], dtype=np.int32)
        if gnodes.size < 3:
            continue
        if gnodes.size > 3:
            gnodes = gnodes[:3]

        a, b, c = X[gnodes[0]], X[gnodes[1]], X[gnodes[2]]

        q, w, inside, _ = triangle_plane_projection_barycentric(
            slave_points[i], a, b, c, eps=eps
        )

        if q is None or w is None or not inside:
            # exactly as requested: set to 1 if orthogonal projection is outside
            y[i] = slave_points[i]
            facet_pairs[i] = f
            bary[i] = 0.0
            hit_vertices[i] = gnodes
            gap_signed[i] = outside_value
            dist[i] = np.inf
            continue

        n_pair = interpolate_master_normal_from_triangle(
            gnodes, w, master_global_to_local, master_vertex_normals, eps=eps
        )
        if n_pair is None:
            continue

        ns_i = slave_normals[i]
        nsn = np.linalg.norm(ns_i)
        if nsn > eps:
            ns_i = ns_i / nsn
            if np.dot(n_pair, ns_i) > 0.0:
                n_pair = -n_pair

        # g = (x_s - x_m) . n_m
        g = np.dot(slave_points[i] - q, n_pair)

        y[i] = q
        facet_pairs[i] = f
        bary[i] = w
        hit_mask[i] = True
        gap_signed[i] = g
        hit_vertices[i] = gnodes
        pair_normals[i] = n_pair
        dist[i] = np.linalg.norm(q - slave_points[i])

    return y, facet_pairs, bary, hit_mask, gap_signed, hit_vertices, pair_normals, dist


def ray_casting_signed_gap(
    slave_points,
    slave_normals,
    target_mesh,
    target_facet_ids,
    master_vertex_ids,
    master_vertex_normals,
    eps=1e-12,
):

    slave_points = np.asarray(slave_points, dtype=np.float64)
    slave_normals = np.asarray(slave_normals, dtype=np.float64)
    master_vertex_ids = np.asarray(master_vertex_ids, dtype=np.int32)
    master_vertex_normals = np.asarray(master_vertex_normals, dtype=np.float64)
    target_facet_ids = np.asarray(target_facet_ids, dtype=np.int32)

    fdim = target_mesh.topology.dim - 1
    X = target_mesh.geometry.x
    facet_geom_dofs = np.asarray(
        entities_to_geometry(target_mesh, fdim, target_facet_ids), dtype=np.int32
    )
    master_global_to_local = build_vertex_local_index(master_vertex_ids)

    ns = slave_points.shape[0]

    y = np.zeros((ns, 3), dtype=np.float64)
    facet_pairs = -np.ones(ns, dtype=np.int32)
    bary = np.zeros((ns, 3), dtype=np.float64)
    hit_mask = np.zeros(ns, dtype=bool)
    gap_signed = np.full(ns, np.inf, dtype=np.float64)
    hit_vertices = -np.ones((ns, 3), dtype=np.int32)
    pair_normals = np.zeros((ns, 3), dtype=np.float64)
    dist = np.full(ns, np.inf, dtype=np.float64)

    ray_direction_flag = np.zeros(
        ns, dtype=np.int32
    )  # +1 for +ns, -1 for -ns, 0 no hit

    for i in range(ns):
        xs = slave_points[i]
        ns_i = slave_normals[i].copy()
        nn = np.linalg.norm(ns_i)
        if nn < eps:
            continue
        ns_i /= nn

        best_abs_t = np.inf
        best_signed_t = np.inf
        best_q = None
        best_f = -1
        best_w = None
        best_gnodes = None
        best_npair = None
        best_dir_flag = 0

        for idir, ray_dir in [(+1, ns_i), (-1, -ns_i)]:
            for k, f in enumerate(target_facet_ids):
                gnodes = np.asarray(facet_geom_dofs[k], dtype=np.int32)
                if gnodes.size < 3:
                    continue
                if gnodes.size > 3:
                    gnodes = gnodes[:3]

                a, b, c = X[gnodes[0]], X[gnodes[1]], X[gnodes[2]]

                hit, t, w, q = ray_triangle_intersection(xs, ray_dir, a, b, c, eps=eps)
                if not hit:
                    continue

                n_pair = interpolate_master_normal_from_triangle(
                    gnodes, w, master_global_to_local, master_vertex_normals, eps=eps
                )
                if n_pair is None:
                    continue

                # Keep master/slave normals consistently opposite if possible
                if np.dot(n_pair, ns_i) > 0.0:
                    n_pair = -n_pair

                signed_t = float(idir) * float(t)

                if abs(signed_t) < best_abs_t:
                    best_abs_t = abs(signed_t)
                    best_signed_t = signed_t
                    best_q = q
                    best_f = int(f)
                    best_w = w
                    best_gnodes = gnodes.copy()
                    best_npair = n_pair.copy()
                    best_dir_flag = idir

        if best_q is None:
            continue

        y[i] = best_q
        facet_pairs[i] = best_f
        bary[i] = best_w
        hit_mask[i] = True
        gap_signed[i] = best_signed_t
        hit_vertices[i] = best_gnodes
        pair_normals[i] = best_npair
        dist[i] = abs(best_signed_t)
        ray_direction_flag[i] = best_dir_flag

    return (
        y,
        facet_pairs,
        bary,
        hit_mask,
        gap_signed,
        hit_vertices,
        pair_normals,
        dist,
        ray_direction_flag,
    )


def unique_vertices_from_facets(mesh, facet_ids):
    fdim = mesh.topology.dim - 1
    facet_ids = np.asarray(facet_ids, dtype=np.int32)
    facet_geom = entities_to_geometry(mesh, fdim, facet_ids)
    verts = np.unique(np.asarray(facet_geom, dtype=np.int32).ravel())
    return np.sort(verts).astype(np.int32)


def build_vertex_local_index(vertex_ids):
    return {int(v): k for k, v in enumerate(np.asarray(vertex_ids, dtype=np.int32))}


def project_point_to_triangle(p, a, b, c):
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy(), np.array([1.0, 0.0, 0.0], dtype=np.float64)

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b.copy(), np.array([0.0, 1.0, 0.0], dtype=np.float64)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        q = a + v * ab
        return q, np.array([1.0 - v, v, 0.0], dtype=np.float64)

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c.copy(), np.array([0.0, 0.0, 1.0], dtype=np.float64)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        q = a + w * ac
        return q, np.array([1.0 - w, 0.0, w], dtype=np.float64)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        bc = c - b
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        q = b + w * bc
        return q, np.array([0.0, 1.0 - w, w], dtype=np.float64)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    u = 1.0 - v - w
    q = u * a + v * b + w * c
    return q, np.array([u, v, w], dtype=np.float64)


def build_closest_point_search_data(mesh, facet_ids):
    fdim = mesh.topology.dim - 1
    facet_ids = np.asarray(facet_ids, dtype=np.int32)

    tree = bb_tree(mesh, fdim, entities=facet_ids)
    midpoint_tree = create_midpoint_tree(mesh, fdim, facet_ids)
    facet_geom_dofs = np.asarray(
        entities_to_geometry(mesh, fdim, facet_ids), dtype=np.int32
    )
    facet_to_local = {int(f): i for i, f in enumerate(facet_ids)}

    return {
        "tree": tree,
        "midpoint_tree": midpoint_tree,
        "facet_ids": facet_ids,
        "facet_geom_dofs": facet_geom_dofs,
        "facet_to_local": facet_to_local,
    }


def closest_point_signed_gap(
    slave_points,
    slave_normals,
    target_mesh,
    target_facet_ids,
    master_vertex_ids,
    master_vertex_normals,
    search_data=None,
    max_dist=np.inf,
    normal_tol=None,
    eps=1e-12,
):
    slave_points = np.asarray(slave_points, dtype=np.float64)
    slave_normals = np.asarray(slave_normals, dtype=np.float64)
    master_vertex_ids = np.asarray(master_vertex_ids, dtype=np.int32)
    master_vertex_normals = np.asarray(master_vertex_normals, dtype=np.float64)

    if search_data is None:
        search_data = build_closest_point_search_data(target_mesh, target_facet_ids)

    tree = search_data["tree"]
    midpoint_tree = search_data["midpoint_tree"]
    facet_geom_dofs = search_data["facet_geom_dofs"]
    facet_to_local = search_data["facet_to_local"]

    master_global_to_local = build_vertex_local_index(master_vertex_ids)
    X = target_mesh.geometry.x
    ns = slave_points.shape[0]

    y = np.zeros((ns, 3), dtype=np.float64)
    facet_pairs = -np.ones(ns, dtype=np.int32)
    bary = np.zeros((ns, 3), dtype=np.float64)
    hit_mask = np.zeros(ns, dtype=bool)
    gap_signed = np.full(ns, np.inf, dtype=np.float64)
    hit_vertices = -np.ones((ns, 3), dtype=np.int32)
    pair_normals = np.zeros((ns, 3), dtype=np.float64)
    dist = np.full(ns, np.inf, dtype=np.float64)

    closest_facets = compute_closest_entity(
        tree, midpoint_tree, target_mesh, slave_points
    )

    for i in range(ns):
        f = int(closest_facets[i])
        if f < 0:
            continue

        jfacet = facet_to_local.get(f, None)
        if jfacet is None:
            continue

        gnodes = np.asarray(facet_geom_dofs[jfacet], dtype=np.int32)
        if gnodes.size < 3:
            continue
        if gnodes.size > 3:
            gnodes = gnodes[:3]

        a, b, c = X[gnodes[0]], X[gnodes[1]], X[gnodes[2]]
        q, w = project_point_to_triangle(slave_points[i], a, b, c)

        # Pair normal from master nodal normals interpolated at q
        n_pair = np.zeros(3, dtype=np.float64)
        ok = True
        for a_loc in range(3):
            vg = int(gnodes[a_loc])
            jv = master_global_to_local.get(vg, None)
            if jv is None:
                ok = False
                break
            n_pair += w[a_loc] * master_vertex_normals[jv]

        if not ok:
            continue

        nn = np.linalg.norm(n_pair)
        if nn < eps:
            continue
        n_pair /= nn

        # Orient pair normal consistently against slave normal
        ns_i = slave_normals[i]
        nsn = np.linalg.norm(ns_i)
        if nsn > eps:
            ns_i = ns_i / nsn
            if np.dot(n_pair, ns_i) > 0.0:
                n_pair = -n_pair

            if normal_tol is not None and np.dot(n_pair, ns_i) > -normal_tol:
                continue

        dvec = slave_points[i] - q
        d = np.linalg.norm(dvec)
        if d > max_dist:
            continue

        g = np.dot(dvec, n_pair)

        y[i] = q
        facet_pairs[i] = f
        bary[i] = w
        hit_mask[i] = True
        gap_signed[i] = g
        hit_vertices[i] = gnodes
        pair_normals[i] = n_pair
        dist[i] = d

    return y, facet_pairs, bary, hit_mask, gap_signed, hit_vertices, pair_normals, dist


# ============================================================
# FE system
# ============================================================


def build_contact_normals(
    mesh,
    contact_vertex_ids: np.ndarray,
    ds_contact,
    eps: float = 1e-8,
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


# ============================================================
# Transfer / evaluation operators
# ============================================================


def build_transfer_operators(
    slave_vertex_ids,
    slave_normals,
    master_vertex_ids,
    master_normals,
    pair_normals,
    hit_vertices,
    bary,
    hit_mask,
):
    slave_vertex_ids = np.asarray(slave_vertex_ids, dtype=np.int32)
    master_vertex_ids = np.asarray(master_vertex_ids, dtype=np.int32)
    slave_normals = np.asarray(slave_normals, dtype=np.float64)
    master_normals = np.asarray(master_normals, dtype=np.float64)
    pair_normals = np.asarray(pair_normals, dtype=np.float64)
    hit_vertices = np.asarray(hit_vertices, dtype=np.int32)
    bary = np.asarray(bary, dtype=np.float64)
    hit_mask = np.asarray(hit_mask, dtype=bool)

    ns = slave_vertex_ids.size
    nm = master_vertex_ids.size
    master_global_to_local = build_vertex_local_index(master_vertex_ids)

    # N_ms : slave nodal amplitudes -> master nodal amplitudes
    N_ms = np.zeros((nm, ns), dtype=np.float64)

    # P_m : master nodal normal displacements -> pair-normal displacement
    P_m = np.zeros((ns, nm), dtype=np.float64)

    for i in range(ns):
        if not hit_mask[i]:
            continue

        ns_i = slave_normals[i]
        npair_i = pair_normals[i]
        verts = hit_vertices[i]
        w = bary[i]

        for a in range(3):
            vg = int(verts[a])
            j = master_global_to_local.get(vg, None)
            if j is None:
                raise RuntimeError(
                    f"Projected master vertex {vg} is not in master contact vertex set."
                )

            nm_j = master_normals[j]

            # slave amplitudes -> master nodal amplitudes
            N_ms[j, i] += w[a] * float(np.dot(nm_j, ns_i))

            # master nodal displacements -> pair displacement along pair normal
            P_m[i, j] += w[a] * float(np.dot(npair_i, nm_j))

    return P_m, N_ms


def vertex_to_parent_dofs(V, vertex_ids):
    gdim = V.mesh.geometry.dim
    vertex_ids = np.asarray(vertex_ids, dtype=np.int32)

    comp_dofs = []
    for comp in range(gdim):
        Vc, sub_to_parent = V.sub(comp).collapse()
        sub_to_parent = np.asarray(sub_to_parent, dtype=np.int32)
        dofs_sub = locate_dofs_topological(Vc, 0, vertex_ids)
        dofs_sub = np.asarray(dofs_sub, dtype=np.int32)
        comp_dofs.append(sub_to_parent[dofs_sub])

    return comp_dofs


# ============================================================
# Debug export
# ============================================================


def export_gap_method_vtk(
    filename,
    slave_points,
    projected_points,
    normals_to_plot,
    hit_mask,
    gap_signed,
    method_id=0,
    direction_flag=None,
):
    slave_points = np.asarray(slave_points, dtype=float)
    projected_points = np.asarray(projected_points, dtype=float)
    normals_to_plot = np.asarray(normals_to_plot, dtype=float)
    hit_mask = np.asarray(hit_mask, dtype=bool)
    gap_signed = np.asarray(gap_signed, dtype=float)

    if direction_flag is None:
        direction_flag = np.zeros_like(gap_signed, dtype=np.int32)
    else:
        direction_flag = np.asarray(direction_flag, dtype=np.int32)

    ids = np.arange(slave_points.shape[0], dtype=np.int64)
    n = len(ids)

    points = np.zeros((2 * n, 3), dtype=float)
    normals = np.zeros((2 * n, 3), dtype=float)
    kinds = np.zeros(2 * n, dtype=float)
    point_hit = np.zeros(2 * n, dtype=float)
    lines = np.zeros((n, 2), dtype=np.int64)

    cell_gap = np.zeros(n, dtype=float)
    cell_hit = np.zeros(n, dtype=float)
    cell_sid = np.zeros(n, dtype=float)
    cell_method = np.full(n, float(method_id), dtype=float)
    cell_dir = np.zeros(n, dtype=float)

    for k, i in enumerate(ids):
        p0 = 2 * k
        p1 = 2 * k + 1

        points[p0] = slave_points[i]
        points[p1] = projected_points[i] if hit_mask[i] else slave_points[i]

        normals[p0] = normals_to_plot[i]
        normals[p1] = 0.0

        kinds[p0] = 0.0
        kinds[p1] = 1.0

        point_hit[p0] = float(hit_mask[i])
        point_hit[p1] = float(hit_mask[i])

        lines[k] = [p0, p1]

        cell_gap[k] = gap_signed[i]
        cell_hit[k] = float(hit_mask[i])
        cell_sid[k] = float(i)
        cell_dir[k] = float(direction_flag[i])

    meshio.write_points_cells(
        filename,
        points,
        [("line", lines)],
        point_data={
            "kind": kinds,
            "normal": normals,
            "hit_point": point_hit,
        },
        cell_data={
            "gap_signed": [cell_gap],
            "hit": [cell_hit],
            "slave_id": [cell_sid],
            "method_id": [cell_method],
            "direction_flag": [cell_dir],
        },
    )


# def export_all_projections_vtk(
#     filename,
#     slave_points,
#     projected_points,
#     normals_to_plot,
#     hit_mask,
#     gap_signed,
# ):
#     slave_points = np.asarray(slave_points, dtype=float)
#     projected_points = np.asarray(projected_points, dtype=float)
#     normals_to_plot = np.asarray(normals_to_plot, dtype=float)
#     hit_mask = np.asarray(hit_mask, dtype=bool)
#     gap_signed = np.asarray(gap_signed, dtype=float)

#     ids = np.where(hit_mask)[0]
#     nh = len(ids)

#     if nh == 0:
#         raise RuntimeError("No valid projections to export.")

#     points = np.zeros((2 * nh, 3), dtype=float)
#     normals = np.zeros((2 * nh, 3), dtype=float)
#     kinds = np.zeros(2 * nh, dtype=float)
#     lines = np.zeros((nh, 2), dtype=np.int64)
#     gaps = np.zeros(nh, dtype=float)
#     slave_id = np.zeros(nh, dtype=float)

#     for k, i in enumerate(ids):
#         p0 = 2 * k
#         p1 = 2 * k + 1

#         points[p0] = slave_points[i]
#         points[p1] = projected_points[i]

#         normals[p0] = normals_to_plot[i]
#         normals[p1] = 0.0

#         kinds[p0] = 0.0
#         kinds[p1] = 1.0

#         lines[k] = [p0, p1]
#         gaps[k] = gap_signed[i]
#         slave_id[k] = float(i)

#     meshio.write_points_cells(
#         filename,
#         points,
#         [("line", lines)],
#         point_data={
#             "kind": kinds,
#             "normal": normals,
#         },
#         cell_data={
#             "gap_signed": [gaps],
#             "slave_id": [slave_id],
#         },
#     )


# ============================================================
# Main
# ============================================================

comm = MPI.COMM_WORLD

# ---------------- Meshes ----------------
cube, _, _ = gmshio.read_from_msh("./msh_files/cube_tetra.msh", comm, 0, gdim=3)
hemisphere, _, _ = gmshio.read_from_msh("./msh_files/hemisphere1.msh", comm, 0, gdim=3)

dz = 1.9
hemisphere.geometry.x[:, 0] *= -1.0
hemisphere.geometry.x[:, 2] *= -1.0
hemisphere.geometry.x[:, 2] += float(dz)

hemisphere.geometry.x[:, 0] += 0.5
hemisphere.geometry.x[:, 1] += 0.5

cube_V, cube_A, cube_b, cube_problem = build_system(
    cube,
    E=1.0e9,
    nu=0.3,
    Gamma_u_locator=lambda x: np.isclose(x[2], 0.0),
)

hemi_V, hemi_A, hemi_b, hemi_problem = build_system(
    hemisphere,
    E=1.0e9,
    nu=0.3,
    Gamma_u_locator=lambda x: np.isclose(x[2], dz, atol=1e-8),
)

# ---------------- Contact boundaries ----------------
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

# Hemisphere contact boundary
hemi_tdim = hemisphere.topology.dim
hemi_fdim = hemi_tdim - 1


def hemi_Gamma_c_locator(x):
    return (x[2] > dz - 1.0) & (x[2] < dz - 0.05)


hemi_Gamma_c = locate_entities_boundary(hemisphere, hemi_fdim, hemi_Gamma_c_locator)
hemi_Gamma_c_id = 1
hemi_Gamma_c_tags = np.full(hemi_Gamma_c.shape, hemi_Gamma_c_id, dtype=np.int32)

hemi_order = np.argsort(hemi_Gamma_c)
hemi_Gamma_c_facet_indices = hemi_Gamma_c[hemi_order].astype(np.int32)
hemi_Gamma_c_facet_values = hemi_Gamma_c_tags[hemi_order].astype(np.int32)

hemi_mt = meshtags(
    hemisphere,
    hemi_fdim,
    hemi_Gamma_c_facet_indices,
    hemi_Gamma_c_facet_values,
)
ds_hemi = Measure("ds", domain=hemisphere, subdomain_data=hemi_mt)

hemi_Ic = unique_vertices_from_facets(hemisphere, hemi_Gamma_c)

hemi_normals, hemi_normal_field = build_contact_normals(
    mesh=hemisphere,
    contact_vertex_ids=hemi_Ic,
    ds_contact=ds_hemi(hemi_Gamma_c_id),
    save_field=True,
)

print(f"Cube contact facets: {cube_Gamma_c.size}, contact vertices: {cube_Ic.size}")
print(
    f"Hemisphere contact facets: {hemi_Gamma_c.size}, contact vertices: {hemi_Ic.size}"
)
print(
    f"{hemisphere.geometry.x[hemi_Ic].shape[0]} slave contact vertices, "
    f"{hemi_normals.shape[0]} slave normals"
)

# ---------------- Projection: slave = hemisphere, master = cube ----------------
cube_search = build_closest_point_search_data(cube, cube_Gamma_c)

# ==================  1  ===============
# y, facet_pairs, bary, hit_mask, g0, hit_vertices, pair_normals, dist = (
#     closest_point_signed_gap(
#         slave_points=hemisphere.geometry.x[hemi_Ic],
#         slave_normals=hemi_normals,
#         target_mesh=cube,
#         target_facet_ids=cube_Gamma_c,
#         master_vertex_ids=cube_Ic,
#         master_vertex_normals=cube_normals,
#         search_data=cube_search,
#         max_dist=np.inf,
#         normal_tol=None,
#     )
# )

# print(f"Projected pairs found: {np.count_nonzero(hit_mask)} / {hemi_Ic.size}")
# print("First signed gaps:", g0[:10])

# =================  2  ====================
# y, facet_pairs, bary, hit_mask, g0, hit_vertices, pair_normals, dist = (
#     normal_projection_signed_gap(
#         slave_points=hemisphere.geometry.x[hemi_Ic],
#         slave_normals=hemi_normals,
#         target_mesh=cube,
#         target_facet_ids=cube_Gamma_c,
#         master_vertex_ids=cube_Ic,
#         master_vertex_normals=cube_normals,
#         search_data=cube_search,
#         outside_value=1.0,
#     )
# )

# print(f"Projected pairs found: {np.count_nonzero(hit_mask)} / {hemi_Ic.size}")
# print("First signed gaps:", g0[:10])

# ==================  3 ===================================
y, facet_pairs, bary, hit_mask, g0, hit_vertices, pair_normals, dist, _ = (
    ray_casting_signed_gap(
        slave_points=hemisphere.geometry.x[hemi_Ic],
        slave_normals=hemi_normals,
        target_mesh=cube,
        target_facet_ids=cube_Gamma_c,
        master_vertex_ids=cube_Ic,
        master_vertex_normals=cube_normals,
    )
)

print(f"Projected pairs found: {np.count_nonzero(hit_mask)} / {hemi_Ic.size}")
print("First signed gaps:", g0[:10])

# ---------------- Compliance matrices ----------------
cube_Sc = Sc_normal(cube_A, cube_b, cube_Ic, cube_normals)
hemi_Sc = Sc_normal(hemi_A, hemi_b, hemi_Ic, hemi_normals)

Sm = cube_Sc.sample_n(show=True)  # (nm, nm)
Ss = hemi_Sc.sample_n(show=True)  # (ns, ns)

# ---------------- Transfer matrices ----------------
P_m, N_ms = build_transfer_operators(
    slave_vertex_ids=hemi_Ic,
    slave_normals=hemi_normals,
    master_vertex_ids=cube_Ic,
    master_normals=cube_normals,
    pair_normals=pair_normals,
    hit_vertices=hit_vertices,
    bary=bary,
    hit_mask=hit_mask,
)

print("Shapes:")
print("  Ss  :", Ss.shape)
print("  Sm  :", Sm.shape)
print("  P_m :", P_m.shape)
print("  N_ms:", N_ms.shape)

# ---------------- Reduced operator and gap ----------------
Wm = P_m @ Sm @ N_ms
W = Ss - Wm

cube_u = Function(cube_V)
cube_u.name = "um"
cube_p = Function(cube_V)
cube_p.name = "pm"

hemi_u = Function(hemi_V)
hemi_u.name = "us"
hemi_p = Function(hemi_V)
hemi_p.name = "ps"

cube_contact_dofs = vertex_to_parent_dofs(cube_V, cube_Ic)
hemi_contact_dofs = vertex_to_parent_dofs(hemi_V, hemi_Ic)

with VTKFile(cube.comm, "./results/bilateral/cube.pvd", "w") as vtk1, VTKFile(
    hemisphere.comm, "./results/bilateral/hemi.pvd", "w"
) as vtk2:

    vtk1.write_function([cube_u, cube_p], 0)
    vtk2.write_function([hemi_u, hemi_p], 0)

    ps, exit_code, _ = lemkelcp(W, g0, maxIter=10000)
    print("Lemke exit code:", exit_code)

    ps = np.asarray(ps, dtype=np.float64)
    pm = -N_ms @ ps

    # ---------------- Cube visualization ----------------
    solver_petsc_cube = PETSc.KSP().create(cube.comm)
    solver_petsc_cube.setOperators(cube_A)
    solver_petsc_cube.setType("preonly")
    solver_petsc_cube.getPC().setType("lu")
    solver_petsc_cube.setFromOptions()
    solver_petsc_cube.setUp()

    cube_b_ = cube_b.copy()
    cube_u_ = PETSc.Vec().createMPI(cube_b_.getSize(), comm=cube.comm)

    cube_b_.set(0)
    for i in range(len(cube_Ic)):
        cube_vals = -pm[i] * cube_normals[i]
        cube_b_.setValue(
            cube_contact_dofs[0][i], cube_vals[0], addv=PETSc.InsertMode.ADD_VALUES
        )
        cube_b_.setValue(
            cube_contact_dofs[1][i], cube_vals[1], addv=PETSc.InsertMode.ADD_VALUES
        )
        cube_b_.setValue(
            cube_contact_dofs[2][i], cube_vals[2], addv=PETSc.InsertMode.ADD_VALUES
        )

    cube_b_.assemble()
    solver_petsc_cube.solve(cube_b_, cube_u_)

    cube_u.x.array[:] = cube_u_.array
    cube_u.x.scatter_forward()

    cube_p.x.array[:] = cube_b_.array
    cube_p.x.scatter_forward()

    # ---------------- Hemisphere visualization ----------------
    solver_petsc_hemi = PETSc.KSP().create(hemisphere.comm)
    solver_petsc_hemi.setOperators(hemi_A)
    solver_petsc_hemi.setType("preonly")
    solver_petsc_hemi.getPC().setType("lu")
    solver_petsc_hemi.setFromOptions()
    solver_petsc_hemi.setUp()

    hemi_b_ = hemi_b.copy()
    hemi_u_ = PETSc.Vec().createMPI(hemi_b_.getSize(), comm=hemisphere.comm)

    hemi_b_.set(0)
    for i in range(len(hemi_Ic)):
        hemi_vals = -ps[i] * hemi_normals[i]
        hemi_b_.setValue(
            hemi_contact_dofs[0][i], hemi_vals[0], addv=PETSc.InsertMode.ADD_VALUES
        )
        hemi_b_.setValue(
            hemi_contact_dofs[1][i], hemi_vals[1], addv=PETSc.InsertMode.ADD_VALUES
        )
        hemi_b_.setValue(
            hemi_contact_dofs[2][i], hemi_vals[2], addv=PETSc.InsertMode.ADD_VALUES
        )

    hemi_b_.assemble()
    solver_petsc_hemi.solve(hemi_b_, hemi_u_)

    hemi_u.x.array[:] = hemi_u_.array
    hemi_u.x.scatter_forward()

    hemi_p.x.array[:] = hemi_b_.array
    hemi_p.x.scatter_forward()

    vtk1.write_function([cube_u, cube_p], 1)
    vtk2.write_function([hemi_u, hemi_p], 1)

# ---------------- Debug export ----------------

export_gap_method_vtk(
    "./results/bilateral/all_projections.vtu",
    hemisphere.geometry.x[hemi_Ic],
    hit_vertices,
    pair_normals,
    hit_mask,
    g0,
    method_id=1,
    direction_flag=None,
)

# export_all_projections_vtk(
#     "./results/bilateral/all_projections.vtu",
#     slave_points=hemisphere.geometry.x[hemi_Ic],
#     projected_points=y,
#     normals_to_plot=pair_normals,
#     hit_mask=hit_mask,
#     gap_signed=g0,
# )

print("Done.")
