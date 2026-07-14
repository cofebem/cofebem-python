from pathlib import Path

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
from dolfinx.mesh import entities_to_geometry
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

from cofebem.contact.lcp_solvers.lemke import lemkelcp
from cofebem.contact.lcp_solvers.psor import psor_lcp

import matplotlib.pyplot as plt

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def Sc_n(A, Ic, nrm, tdim=3, f0=1e9, show=True):
    comm = MPI.COMM_WORLD
    f0 = float(f0)
    tdim = int(tdim)
    Ic = np.asarray(Ic, dtype=np.int64)
    nrm = np.asarray(nrm, dtype=float)
    nc = int(Ic.size)

    m = np.linalg.norm(nrm, axis=1)
    n = nrm / m[:, None]

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.setFromOptions()
    ksp.setUp()

    rhs = A.createVecRight()
    u = A.createVecRight()

    cd = np.stack([Ic * tdim + c for c in range(tdim)], axis=1).astype(np.int32)
    cdf = cd.reshape(-1).astype(np.int32)

    uc = PETSc.Vec().createSeq(tdim * nc, comm=PETSc.COMM_SELF)
    isf = PETSc.IS().createGeneral(cdf, comm=comm)
    ist = PETSc.IS().createStride(tdim * nc, first=0, step=1, comm=PETSc.COMM_SELF)
    sca = PETSc.Scatter().create(u, isf, uc, ist)

    S = np.zeros((nc, nc), dtype=float)

    it = range(nc)
    if show:
        it = tqdm(it, desc="Sampling exact Sc", unit="col")

    for j in it:
        rhs.set(0.0)
        rhs.setValues(cd[j], f0 * n[j], addv=PETSc.InsertMode.INSERT_VALUES)
        rhs.assemble()
        ksp.solve(rhs, u)
        uc.set(0.0)
        sca.scatter(
            u, uc, addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
        )
        ux = uc.getArray(readonly=True).copy().reshape(nc, tdim)
        S[:, j] = np.einsum("ij,ij->i", n, ux) / f0

    return S


def Sc_cols(A, Ic, nrm, sid, tdim=3, f0=1e9, show=True):
    comm = MPI.COMM_WORLD
    f0 = float(f0)
    tdim = int(tdim)
    Ic = np.asarray(Ic, dtype=np.int64)
    nrm = np.asarray(nrm, dtype=float)
    sid = np.asarray(sid, dtype=int)
    nc = int(Ic.size)

    m = np.linalg.norm(nrm, axis=1)
    n = nrm / m[:, None]

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.setFromOptions()
    ksp.setUp()

    rhs = A.createVecRight()
    u = A.createVecRight()

    cd = np.stack([Ic * tdim + c for c in range(tdim)], axis=1).astype(np.int32)
    cdf = cd.reshape(-1).astype(np.int32)

    uc = PETSc.Vec().createSeq(tdim * nc, comm=PETSc.COMM_SELF)
    isf = PETSc.IS().createGeneral(cdf, comm=comm)
    ist = PETSc.IS().createStride(tdim * nc, first=0, step=1, comm=PETSc.COMM_SELF)
    sca = PETSc.Scatter().create(u, isf, uc, ist)

    C = np.zeros((nc, sid.size), dtype=float)

    it = range(sid.size)
    if show:
        it = tqdm(it, desc="Sampling shifted templates", unit="col")

    for a in it:
        j = int(sid[a])
        rhs.set(0.0)
        rhs.setValues(cd[j], f0 * n[j], addv=PETSc.InsertMode.INSERT_VALUES)
        rhs.assemble()
        ksp.solve(rhs, u)
        uc.set(0.0)
        sca.scatter(
            u, uc, addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
        )
        ux = uc.getArray(readonly=True).copy().reshape(nc, tdim)
        C[:, a] = np.einsum("ij,ij->i", n, ux) / f0

    return C


def uniq_v(mesh, fs):
    fdim = mesh.topology.dim - 1
    fs = np.asarray(fs, dtype=np.int32)
    fg = entities_to_geometry(mesh, fdim, fs)
    v = np.unique(np.asarray(fg, dtype=np.int32).ravel())
    return np.sort(v).astype(np.int32)


def nrm_bnd(mesh, Iv, ds_c, eps=1e-8, save=False):
    gdim = mesh.geometry.dim
    Vn = functionspace(mesh, ("CG", 1, (gdim,)))

    n = FacetNormal(mesh)
    u = TrialFunction(Vn)
    v = TestFunction(Vn)

    a = eps * inner(u, v) * dx + inner(u, v) * ds_c
    L = inner(n, v) * ds_c

    nf = Function(Vn)
    nf.name = "n"

    LinearProblem(
        a=a, L=L, u=nf, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    ).solve()
    nf.x.scatter_forward()

    if save:
        with VTKFile(mesh.comm, "n.pvd", "w") as vtk:
            vtk.write_function(nf)

    Iv = np.asarray(Iv, dtype=np.int32)
    nrm = np.zeros((Iv.size, gdim), dtype=float)

    for c in range(gdim):
        Vc, mp = Vn.sub(c).collapse()
        mp = np.asarray(mp, dtype=np.int32)
        ds = locate_dofs_topological(Vc, 0, Iv)
        ds = np.asarray(ds, dtype=np.int32)
        pd = mp[ds]
        nrm[:, c] = nf.x.array[pd]

    m = np.linalg.norm(nrm, axis=1)
    nrm /= m[:, None]
    return nrm, nf


def sys(mesh, E=1.0e9, nu=0.3, gu=None):
    tdim = mesh.topology.dim
    fdim = tdim - 1

    la = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    V = functionspace(mesh, ("Lagrange", 1, (tdim,)))
    u = TrialFunction(V)
    v = TestFunction(V)

    def eps(w):
        return sym(grad(w))

    def sig(w):
        return la * tr(eps(w)) * Identity(tdim) + 2 * mu * eps(w)

    f = Constant(mesh, np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType))
    a = inner(sig(u), eps(v)) * dx
    L = inner(f, v) * dx

    Gu = locate_entities_boundary(mesh, fdim, gu)
    Gd = locate_dofs_topological(V, fdim, Gu)
    u0 = np.array([0, 0, 0], dtype=PETSc.ScalarType)
    bc = dirichletbc(u0, Gd, V)
    bcs = [bc]

    pb = LinearProblem(
        a=a, L=L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )

    pb._A.zeroEntries()
    assemble_matrix_mat(pb._A, pb._a, bcs=pb.bcs)
    pb._A.assemble()

    with pb._b.localForm() as b0:
        b0.set(0)
    assemble_vector(pb._b, pb._L)
    apply_lifting(pb._b, [pb._a], bcs=[pb.bcs])
    pb._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    for bc in pb.bcs:
        bc.set(pb._b.array_w)

    return V, pb._A, pb._b, pb


def surf2(x):
    x = np.asarray(x, dtype=float)
    x0 = np.mean(x, axis=0)
    _, _, vt = np.linalg.svd(x - x0, full_matrices=False)
    return (x - x0) @ vt[:2].T


def dmat2(x):
    x = np.asarray(x, dtype=float)
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)


def anchors(y):
    y = np.asarray(y, dtype=float)
    mn = np.min(y, axis=0)
    mx = np.max(y, axis=0)
    c = 0.5 * (mn + mx)

    pts = [
        c,
        np.array([mn[0], mn[1]]),
        np.array([mn[0], mx[1]]),
        np.array([mx[0], mn[1]]),
        np.array([mx[0], mx[1]]),
        np.array([c[0], mn[1]]),
        np.array([c[0], mx[1]]),
        np.array([mn[0], c[1]]),
        np.array([mx[0], c[1]]),
    ]

    out = []
    for p in pts:
        j = int(np.argmin(np.linalg.norm(y - p, axis=1)))
        if j not in out:
            out.append(j)

    return np.array(out, dtype=int)


def fps_id(y, m, fixed=None):
    y = np.asarray(y, dtype=float)
    n = y.shape[0]

    if m >= n:
        return np.arange(n, dtype=int)

    sid = []

    if fixed is not None:
        for j in fixed:
            j = int(j)
            if 0 <= j < n and j not in sid:
                sid.append(j)
                if len(sid) == m:
                    return np.array(sid, dtype=int)

    D = dmat2(y)

    if len(sid) == 0:
        c = np.mean(y, axis=0)
        sid.append(int(np.argmin(np.linalg.norm(y - c, axis=1))))

    md = np.min(D[np.array(sid)], axis=0)

    while len(sid) < m:
        j = int(np.argmax(md))
        if j in sid:
            break
        sid.append(j)
        md = np.minimum(md, D[j])

    return np.array(sid, dtype=int)


def pick_sid(x, ns, mode="mixed"):
    y = surf2(x)

    if mode == "fps":
        return fps_id(y, ns)

    if mode == "mixed":
        return fps_id(y, ns, fixed=anchors(y))

    if mode == "center":
        c = np.mean(y, axis=0)
        j = int(np.argmin(np.linalg.norm(y - c, axis=1)))
        return fps_id(y, ns, fixed=[j])

    return fps_id(y, ns)


def qknn(y, q, k, tree=None):
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    k = min(int(k), y.shape[0])

    if q.ndim == 1:
        q = q[None, :]

    if tree is not None:
        d, ind = tree.query(q, k=k)
        if k == 1:
            d = d[:, None]
            ind = ind[:, None]
        return np.asarray(d, dtype=float), np.asarray(ind, dtype=int)

    D = np.linalg.norm(q[:, None, :] - y[None, :, :], axis=2)
    ind = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    d = np.take_along_axis(D, ind, axis=1)
    o = np.argsort(d, axis=1)
    ind = np.take_along_axis(ind, o, axis=1)
    d = np.take_along_axis(d, o, axis=1)

    return d, ind


def idw(y, v, q, k=8, p=3.0, tree=None, eps=1e-14):
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    q = np.asarray(q, dtype=float)

    one = False
    if q.ndim == 1:
        one = True
        q = q[None, :]

    d, ind = qknn(y, q, k, tree=tree)

    out = np.zeros(q.shape[0], dtype=float)

    hit = d[:, 0] < eps
    if np.any(hit):
        out[hit] = v[ind[hit, 0]]

    rem = ~hit
    if np.any(rem):
        dr = d[rem]
        ir = ind[rem]
        w = 1.0 / (dr**p + eps)
        w /= np.sum(w, axis=1)[:, None]
        out[rem] = np.sum(w * v[ir], axis=1)

    if one:
        return float(out[0])

    return out


def near_samp(y, sid, j, ps=4, q=2.0, eps=1e-14):
    sid = np.asarray(sid, dtype=int)
    r = np.linalg.norm(y[sid] - y[int(j)], axis=1)
    o = np.argsort(r)
    m = min(int(ps), sid.size)
    nb = sid[o[:m]]
    rr = r[o[:m]]

    if rr[0] < eps:
        return nb[:1], np.array([1.0], dtype=float), rr[:1]

    w = 1.0 / (rr**q + eps)
    w /= np.sum(w)

    return nb, w, rr


def shcol(y, c, s, j, kint=8, pint=3.0, tree=None):
    q = y - y[int(j)] + y[int(s)]
    return idw(y, c, q, k=kint, p=pint, tree=tree)


def psd_fix(A, rtol=1e-12):
    B = 0.5 * (A + A.T)
    ev, Q = np.linalg.eigh(B)
    mx = max(float(np.max(np.abs(ev))), 1.0)
    ev = np.maximum(ev, rtol * mx)
    return (Q * ev) @ Q.T


def S_shift_cols(
    C, sid, x, ps=4, qs=2.0, kint=8, pint=3.0, sym=True, psd=False, show=True
):
    x = np.asarray(x, dtype=float)
    sid = np.asarray(sid, dtype=int)
    C = np.asarray(C, dtype=float)

    n = x.shape[0]
    y = surf2(x)
    tree = cKDTree(y) if cKDTree is not None else None

    A = np.zeros((n, n), dtype=float)
    A[:, sid] = C

    pos = {int(sid[a]): a for a in range(sid.size)}
    ss = set(int(a) for a in sid)
    miss = [j for j in range(n) if j not in ss]

    it = miss
    if show:
        it = tqdm(miss, desc="Building shifted Sc", unit="col")

    info = {}

    for j in it:
        nb, w, rr = near_samp(y, sid, j, ps=ps, q=qs)
        c = np.zeros(n, dtype=float)

        for a, wa in zip(nb, w):
            a = int(a)
            c += wa * shcol(y, C[:, pos[a]], a, j, kint=kint, pint=pint, tree=tree)

        A[:, j] = c

        info[int(j)] = {
            "nb": nb.copy(),
            "w": w.copy(),
            "r": rr.copy(),
        }

    for j in sid:
        info[int(j)] = {
            "nb": np.array([j], dtype=int),
            "w": np.array([1.0], dtype=float),
            "r": np.array([0.0], dtype=float),
        }

    if sym:
        A = 0.5 * (A + A.T)

    if psd:
        A = psd_fix(A)

    par = {
        "ps": ps,
        "qs": qs,
        "kint": kint,
        "pint": pint,
        "sym": sym,
        "psd": psd,
    }

    return A, info, par


def S_shift(
    S0,
    x,
    ns=20,
    sid=None,
    mode="mixed",
    ps=4,
    qs=2.0,
    kint=8,
    pint=3.0,
    sym=True,
    psd=False,
    show=True,
):
    S0 = np.asarray(S0, dtype=float)
    x = np.asarray(x, dtype=float)

    if sid is None:
        sid = pick_sid(x, ns, mode=mode)
    else:
        sid = np.asarray(sid, dtype=int)

    C = S0[:, sid].copy()

    A, info, par = S_shift_cols(
        C=C,
        sid=sid,
        x=x,
        ps=ps,
        qs=qs,
        kint=kint,
        pint=pint,
        sym=sym,
        psd=psd,
        show=show,
    )

    par["ns"] = int(sid.size)
    par["mode"] = mode

    return A, sid, info, par, x[sid]


def S_shift_fem(
    A,
    Ic,
    nrm,
    x,
    ns=20,
    sid=None,
    mode="mixed",
    tdim=3,
    f0=1e9,
    ps=4,
    qs=2.0,
    kint=8,
    pint=3.0,
    sym=True,
    psd=False,
    show=True,
):
    x = np.asarray(x, dtype=float)

    if sid is None:
        sid = pick_sid(x, ns, mode=mode)
    else:
        sid = np.asarray(sid, dtype=int)

    C = Sc_cols(A, Ic, nrm, sid, tdim=tdim, f0=f0, show=show)

    Sa, info, par = S_shift_cols(
        C=C,
        sid=sid,
        x=x,
        ps=ps,
        qs=qs,
        kint=kint,
        pint=pint,
        sym=sym,
        psd=psd,
        show=show,
    )

    par["ns"] = int(sid.size)
    par["mode"] = mode

    return Sa, sid, info, par, x[sid], C


def hlen_from_x(x):
    D = dmat2(x)
    E = D.copy()
    np.fill_diagonal(E, np.inf)
    return float(np.median(np.min(E, axis=1)))


def sphere_gap(x, center_xy, radius, delta):
    x = np.asarray(x, dtype=float)
    center_xy = np.asarray(center_xy, dtype=float)
    z_top = float(np.max(x[:, 2]))
    z0 = z_top + radius - delta

    r2 = np.sum((x[:, :2] - center_xy[None, :]) ** 2, axis=1)
    gap = np.full(x.shape[0], 10.0 * radius, dtype=float)

    inside = r2 <= radius**2
    gap[inside] = z0 - np.sqrt(radius**2 - r2[inside]) - x[inside, 2]
    return gap


def contact_cases(x):
    xy_min = np.min(x[:, :2], axis=0)
    xy_max = np.max(x[:, :2], axis=0)
    span = xy_max - xy_min
    width = float(np.min(span))
    h = hlen_from_x(x[:, :2])
    radius = max(3.0 * h, 0.25 * width)
    delta = min(max(0.75 * h, 0.05 * width), 0.35 * radius)

    return [
        {
            "name": "center",
            "center_xy": xy_min + 0.5 * span,
            "radius": radius,
            "delta": delta,
        },
        {
            "name": "corner",
            "center_xy": xy_min + 0.18 * span,
            "radius": radius,
            "delta": delta,
        },
    ]


def solve_contact_fc(S, gap, max_iter=5000, tol=1e-10):
    S = np.asarray(S, dtype=float)
    gap = np.asarray(gap, dtype=float)

    fc, code, msg = lemkelcp(S, gap, maxIter=max_iter)
    if fc is not None and code == 0:
        w = S @ fc + gap
        return fc, w, {"solver": "lemke", "code": code, "msg": msg}

    fc, w, its, history = psor_lcp(S, gap, omega=1.1, tol=tol, max_iter=max_iter)
    return (
        fc,
        w,
        {
            "solver": "psor",
            "code": code,
            "msg": msg,
            "iterations": its,
            "residual": history[-1] if history else np.nan,
        },
    )


def set_contact_field(fn, Ic, nrm, values, tdim, absolute=False):
    values = np.asarray(values, dtype=float)
    if absolute:
        values = np.abs(values)

    fn.x.array[:] = 0.0
    for c in range(tdim):
        fn.x.array[tdim * Ic + c] = values * nrm[:, c]
    fn.x.scatter_forward()


def displacement_solver(A, comm):
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.setFromOptions()
    ksp.setUp()
    return ksp


def displacement_from_fc(A, ksp, Ic, nrm, fc, tdim, force_sign=-1.0):
    rhs = A.createVecRight()
    u = A.createVecRight()

    cd = np.stack([Ic * tdim + c for c in range(tdim)], axis=1).astype(np.int32)
    values = force_sign * np.asarray(fc, dtype=float)[:, None] * nrm

    rhs.set(0.0)
    rhs.setValues(
        cd.reshape(-1),
        values.reshape(-1).astype(PETSc.ScalarType),
        addv=PETSc.InsertMode.INSERT_VALUES,
    )
    rhs.assemble()
    ksp.solve(rhs, u)
    return u.array.copy()


def set_displacement_field(fn, values):
    values = np.asarray(values, dtype=PETSc.ScalarType)
    fn.x.array[:] = values
    fn.x.scatter_forward()


def export_contact_results(mesh, V, Ic, nrm, tdim, case_name, records, outdir):
    outdir = Path(outdir)
    if mesh.comm.Get_rank() == 0:
        outdir.mkdir(parents=True, exist_ok=True)
    mesh.comm.Barrier()

    fc_exact_fn = Function(V)
    fc_exact_fn.name = "fc_exact"
    fc_app_fn = Function(V)
    fc_app_fn.name = "fc_app"
    fc_err_fn = Function(V)
    fc_err_fn.name = "fc_abs_error"
    u_exact_fn = Function(V)
    u_exact_fn.name = "u_exact"
    u_app_fn = Function(V)
    u_app_fn.name = "u_app"
    u_err_fn = Function(V)
    u_err_fn.name = "u_error"

    path = outdir / f"{case_name}.pvd"
    with VTKFile(mesh.comm, str(path), "w") as vtk:
        vtk.write_mesh(mesh)
        for rec in records:
            set_contact_field(fc_exact_fn, Ic, nrm, rec["fc_exact"], tdim)
            set_contact_field(fc_app_fn, Ic, nrm, rec["fc_app"], tdim)
            set_contact_field(
                fc_err_fn, Ic, nrm, rec["fc_app"] - rec["fc_exact"], tdim, absolute=True
            )
            set_displacement_field(u_exact_fn, rec["u_exact"])
            set_displacement_field(u_app_fn, rec["u_app"])
            set_displacement_field(u_err_fn, rec["u_app"] - rec["u_exact"])
            vtk.write_function(
                [fc_exact_fn, fc_app_fn, fc_err_fn, u_exact_fn, u_app_fn, u_err_fn],
                t=float(rec["ns"]),
            )

    return path


def run_contact_comparison(mesh, V, A, Ic, nrm, tdim, S0, approximations, x):
    outdir = Path("results/Sc_shift_contact")
    all_results = {}
    ksp_u = displacement_solver(A, mesh.comm)

    for case in contact_cases(x):
        name = case["name"]
        gap = sphere_gap(
            x, center_xy=case["center_xy"], radius=case["radius"], delta=case["delta"]
        )

        fc_exact, w_exact, info_exact = solve_contact_fc(S0, gap)
        u_exact = displacement_from_fc(A, ksp_u, Ic, nrm, fc_exact, tdim)
        exact_norm = np.linalg.norm(fc_exact)

        print(f"\n--- Contact case: {name} ---")
        print("center_xy =", case["center_xy"])
        print("radius    =", case["radius"])
        print("delta     =", case["delta"])
        print("initial penetrating nodes =", np.count_nonzero(gap < 0.0))
        print("exact solver =", info_exact["solver"], info_exact["msg"])
        print("exact active nodes =", np.count_nonzero(fc_exact > 1e-12))
        print("||fc_exact||2 =", exact_norm)

        records = []
        for app in approximations:
            ns = app["ns"]
            fc_app, w_app, info_app = solve_contact_fc(app["S"], gap)
            u_app = displacement_from_fc(A, ksp_u, Ic, nrm, fc_app, tdim)
            l2_error = np.linalg.norm(fc_app - fc_exact)
            rel_l2_error = l2_error / exact_norm if exact_norm > 0.0 else l2_error

            print(f"ns = {ns}")
            print("  app solver =", info_app["solver"], info_app["msg"])
            print("  active nodes =", np.count_nonzero(fc_app > 1e-12))
            print("  ||fc_app - fc_exact||2 =", l2_error)
            print("  relative L2 error      =", rel_l2_error)

            records.append(
                {
                    "ns": ns,
                    "fc_exact": fc_exact.copy(),
                    "fc_app": fc_app.copy(),
                    "u_exact": u_exact.copy(),
                    "u_app": u_app.copy(),
                    "l2_error": l2_error,
                    "rel_l2_error": rel_l2_error,
                    "gap": gap.copy(),
                    "w_exact": w_exact.copy(),
                    "w_app": w_app.copy(),
                }
            )

        path = export_contact_results(mesh, V, Ic, nrm, tdim, name, records, outdir)
        print("exported contact fields to", path)
        all_results[name] = records

    return all_results


def plot_sampled_points(xc, sid, outname="shifted_sampled_points.png"):
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.plot(xc[:, 0], xc[:, 1], "kx", markersize=4, label="contact nodes")
    ax.plot(xc[sid, 0], xc[sid, 1], "ro", markersize=6, label="sampled columns")
    for a, j in enumerate(sid):
        ax.text(xc[j, 0], xc[j, 1], str(a), fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Sampled template columns")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.savefig(outname, dpi=200)
    return fig, ax


def main():
    comm = MPI.COMM_WORLD

    mesh, _, _ = gmshio.read_from_msh("./msh_files/cube_tetra.msh", comm, 0, gdim=3)

    V, A, b, pb = sys(mesh, E=1.0e9, nu=0.3, gu=lambda x: np.isclose(x[2], 0.0))

    tdim = mesh.topology.dim
    fdim = tdim - 1

    Gc = locate_entities_boundary(
        mesh, fdim, lambda x: np.isclose(x[2], 1.0, atol=1e-8)
    )
    gid = 1
    gt = np.full(Gc.shape, gid, dtype=np.int32)

    od = np.argsort(Gc)
    fi = Gc[od].astype(np.int32)
    fv = gt[od].astype(np.int32)

    mt = meshtags(mesh, fdim, fi, fv)
    ds = Measure("ds", domain=mesh, subdomain_data=mt)

    Ic = uniq_v(mesh, Gc)

    nrm, nf = nrm_bnd(mesh=mesh, Iv=Ic, ds_c=ds(gid), save=True)

    xc = mesh.geometry.x[Ic]

    S0 = Sc_n(A, Ic, nrm, tdim, show=True)
    _, s0, _ = np.linalg.svd(S0)

    nss = [10, 20, 40, 70, 99]

    fig, ax = plt.subplots(figsize=(18, 5), constrained_layout=True)
    ax.semilogy(
        np.arange(1, len(s0) + 1),
        s0,
        marker="o",
        markersize=4,
        linewidth=1.8,
        label=r"$S_c$",
    )

    rel_errors = []
    approximations = []

    for ns in nss:
        Sa, sid, inf, par, xsampled = S_shift(
            S0,
            xc,
            ns=ns,
            mode="mixed",
            ps=4,
            qs=2.0,
            kint=8,
            pint=3.0,
            sym=True,
            psd=False,
            show=True,
        )

        R = S0 - Sa
        _, sa, _ = np.linalg.svd(Sa)
        rel_err = np.linalg.norm(R, "fro") / np.linalg.norm(S0, "fro")
        rel_errors.append(rel_err)
        approximations.append({"ns": ns, "S": Sa, "sid": sid, "params": par})

        ax.semilogy(
            np.arange(1, len(sa) + 1),
            sa,
            marker="s",
            markersize=4,
            linewidth=1.8,
            label=rf"$S_c^{{shift}}$, $n_s={ns}$",
        )

        B = 0.5 * (Sa + Sa.T)
        ev = np.linalg.eigvalsh(B)
        sym_err = np.linalg.norm(Sa - Sa.T, "fro") / np.linalg.norm(Sa, "fro")

        print("\n--- Params ---")
        print("ns   =", ns)
        print("mode =", par["mode"])
        print("ps   =", par["ps"])
        print("qs   =", par["qs"])
        print("kint =", par["kint"])
        print("pint =", par["pint"])
        print("sym  =", par["sym"])
        print("psd  =", par["psd"])

        print("\n--- Frobenius norms ---")
        print("||S0||F =", np.linalg.norm(S0, "fro"))
        print("||Sa||F =", np.linalg.norm(Sa, "fro"))
        print("||R||F  =", np.linalg.norm(R, "fro"))

        print("\n--- Relative residual ---")
        print(rel_err)

        print("\n--- Symmetry and eigenvalues ---")
        print("sym err =", sym_err)
        print("eig min =", ev[0])
        print("eig max =", ev[-1])

        print("\n--- Diagonal min/max ---")
        print("S0 :", np.min(np.diag(S0)), np.max(np.diag(S0)))
        print("Sa :", np.min(np.diag(Sa)), np.max(np.diag(Sa)))

        print("\n--- Sampled columns ---")
        print(sid)

        if ns == nss[-1]:
            plot_sampled_points(xc, sid)

    ax.set_title("Spectrum")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    fig.savefig("shifted_spectrum.png", dpi=200)

    fig_err, ax_err = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax_err.plot(nss, rel_errors, marker="o", markersize=5, linewidth=1.8)
    ax_err.set_title("Relative approximation error")
    ax_err.set_xlabel(r"$n_s$")
    ax_err.set_ylabel(r"$\|S_c - S_c^{shift}\|_F / \|S_c\|_F$")
    ax_err.grid(True, which="both", linestyle="--", alpha=0.5)
    fig_err.savefig("shifted_relative_error.png", dpi=200)

    contact_results = run_contact_comparison(
        mesh=mesh,
        V=V,
        A=A,
        Ic=Ic,
        nrm=nrm,
        tdim=tdim,
        S0=S0,
        approximations=approximations,
        x=xc,
    )

    fig_fc, ax_fc = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for case_name, records in contact_results.items():
        ax_fc.plot(
            [rec["ns"] for rec in records],
            [rec["rel_l2_error"] for rec in records],
            marker="o",
            markersize=5,
            linewidth=1.8,
            label=case_name,
        )
    ax_fc.set_title("Contact force error")
    ax_fc.set_xlabel(r"$n_s$")
    ax_fc.set_ylabel(r"$\|f_c^{shift} - f_c^{exact}\|_2 / \|f_c^{exact}\|_2$")
    ax_fc.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_fc.legend()
    fig_fc.savefig("shifted_contact_force_error.png", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()
