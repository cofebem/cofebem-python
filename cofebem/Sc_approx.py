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

from cofebem.contact.lcp_solvers.lemke import lemkelcp
from cofebem.contact.lcp_solvers.psor import psor_lcp

import matplotlib.pyplot as plt


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
        it = tqdm(it, desc="Sampling Snn", unit="col")

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
        a=a,
        L=L,
        u=nf,
        bcs=[],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
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
        a=a,
        L=L,
        bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
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


def dmat(x):
    x = np.asarray(x, dtype=float)
    y = x[:, None, :] - x[None, :, :]
    return np.linalg.norm(y, axis=2)


def hlen(D):
    E = D.copy()
    np.fill_diagonal(E, np.inf)
    return float(np.median(np.min(E, axis=1)))


def fps(x, m):
    n = x.shape[0]
    if m >= n:
        return np.arange(n, dtype=int)
    D = dmat(x)
    s = [int(np.argmax(np.sum(D, axis=1)))]
    md = D[s[0]].copy()
    while len(s) < m:
        j = int(np.argmax(md))
        s.append(j)
        md = np.minimum(md, D[j])
    return np.array(sorted(set(s)), dtype=int)


def near_id(x, k, rn):
    r = np.linalg.norm(x - x[k], axis=1)
    return np.where(r <= rn)[0]


def knn(x, s, k, p):
    s = np.asarray(s, dtype=int)
    r = np.linalg.norm(x[s] - x[k], axis=1)
    o = np.argsort(r)
    q = min(p, len(s))
    return s[o[:q]], r[o[:q]]


def wgt(r, q=1.0, eps=1e-14):
    z = 1.0 / (r**q + eps)
    return z / np.sum(z)


def fit_tail(c, x, j, rf):
    r = np.linalg.norm(x - x[j], axis=1)
    mk = r >= rf
    rr = r[mk]
    yy = np.asarray(c, dtype=float)[mk]

    if rr.size < 5:
        return 0.0, max(rf * 0.2, 1e-6), float(np.mean(c))

    b0 = max(np.min(rr[rr > 0]) * 0.25 if np.any(rr > 0) else 1e-6, 1e-6)
    b1 = max(np.max(rr), b0 * 10)
    bg = np.geomspace(b0, b1, 48)

    be = np.inf
    out = (0.0, b0, float(np.mean(yy)))

    for b in bg:
        ph = 1.0 / (rr + b)
        A = np.column_stack([ph, np.ones_like(ph)])
        co, *_ = np.linalg.lstsq(A, yy, rcond=None)
        a, g = co
        e = np.linalg.norm(A @ co - yy)
        if e < be:
            be = e
            out = (float(a), float(b), float(g))

    return out


def ip_par(x, s, prm, k, p=4, q=2.0):
    nb, rr = knn(x, s, k, p)
    ww = wgt(rr, q=q)
    a = 0.0
    b = 0.0
    g = 0.0
    for t, j in enumerate(nb):
        aj, bj, gj = prm[int(j)]
        a += ww[t] * aj
        b += ww[t] * bj
        g += ww[t] * gj
    return a, b, g, nb, rr, ww


def rad_sm(c, rs, rt, bw):
    rs = np.asarray(rs, dtype=float)
    rt = np.asarray(rt, dtype=float)
    c = np.asarray(c, dtype=float)
    Y = np.zeros_like(rt)
    for i, r in enumerate(rt):
        z = np.exp(-0.5 * ((rs - r) / bw) ** 2)
        s = np.sum(z)
        if s <= 1e-30:
            j = int(np.argmin(np.abs(rs - r)))
            Y[i] = c[j]
        else:
            Y[i] = np.dot(z, c) / s
    return Y


def pod_near(x, S, s, k, rn, p=4, rp=3, bw=None, q=2.0):
    idk = near_id(x, k, rn)
    rk = np.linalg.norm(x[idk] - x[k], axis=1)
    nb, rr = knn(x, s, k, p)
    X = []
    for j in nb:
        rj = np.linalg.norm(x - x[j], axis=1)
        cj = S[int(j)]
        v = rad_sm(cj, rj, rk, bw)
        X.append(v)
    X = np.column_stack(X)
    U, sv, VT = np.linalg.svd(X, full_matrices=False)
    r = min(rp, U.shape[1])
    Ur = U[:, :r]
    A = Ur.T @ X
    ww = wgt(rr, q=q)
    ak = A @ ww
    vk = Ur @ ak
    return idk, vk, sv, nb, rr, ww


def rec_col(x, S, s, prm, k, rn, rf, p=4, rp=3, bw=None, q=2.0):
    idk, vk, sv, nb, rr, ww = pod_near(x, S, s, k, rn, p=p, rp=rp, bw=bw, q=q)
    a, b, g, nb2, rr2, ww2 = ip_par(x, s, prm, k, p=p, q=q)
    r = np.linalg.norm(x - x[k], axis=1)
    c = a / (r + b) + g
    c[idk] = vk
    return c, {"id": idk, "a": a, "b": b, "g": g, "nb": nb, "w": ww, "sv": sv}


def S_app(S0, x, ns=12, rn=None, rf=None, p=4, rp=3, bw=None, show=True):
    n = S0.shape[0]
    D = dmat(x)
    h = hlen(D)

    if rn is None:
        rn = 3.0 * h
    if rf is None:
        rf = 2.5 * h
    if bw is None:
        bw = 1.25 * h

    sid = fps(x, ns)

    Sx = {int(j): S0[:, j].copy() for j in sid}
    prm = {int(j): fit_tail(S0[:, j], x, int(j), rf) for j in sid}

    A = np.zeros_like(S0)
    A[:, sid] = S0[:, sid]

    miss = [j for j in range(n) if j not in set(sid)]
    it = miss
    if show:
        it = tqdm(miss, desc="Building Sapp", unit="col")

    inf = {}
    for k in it:
        ck, ik = rec_col(x, Sx, sid, prm, int(k), rn, rf, p=p, rp=rp, bw=bw)
        A[:, k] = ck
        inf[int(k)] = ik

    for j in sid:
        inf[int(j)] = {
            "id": near_id(x, int(j), rn),
            "a": None,
            "b": None,
            "g": None,
            "nb": np.array([j]),
            "w": np.array([1.0]),
            "sv": None,
        }

    return A, sid, prm, inf, {"h": h, "rn": rn, "rf": rf, "bw": bw}, x[sid]


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

    D = dmat(x)
    h = hlen(D)
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
    return fc, w, {
        "solver": "psor",
        "code": code,
        "msg": msg,
        "iterations": its,
        "residual": history[-1] if history else np.nan,
    }


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
                fc_err_fn,
                Ic,
                nrm,
                rec["fc_app"] - rec["fc_exact"],
                tdim,
                absolute=True,
            )
            set_displacement_field(u_exact_fn, rec["u_exact"])
            set_displacement_field(u_app_fn, rec["u_app"])
            set_displacement_field(u_err_fn, rec["u_app"] - rec["u_exact"])
            vtk.write_function(
                [
                    fc_exact_fn,
                    fc_app_fn,
                    fc_err_fn,
                    u_exact_fn,
                    u_app_fn,
                    u_err_fn,
                ],
                t=float(rec["ns"]),
            )

    return path


def run_contact_comparison(mesh, V, A, Ic, nrm, tdim, S0, approximations, x):
    outdir = Path("results/Sc_approx_contact")
    all_results = {}
    ksp_u = displacement_solver(A, mesh.comm)

    for case in contact_cases(x):
        name = case["name"]
        gap = sphere_gap(
            x,
            center_xy=case["center_xy"],
            radius=case["radius"],
            delta=case["delta"],
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


comm = MPI.COMM_WORLD

mesh, _, _ = gmshio.read_from_msh("./msh_files/cube_tetra.msh", comm, 0, gdim=3)

V, A, b, pb = sys(
    mesh,
    E=1.0e9,
    nu=0.3,
    gu=lambda x: np.isclose(x[2], 0.0),
)

tdim = mesh.topology.dim
fdim = tdim - 1

Gc = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[2], 1.0, atol=1e-8))
gid = 1
gt = np.full(Gc.shape, gid, dtype=np.int32)

od = np.argsort(Gc)
fi = Gc[od].astype(np.int32)
fv = gt[od].astype(np.int32)

mt = meshtags(mesh, fdim, fi, fv)
ds = Measure("ds", domain=mesh, subdomain_data=mt)

Ic = uniq_v(mesh, Gc)

nrm, nf = nrm_bnd(
    mesh=mesh,
    Iv=Ic,
    ds_c=ds(gid),
)

xc = mesh.geometry.x[Ic]

S0 = Sc_n(A, Ic, nrm, tdim, show=True)
_, s0, _ = np.linalg.svd(S0)

# ns = 12
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
    Sa, sid, prm, inf, par, xsampled = S_app(
        S0, xc, ns=ns, rn=None, rf=None, p=4, rp=3, bw=None, show=True
    )

    # Sa = 0.5 * (Sa + Sa.T)
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
        label=rf"$S_c^{{app}}$, $n_s={ns}$",
    )

    print("\n--- Params ---")
    print("ns =", ns)
    print("h  =", par["h"])
    print("rn =", par["rn"])
    print("rf =", par["rf"])
    print("bw =", par["bw"])

    print("\n--- Frobenius norms ---")
    print("||S0||F =", np.linalg.norm(S0, "fro"))
    print("||Sa||F =", np.linalg.norm(Sa, "fro"))
    print("||R||F  =", np.linalg.norm(R, "fro"))

    print("\n--- Relative residual ---")
    print(rel_err)

    print("\n--- Diagonal min/max ---")
    print("S0 :", np.min(np.diag(S0)), np.max(np.diag(S0)))
    print("Sa :", np.min(np.diag(Sa)), np.max(np.diag(Sa)))

    print("\n--- Sampled columns ---")
    print(sid)

    # fig, ax2 = plt.subplots()
    # print(xsampled[:, 2])
    # ax2.plot(xsampled[:, 0], xsampled[:, 1], "ro")

    # for i, idx in enumerate(sid):
    #     ax2.text(xc[idx, 0], xc[idx, 1], f"{i}")
    # ax2.plot(xc[:, 0], xc[:, 1], "bx", zorder=-1)

ax.set_title("Spectrum")
ax.set_xlabel("Singular value index")
ax.set_ylabel("Singular value")
ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.legend()

fig_err, ax_err = plt.subplots(figsize=(8, 5), constrained_layout=True)
ax_err.plot(
    nss,
    rel_errors,
    marker="o",
    markersize=5,
    linewidth=1.8,
)
ax_err.set_title("Relative approximation error")
ax_err.set_xlabel(r"$n_s$")
ax_err.set_ylabel(r"$\|S_c - S_c^{app}\|_F / \|S_c\|_F$")
ax_err.grid(True, which="both", linestyle="--", alpha=0.5)

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
ax_fc.set_ylabel(r"$\|f_c^{app} - f_c^{exact}\|_2 / \|f_c^{exact}\|_2$")
ax_fc.grid(True, which="both", linestyle="--", alpha=0.5)
ax_fc.legend()

plt.show()
