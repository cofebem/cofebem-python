import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from tqdm import tqdm

from scipy.optimize import curve_fit
from dolfinx.mesh import create_rectangle, locate_entities_boundary, meshtags, CellType
from dolfinx.io import VTKFile
from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import (
    LinearProblem,
    assemble_matrix,
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
            u,
            uc,
            addv=PETSc.InsertMode.INSERT_VALUES,
            mode=PETSc.ScatterMode.FORWARD,
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
        petsc_options_prefix="nrm_",
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

    la = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    V = functionspace(mesh, ("Lagrange", 1, (tdim,)))
    u = TrialFunction(V)
    v = TestFunction(V)

    def eps(w):
        return sym(grad(w))

    def sig(w):
        return la * tr(eps(w)) * Identity(tdim) + 2.0 * mu * eps(w)

    f = Constant(mesh, np.array([0.0, 0.0], dtype=PETSc.ScalarType))

    a = inner(sig(u), eps(v)) * dx
    L = inner(f, v) * dx

    Gu = locate_entities_boundary(mesh, fdim, gu)
    Gd = locate_dofs_topological(V, fdim, Gu)

    u0 = np.array([0.0, 0.0], dtype=PETSc.ScalarType)
    bc = dirichletbc(u0, Gd, V)
    bcs = [bc]

    pb = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix="sc_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    pb._A.zeroEntries()
    assemble_matrix(pb._A, pb._a, bcs=pb.bcs)
    pb._A.assemble()

    with pb._b.localForm() as b0:
        b0.set(0.0)

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


def hyperbolic_model(r, alpha, beta):
    return alpha / (r + beta)


def log_model(r, A, beta, C):
    return -A * np.log(r + beta) + C


comm = MPI.COMM_WORLD

nx, ny = 20, 20

mesh = create_rectangle(
    comm,
    [np.array([0.0, 0.0]), np.array([2.0, 1.0])],
    [nx, ny],
    CellType.quadrilateral,
)

plt.figure(figsize=(7, 5))
plt.plot(mesh.geometry.x[:, 0], mesh.geometry.x[:, 1], "ro")
plt.gca().set_aspect("equal")
plt.title("Mesh nodes")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()


V, A, b, pb = sys(
    mesh,
    E=1.0e9,
    nu=0.3,
    gu=lambda x: np.isclose(x[1], 0.0),
)

tdim = mesh.topology.dim
fdim = tdim - 1

Gc = locate_entities_boundary(
    mesh,
    fdim,
    lambda x: np.isclose(x[1], 1.0, atol=1e-8),
)

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


col_id_local = 5
col_id_global = Ic[col_id_local]

plt.figure(figsize=(7, 5))
plt.plot(mesh.geometry.x[:, 0], mesh.geometry.x[:, 1], "ro")
plt.plot(
    mesh.geometry.x[col_id_global, 0],
    mesh.geometry.x[col_id_global, 1],
    "bx",
    markersize=10,
    markeredgewidth=2,
    label="loaded node",
)
plt.gca().set_aspect("equal")
plt.title("Selected loaded contact node")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()


S0_col = S0[:, col_id_local]

D = dmat(xc)
r = D[col_id_local, :]
h = hlen(D)

alpha0 = S0_col.max() * h
beta0_hyp = h

popt_hyp, pcov_hyp = curve_fit(
    hyperbolic_model,
    r,
    S0_col,
    p0=[alpha0, beta0_hyp],
    bounds=([-np.inf, 1e-14], [np.inf, np.inf]),
    maxfev=10000,
    method="trf",
)

alpha_hyp, beta_hyp = popt_hyp
S_hyp = hyperbolic_model(r, alpha_hyp, beta_hyp)

err_hyp = np.linalg.norm(S0_col - S_hyp) / np.linalg.norm(S0_col)


rmax = np.max(r)
ymax = np.max(S0_col)
ymin = np.min(S0_col)

beta0_log = h
den = np.log(rmax + beta0_log) - np.log(beta0_log)

if abs(den) < 1e-14:
    A0 = 1.0
else:
    A0 = max((ymax - ymin) / den, 1e-16)

C0 = ymin + A0 * np.log(rmax + beta0_log)

popt_log, pcov_log = curve_fit(
    log_model,
    r,
    S0_col,
    p0=[A0, beta0_log, C0],
    bounds=([0.0, 1e-14, -np.inf], [np.inf, np.inf, np.inf]),
    maxfev=10000,
    method="trf",
)

A_log, beta_log, C_log = popt_log
S_log = log_model(r, A_log, beta_log, C_log)

err_log = np.linalg.norm(S0_col - S_log) / np.linalg.norm(S0_col)


ids = np.argsort(r)

rs = r[ids]
ys = S0_col[ids]
yh = S_hyp[ids]
yl = S_log[ids]

rp = np.linspace(0.0, rmax, 500)
yh_p = hyperbolic_model(rp, alpha_hyp, beta_hyp)
yl_p = log_model(rp, A_log, beta_log, C_log)

print(f"h                         = {h:.6e}")
print(f"Hyperbolic model error    = {err_hyp:.6e}")
print(f"Log model error           = {err_log:.6e}")
print(f"Hyperbolic parameters     = alpha={alpha_hyp:.6e}, beta={beta_hyp:.6e}")
print(f"Log parameters            = A={A_log:.6e}, beta={beta_log:.6e}, C={C_log:.6e}")


fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    rs,
    ys,
    "ro",
    label=r"FE column $S_c[:,j]$",
)

ax.plot(
    rp,
    yh_p,
    "b-",
    linewidth=2,
    label=rf"Hyperbolic $\alpha/(r+\beta)$, err = {err_hyp:.2e}",
)

ax.plot(
    rp,
    yl_p,
    "g-",
    linewidth=2,
    label=rf"Log $-A\log(r+\beta)+C$, err = {err_log:.2e}",
)

ax.set_title("Comparison of hyperbolic and logarithmic fits")
ax.set_xlabel(r"Distance from loaded node $r$")
ax.set_ylabel(r"Normal compliance value $S_c(r)$")
ax.legend()
ax.grid(True)

plt.show()


fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    rs,
    ys - yh,
    "bo-",
    label=r"FE - hyperbolic fit",
)

ax.plot(
    rs,
    ys - yl,
    "go-",
    label=r"FE - log fit",
)

ax.axhline(0.0, color="k", linewidth=1)

ax.set_title("Residual comparison")
ax.set_xlabel(r"Distance from loaded node $r$")
ax.set_ylabel("Residual")
ax.legend()
ax.grid(True)

plt.show()


fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    rs,
    ys,
    "ro",
    label=r"FE column $S_c[:,j]$",
)

ax.plot(
    rs,
    yh,
    "bo",
    markerfacecolor="none",
    label=r"Hyperbolic prediction at contact nodes",
)

ax.plot(
    rs,
    yl,
    "go",
    markerfacecolor="none",
    label=r"Log prediction at contact nodes",
)

ax.set_title("Predicted values at contact nodes")
ax.set_xlabel(r"Distance from loaded node $r$")
ax.set_ylabel(r"Normal compliance value $S_c(r)$")
ax.legend()
ax.grid(True)

plt.show()
