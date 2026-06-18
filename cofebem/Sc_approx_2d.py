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
    form,
    create_form,
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
from matplotlib.collections import LineCollection


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


def proj_nrm_Gc(vec, Ic, nrm, tdim):
    comm = vec.getComm()
    Ic = np.asarray(Ic, dtype=np.int64)
    nrm = np.asarray(nrm, dtype=float)
    nc = int(Ic.size)

    m = np.linalg.norm(nrm, axis=1)
    n = nrm / m[:, None]

    cd = np.stack([Ic * tdim + c for c in range(tdim)], axis=1).astype(np.int32)
    cdf = cd.reshape(-1).astype(np.int32)

    uc = PETSc.Vec().createSeq(tdim * nc, comm=PETSc.COMM_SELF)
    isf = PETSc.IS().createGeneral(cdf, comm=comm)
    ist = PETSc.IS().createStride(tdim * nc, first=0, step=1, comm=PETSc.COMM_SELF)
    sca = PETSc.Scatter().create(vec, isf, uc, ist)

    uc.set(0.0)
    sca.scatter(
        vec,
        uc,
        addv=PETSc.InsertMode.INSERT_VALUES,
        mode=PETSc.ScatterMode.FORWARD,
    )

    ux = uc.getArray(readonly=True).copy().reshape(nc, tdim)
    un = np.einsum("ij,ij->i", n, ux)
    return ux, un


def solve_uniform_p0(A, pb, mesh, V, Ic, nrm, ds_c, p0=1.0):
    tdim = mesh.topology.dim
    v = TestFunction(V)
    nf = FacetNormal(mesh)

    p = Constant(mesh, PETSc.ScalarType(p0))
    Lp = form(inner(p * nf, v) * ds_c)

    bp = A.createVecRight()
    up = A.createVecRight()

    with bp.localForm() as b0:
        b0.set(0.0)

    assemble_vector(bp, Lp)
    apply_lifting(bp, [pb._a], bcs=[pb.bcs])
    bp.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    for bc in pb.bcs:
        bc.set(bp.array_w)

    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.setFromOptions()
    ksp.setUp()

    ksp.solve(bp, up)

    uh = Function(V)
    uh.x.array[:] = up.getArray(readonly=True)
    uh.x.scatter_forward()
    uh.name = "u_ring"

    _, q_ring = proj_nrm_Gc(bp, Ic, nrm, tdim)
    _, u_ring = proj_nrm_Gc(up, Ic, nrm, tdim)

    return u_ring, q_ring, uh


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


def plot_mesh(mesh, loaded_nodes=None):
    fig, ax = plt.subplots(figsize=(7, 5))

    tdim = mesh.topology.dim
    fdim = tdim - 1

    mesh.topology.create_connectivity(tdim, fdim)
    mesh.topology.create_connectivity(fdim, 0)

    c2f = mesh.topology.connectivity(tdim, fdim)
    f2v = mesh.topology.connectivity(fdim, 0)

    x = mesh.geometry.x[:, :2]

    facets = set()
    for c in range(mesh.topology.index_map(tdim).size_local):
        for f in c2f.links(c):
            facets.add(int(f))

    segs = []
    for f in facets:
        vs = f2v.links(f)
        if len(vs) == 2:
            segs.append([x[vs[0]], x[vs[1]]])

    lc = LineCollection(segs, colors="k", linewidths=0.8)
    ax.add_collection(lc)

    ax.plot(x[:, 0], x[:, 1], "ro", label="nodes")

    if loaded_nodes is not None:
        loaded_nodes = np.atleast_1d(np.asarray(loaded_nodes, dtype=np.int32))

        ax.plot(
            x[loaded_nodes, 0],
            x[loaded_nodes, 1],
            "bx",
            markersize=10,
            markeredgewidth=2,
            label="loaded nodes",
        )

    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Mesh with nodes and loaded nodes")
    ax.grid(False)
    ax.legend()

    plt.show()


def dmat(x):
    x = np.asarray(x, dtype=float)
    y = x[:, None, :] - x[None, :, :]
    return np.linalg.norm(y, axis=2)


comm = MPI.COMM_WORLD

nx, ny = 5, 10

mesh = create_rectangle(
    comm,
    [np.array([0.0, 0.0]), np.array([2.0, 1.0])],
    [nx, ny],
    CellType.quadrilateral,
)


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
Nc = len(xc)
Sc = Sc_n(A, Ic, nrm, tdim, show=True)

# plt.imshow(np.log(np.abs(Sc)), aspect="equal", cmap="viridis")
# plt.axis("off")
# plt.tight_layout()
# plt.savefig("heatmap_Sc.png", dpi=500)
# plt.show()
# J_loc = [0, 2]
J_loc = [0, int(nx / 2)]
J_glob = Ic[J_loc]


plot_mesh(mesh, J_glob)

Sc_sampled = Sc[:, J_loc]

Sc_tilde = np.zeros((Nc, Nc))
# from the symmetry of the operator
Sc_tilde[:, J_loc] = Sc_sampled
Sc_tilde[J_loc, :] = Sc_sampled.T

mid = int(nx / 2)
# from the symmetry of the mesh
# perm = np.array([4, 3, 2, 1, 0], dtype=int)

perm = np.arange(nx, -1, -1)
Sc_tilde[:, nx] = Sc_tilde[perm, 0]
Sc_tilde[nx, :] = Sc_tilde[:, nx]


D = dmat(xc)


u_ring, _, _ = solve_uniform_p0(
    A=A,
    pb=pb,
    mesh=mesh,
    V=V,
    Ic=Ic,
    nrm=nrm,
    ds_c=ds(gid),
    p0=1.0,
)


xg = mesh.geometry.x[Ic, 0]
ids = np.argsort(xg)

xg = xg[ids]
u_plot = u_ring[ids]

plt.figure(figsize=(7, 5))
plt.plot(xg, u_plot, "ro-", linewidth=1.5, markersize=6, label=r"$u_{\mathrm{ring}}$")
plt.title("Displacement due to uniform pressure on $\Gamma_c$")
plt.xlabel("x")
plt.ylabel(r"$u_{\mathrm{ring}}$")
plt.grid(True)
plt.legend()
plt.show()


def lin_w(x, xa, xb):
    xa = float(xa)
    xb = float(xb)
    x = float(x)
    if abs(xb - xa) < 1e-14:
        return 1.0, 0.0
    wa = (xb - x) / (xb - xa)
    wb = (x - xa) / (xb - xa)
    return wa, wb


xgc = xc[:, 0]

w0, w2 = lin_w(xgc[1], xgc[0], xgc[int((nx / 2) - 1)])
print("------------------ dist interpolation---------------------------")
print(w0, w2)
S11 = w0 * Sc_tilde[0, 0] + w2 * Sc_tilde[mid, mid]

S33 = S11

m02 = 0.5 * (xgc[0] + xgc[2])
m13 = 0.5 * (xgc[1] + xgc[3])
m24 = 0.5 * (xgc[2] + xgc[4])

w02, w24 = lin_w(m13, m02, m24)
S13 = w02 * Sc_tilde[0, 2] + w24 * Sc_tilde[2, 4]

Sc_tilde[1, 1] = S11
Sc_tilde[3, 3] = S33
Sc_tilde[1, 3] = S13
Sc_tilde[3, 1] = S13

w0 = (u_ring[1] - u_ring[mid]) / (u_ring[0] - u_ring[mid])
w2 = 1 - w0
S11 = w0 * Sc_tilde[0, 0] + w2 * Sc_tilde[mid, mid]
print(w0, w2)
print(f"err_weight_dist at (1,1)= {np.abs(S11-Sc[1,1])/np.abs(Sc[1,1])}")

# print(Sc[0, 0], Sc[1, 1], Sc[mid, mid])
# print(S11)

# print(Sc[1, 3], Sc[0, 2], Sc[2, 4])
# print(Sc_tilde[1, 3], Sc_tilde[0, 2], Sc_tilde[2, 4])
# print(S13)
# print(w02, w24)
# print(f"err_weight_dist at (1,3)= {np.abs(S13-Sc[1,3])/np.abs(Sc[1,3])}")
np.set_printoptions(precision=3, suppress=True)
# print(Sc * 1.0e11)


# --- S13 estimation using interpolated logarithmic models ---


def log_model(r, A, B):
    return -A * np.log(r) + B


def fit_log_column(Sc_col, D_col, j, rmin=1.0 / 4.0):
    """
    Fit Sc[:, j] ≈ -A log(r) + B
    skipping the self value and near-field points.
    """
    mask = np.ones_like(Sc_col, dtype=bool)

    mask[j] = False

    mask &= D_col > rmin

    r_fit = D_col[mask]
    s_fit = Sc_col[mask]

    popt, _ = curve_fit(log_model, r_fit, s_fit)
    A, B = popt

    return A, B, r_fit, s_fit


# known sampled columns: node 0 and node 2
A0, B0, r0_fit, s0_fit = fit_log_column(
    Sc_tilde[:, 0],
    D[:, 0],
    j=0,
    rmin=0.45,
)

A2, B2, r2_fit, s2_fit = fit_log_column(
    Sc_tilde[:, 2],
    D[:, 2],
    j=2,
    rmin=0.45,
)

# print(len(r0_fit))
# print(r0_fit)
# w0, w2 = lin_w(xgc[3], xgc[0], xgc[2])
# w0 = w2 = 0.5
# A3 = w0 * A0 + w2 * A2
# B3 = w0 * B0 + w2 * B2
# r13 = D[1, 3]
# S13_log_interp = log_model(r13, A3, B3)

# print(f"A0 = {A0:.6e}, B0 = {B0:.6e}")
# print(f"A2 = {A2:.6e}, B2 = {B2:.6e}")
# print(f"w0 = {w0:.6e}, w2 = {w2:.6e}")
# print(f"A3 = {A3:.6e}, B3 = {B3:.6e}")
# print(f"S13 log-interpolated = {S13_log_interp:.6e}")
# print(f"S13 exact            = {Sc[1, 3]:.6e}")
# print(
#     f"err S13 log-interpolated = "
#     f"{abs(S13_log_interp - Sc[1, 3]) / abs(Sc[1, 3]):.3%}"
# )

# S13 = S13_log_interp
# ------------------------ Uniform discrete force on Gamma_c --------------------


def solve_uniform_f0(A, mesh, Ic, nrm, tdim, f0=1.0):
    comm = mesh.comm
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
    u_force = A.createVecRight()

    cd = np.stack([Ic * tdim + c for c in range(tdim)], axis=1).astype(np.int32)
    cdf = cd.reshape(-1).astype(np.int32)

    uc = PETSc.Vec().createSeq(tdim * nc, comm=PETSc.COMM_SELF)
    isf = PETSc.IS().createGeneral(cdf, comm=comm)
    ist = PETSc.IS().createStride(tdim * nc, first=0, step=1, comm=PETSc.COMM_SELF)
    sca = PETSc.Scatter().create(u_force, isf, uc, ist)

    rhs.set(0.0)
    for j in range(nc):
        rhs.setValues(cd[j], f0 * n[j], addv=PETSc.InsertMode.INSERT_VALUES)
    rhs.assemble()

    ksp.solve(rhs, u_force)

    uc.set(0.0)
    sca.scatter(
        u_force,
        uc,
        addv=PETSc.InsertMode.INSERT_VALUES,
        mode=PETSc.ScatterMode.FORWARD,
    )

    ux = uc.getArray(readonly=True).copy().reshape(nc, tdim)
    un = np.einsum("ij,ij->i", n, ux)

    return rhs, u_force, ux, un


rhs_force, u_force_vec, ux_force, u_ring_force = solve_uniform_f0(
    A=A,
    mesh=mesh,
    Ic=Ic,
    nrm=nrm,
    tdim=tdim,
    f0=1.0,
)

xg = mesh.geometry.x[Ic, 0]
ids = np.argsort(xg)
print(u_ring_force * 1e9)
xg = xg[ids]
u_plot = u_ring_force[ids]

print("--------------------uniform pressure/force interpol--------------")
w0 = (u_ring_force[1] - u_ring_force[mid]) / (u_ring_force[0] - u_ring_force[mid])
w2 = 1 - w0
S11 = w0 * Sc_tilde[0, 0] + w2 * Sc_tilde[mid, mid]
print(w0, w2)
print(f"err_weight_dist force at (1,1)= {np.abs(S11-Sc[1,1])/np.abs(Sc[1,1])}")

# print(Sc[0, 0], Sc[1, 1], Sc[mid, mid])
# print(S11)

# S11_direct = u_plot[1]
# print(f"S11 direct = {S11_direct}")
# print(
#     f"err_weight_dist force direct at (1,1)= {np.abs(S11_direct-Sc[1,1])/np.abs(Sc[1,1])}"
# )


def Sii_interpolate(nx, s00, snx2, u_ring):
    print(nx, s00, snx2, u_ring)
    sii = np.zeros((int(nx / 2 - 2)))
    for i in range(int(nx / 2) - 2):
        w0 = (u_ring[i + 1] - u_ring[int(nx / 2)]) / (u_ring[0] - u_ring[int(nx / 2)])
        wnx2 = 1 - w0
        Sii = w0 * s00 + wnx2 * snx2
        sii[i] = Sii

    return sii


sii = Sii_interpolate(nx, Sc_tilde[0, 0], Sc_tilde[mid, mid], u_ring_force)

print(f"sii interpolated = {sii*1e9}")
print(Sc[1, 1] * 1e9, Sc[2, 2] * 1e9)

plt.figure(figsize=(7, 5))
plt.plot(xg, u_plot, "ro-", linewidth=1.5, markersize=6, label=r"$u_{\mathrm{force}}$")
plt.title("Displacement due to uniform discrete force on $\Gamma_c$")
plt.xlabel("x")
plt.ylabel(r"$u_{\mathrm{force}}$")
plt.grid(True)
plt.legend()
plt.show()
