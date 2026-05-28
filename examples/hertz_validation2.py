import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from tqdm import tqdm

from scipy.optimize import curve_fit
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.io import VTKFile, gmsh
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


from cofebem.contact.rigid_indenters import parabolic
from cofebem.contact.lcp_solvers.lemke import lemkelcp

import matplotlib.pyplot as plt


def p_hertz(r, p0, a):
    return p0 * np.sqrt(np.clip(1.0 - (r / a) ** 2, 0.0, None))


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

    f = Constant(mesh, np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType))

    a = inner(sig(u), eps(v)) * dx
    L = inner(f, v) * dx

    Gu = locate_entities_boundary(mesh, fdim, gu)
    Gd = locate_dofs_topological(V, fdim, Gu)

    u0 = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType)
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


if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    L = 40.0
    H = 20.0

    contact_center = np.array([L / 2, L / 2])
    delta = 0.08
    R = 100.0

    E = 1.0e9
    nu = 0.3

    E_star = E / (1.0 - nu**2)
    a_theo = np.sqrt(R * delta)
    P_theo = 4 / 3 * E_star * np.sqrt(R) * delta**1.5
    p0_theo = 3 * P_theo / (2 * np.pi * a_theo**2)

    mesh0 = gmsh.read_from_msh("./cofebem/mesh/smart_Hertz0.msh", comm, 0, gdim=3).mesh

    mesh1 = gmsh.read_from_msh("./cofebem/mesh/smart_Hertz1.msh", comm, 0, gdim=3).mesh

    mesh2 = gmsh.read_from_msh("./cofebem/mesh/smart_Hertz2.msh", comm, 0, gdim=3).mesh

    mesh3 = gmsh.read_from_msh("./cofebem/mesh/smart_Hertz3.msh", comm, 0, gdim=3).mesh

    mesh4 = gmsh.read_from_msh("./cofebem/mesh/smart_Hertz4.msh", comm, 0, gdim=3).mesh

    meshes = [mesh0, mesh1, mesh2, mesh3, mesh4]

    for i, mesh in enumerate(meshes):
        V, A, b, pb = sys(
            mesh,
            E=E,
            nu=nu,
            gu=lambda x: np.isclose(x[2], 0.0),
        )

        tdim = mesh.topology.dim
        fdim = tdim - 1

        Gc = locate_entities_boundary(
            mesh,
            fdim,
            lambda x: np.isclose(x[2], H, atol=1e-8),
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

        Sc = Sc_n(A, Ic, nrm, tdim, show=True)

        np.save(f"./results/hertz/Sc{i}.npy", Sc)

        g = (
            parabolic(
                xc[:, 0],
                xc[:, 1],
                contact_center[0],
                contact_center[1],
                R,
                np.full_like(xc[:, 2], H - delta),
            )
            - xc[:, 2]
        )

        max_iter = 10000
        fc_lemke, _, _ = lemkelcp(Sc, g, max_iter)

        facet2verts = mesh.topology.connectivity(fdim, 0)

        facet_area = np.zeros(len(Gc), dtype=np.float64)
        area_node = np.zeros(mesh.geometry.x.shape[0], dtype=np.float64)

        for local_i, facet in enumerate(Gc):
            verts = facet2verts.links(facet)
            x0, x1, x2 = mesh.geometry.x[verts]

            area_f = 0.5 * np.linalg.norm(np.cross(x1 - x0, x2 - x0))
            facet_area[local_i] = area_f

            share = area_f / 3.0
            for v in verts:
                area_node[v] += share

        p_num = fc_lemke / area_node[Ic]

        rc = np.sqrt(
            (xc[:, 0] - contact_center[0]) ** 2 + (xc[:, 1] - contact_center[1]) ** 2
        )

        p_hertz_ = p_hertz(rc, p0_theo, a_theo)

        p_hertz_avg = (np.pi / 4) * p0_theo
        err_r = (np.abs(p_num - p_hertz_) ** 2) / (p_hertz_avg**2)

        p_fenics_num = Function(V)
        p_fenics_num.name = "p_num"

        p_fenics_hertz = Function(V)
        p_fenics_hertz.name = "p_hertz"

        err_fenics = Function(V)
        err_fenics.name = "error_p"

        with VTKFile(mesh.comm, f"./results/hertz/hertz_smart{i}.pvd", "w") as vtk:
            vtk.write_mesh(mesh)
            vtk.write_function([p_fenics_num, p_fenics_hertz, err_fenics], 0)

        p_fenics_num.x.array[:] = 0
        p_fenics_num.x.array[3 * Ic + 2] = p_num
        p_fenics_num.x.scatter_forward()

        p_fenics_hertz.x.array[:] = 0
        p_fenics_hertz.x.array[3 * Ic + 2] = p_hertz_
        p_fenics_hertz.x.scatter_forward()

        err_fenics.x.array[:] = 0
        err_fenics.x.array[3 * Ic + 2] = err_r
        err_fenics.x.scatter_forward()

        vtk.write_function([p_fenics_num, p_fenics_hertz, err_fenics], 1)

        print(f"Done with mesh {i}")
