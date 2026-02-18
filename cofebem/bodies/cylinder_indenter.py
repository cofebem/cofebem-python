import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from dolfinx.io import gmshio, VTKFile
from dolfinx.fem import functionspace, Function
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import locate_dofs_topological

from cofebem.contact.lcp_solvers.lemke import lemkelcp


# ------------------------------------------------------------
# Flat punch (cylindrical) helpers (NO class)
# ------------------------------------------------------------


def flat_punch_gap_on_plane_xy(xy, z_plane, z_bottom, center_xy, a):
    """
    Flat punch of radius a, with bottom plane at z=z_bottom.
    Gap on plane z=z_plane:
      inside (r<=a):  g = z_bottom - z_plane
      outside:       big positive (discourage contact)
    This mirrors your Cylinder.gap logic but evaluated on Γc at z=z_plane.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim == 1:
        xy = xy[np.newaxis, :]

    dx = xy[:, 0] - float(center_xy[0])
    dy = xy[:, 1] - float(center_xy[1])
    r2 = dx * dx + dy * dy

    g = np.empty(xy.shape[0], dtype=float)

    inside = r2 <= float(a) ** 2
    g[inside] = float(z_bottom) - float(z_plane)

    # outside: make gap large positive (like your "z_bottom + R^2 + z" trick, but on plane)
    g[~inside] = (float(z_bottom) - float(z_plane)) + float(a) ** 2
    return g


def p_flat_punch_popov(r, a, E_star, d, eps=1e-14):
    """
    Popov/MDR for rigid flat punch (radius a) indenting elastic half-space by depth d:
      p(r) = E* d / sqrt(a^2 - r^2),   r < a
      p(r) = 0,                       r >= a

    NOTE: singular at r -> a (edge). We regularize with eps in the denominator.
    """
    r = np.asarray(r, dtype=float)
    p = np.zeros_like(r)

    inside = r < a
    denom = np.sqrt(np.maximum(a * a - r[inside] * r[inside], eps))
    p[inside] = (E_star * float(d)) / denom
    return p


# ------------------------------------------------------------
# Load meshes and compliance matrices (same as you had)
# ------------------------------------------------------------

mesh0, _, _ = gmshio.read_from_msh(
    "./cofebem/mesh/smart_Hertz0.msh", MPI.COMM_WORLD, 0, gdim=3
)
mesh1, _, _ = gmshio.read_from_msh(
    "./cofebem/mesh/smart_Hertz1.msh", MPI.COMM_WORLD, 0, gdim=3
)
mesh2, _, _ = gmshio.read_from_msh(
    "./cofebem/mesh/smart_Hertz2.msh", MPI.COMM_WORLD, 0, gdim=3
)
mesh3, _, _ = gmshio.read_from_msh(
    "./cofebem/mesh/smart_Hertz3.msh", MPI.COMM_WORLD, 0, gdim=3
)
mesh4, _, _ = gmshio.read_from_msh(
    "./cofebem/mesh/smart_Hertz4.msh", MPI.COMM_WORLD, 0, gdim=3
)
meshes = [mesh0, mesh1, mesh2, mesh3, mesh4]

Sc_dense0 = np.load("Sc_smart0.npy")
Sc_dense1 = np.load("Sc_smart1.npy")
Sc_dense2 = np.load("Sc_smart2.npy")
Sc_dense3 = np.load("Sc_smart3.npy")
Sc_dense4 = np.load("Sc_smart4.npy")
Scs = [Sc_dense0, Sc_dense1, Sc_dense2, Sc_dense3, Sc_dense4]


# ------------------------------------------------------------
# Main comparison: flat punch vs COFEBEM
# ------------------------------------------------------------


def flat_punch_vs_cofebem(meshes, Scs):
    # ---- cube geometry ----
    W = 40.0
    Hplane = 20.0
    center_xy = np.array([W / 2.0, W / 2.0])

    # ---- material ----
    E = 1.0e9
    nu = 0.3
    E_star = E / (1.0 - nu**2)

    # ---- FE ----
    tol1 = 1.0e-5
    tdim, fdim = 3, 2
    element_type = "Lagrange"
    element_degree = 1

    def Gamma_c_selector(x):
        return np.isclose(x[2], Hplane, atol=tol1)

    # ---- punch parameters ----
    d = 0.02  # indentation depth (Popov uses d)
    a = 3.0  # punch radius (contact radius fixed by indenter)

    # place punch bottom at z = Hplane - d (so inside gap = -d)
    z_bottom = Hplane - d

    # Popov global quantities
    FN_theo = 2.0 * E_star * d * a
    p_avg = FN_theo / (
        np.pi * a * a
    )  # mean contact pressure (useful for scaling errors)

    # theory curve for plot
    r_th = np.linspace(0.0, a + 0.3, 800)
    p_th = p_flat_punch_popov(r_th, a=a, E_star=E_star, d=d)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r_th, p_th, "-", label="Popov (flat punch)")

    max_iter = 10000
    errors = []
    h_max = np.array([12.0, 9.6, 7.68, 6.144, 4.9152])

    for i, mesh in enumerate(meshes):
        V = functionspace(mesh, (element_type, element_degree, (tdim,)))

        Gamma_c = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_c_selector)
        Ic = locate_dofs_topological(V, fdim, Gamma_c)
        Gamma_c_x = mesh.geometry.x[Ic].reshape(-1, tdim)

        Sc_dense = Scs[i]

        # ---- GAP on Γc ----
        g = flat_punch_gap_on_plane_xy(
            Gamma_c_x[:, :2],
            z_plane=Hplane,
            z_bottom=z_bottom,
            center_xy=center_xy,
            a=a,
        )

        p0_ = np.maximum(-g, 0.0) * 1e16

        # ---- Solve LCP ----
        p_lemke, _, err_hist = lemkelcp(Sc_dense, g, max_iter)

        # ---- Lumped nodal area -> pressure ----
        facet2verts = mesh.topology.connectivity(fdim, 0)
        area_node = np.zeros(mesh.geometry.x.shape[0], dtype=np.float64)

        for facet in Gamma_c:
            verts = facet2verts.links(facet)
            x0v, x1v, x2v = mesh.geometry.x[verts]
            area_f = 0.5 * np.linalg.norm(np.cross(x1v - x0v, x2v - x0v))
            share = area_f / 3.0
            for v in verts:
                area_node[v] += share

        p_num = p_lemke / area_node[Ic]

        # ---- Theory at nodes ----
        r_xy = np.linalg.norm(Gamma_c_x[:, :2] - center_xy[None, :], axis=1)
        p_theo_nodes = p_flat_punch_popov(r_xy, a=a, E_star=E_star, d=d)

        # ---- Error field (scale by mean pressure, finite) ----
        err_r = (np.abs(p_num - p_theo_nodes) ** 2) / (p_avg**2)

        # ---- Export to VTK (store in z-component) ----
        p_fenics_num = Function(V)
        p_fenics_num.name = "p_num"
        p_fenics_theo = Function(V)
        p_fenics_theo.name = "p_popov_punch"
        err_fenics = Function(V)
        err_fenics.name = "error_p"

        p_fenics_num.x.array[:] = 0.0
        p_fenics_num.x.array[3 * Ic + 2] = p_num
        p_fenics_num.x.scatter_forward()

        p_fenics_theo.x.array[:] = 0.0
        p_fenics_theo.x.array[3 * Ic + 2] = p_theo_nodes
        p_fenics_theo.x.scatter_forward()

        err_fenics.x.array[:] = 0.0
        err_fenics.x.array[3 * Ic + 2] = err_r
        err_fenics.x.scatter_forward()

        with VTKFile(mesh.comm, f"flat_punch_smart{i}.pvd", "w") as vtk:
            vtk.write_mesh(mesh)
            vtk.write_function([p_fenics_num, p_fenics_theo, err_fenics], 0)

        # ---- Radial averaging for cleaner 1D comparison ----
        contact_nodes = p_num > 0.0
        a_num = r_xy[contact_nodes].max() if np.any(contact_nodes) else 0.0

        r_contact = r_xy[contact_nodes]
        p_contact = p_num[contact_nodes]

        nbins = 300
        rmax_bins = max(a, a_num)
        bins = np.linspace(0.0, rmax_bins, nbins + 1)
        r_mid = 0.5 * (bins[:-1] + bins[1:])
        p_avg_bins = np.full(nbins, np.nan)

        for k in range(nbins):
            m = (r_contact >= bins[k]) & (r_contact < bins[k + 1])
            if np.any(m):
                p_avg_bins[k] = p_contact[m].mean()

        p_ref = p_flat_punch_popov(r_mid, a=a, E_star=E_star, d=d)
        mask = np.isfinite(p_avg_bins) & np.isfinite(p_ref)

        err = (
            np.linalg.norm((p_avg_bins[mask] - p_ref[mask]) / p_avg)
            / np.sqrt(mask.sum())
            if np.any(mask)
            else np.nan
        )
        errors.append(err)

        ax.plot(
            r_mid[mask], p_avg_bins[mask], "-", alpha=1.0, label=f"cofebem mesh_{i}"
        )

        print(
            f"Done mesh {i} | center=({center_xy[0]:.3g},{center_xy[1]:.3g},{Hplane:.3g}) | "
            f"a={a:.6g} | a_num={a_num:.6g} | d={d:.6g} | "
            f"FN_theo={FN_theo:.3e}"
        )

    ax.set_xlabel(r"radial distance $r$")
    ax.set_ylabel(r"normal pressure $p(r)$")
    ax.set_title(r"Pressure distribution on $\Gamma_c$ (flat punch)")
    ax.legend()
    ax.grid(True)

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.loglog(h_max, errors, "o-", lw=2)

    e = np.array(errors, dtype=float)
    msk = np.isfinite(e) & (e > 0)
    if np.sum(msk) >= 2:
        slope_est = (np.log(e[msk][-1]) - np.log(e[msk][0])) / (
            np.log(h_max[msk][-1]) - np.log(h_max[msk][0])
        )
        xref = np.array([h_max[msk].min(), h_max[msk].max()])
        yref = e[msk][-1] * (xref / h_max[msk][-1]) ** slope_est
        ax1.loglog(xref, yref, "k--", lw=1, label=f"slope ≈ {slope_est:.2f}")

    ax1.set_xlabel("element size $h_{\\max}$")
    ax1.set_ylabel("binned RMS error (scaled by mean pressure)")
    ax1.set_title("Convergence of pressure distribution (flat punch)")
    ax1.grid(True, which="both", ls="--", alpha=0.6)
    ax1.legend()

    plt.tight_layout()
    plt.show()


# run
flat_punch_vs_cofebem(meshes, Scs)
