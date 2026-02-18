import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from mpi4py import MPI
from dolfinx.io import gmshio, VTKFile
from dolfinx.fem import functionspace, Function
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import locate_dofs_topological

from cofebem.contact.lcp_solvers.lemke import lemkelcp
from cofebem.bodies.cone_indenter import Cone


def cone_gap_on_plane_xy(xy, z_plane, apex=(0.0, 0.0, 0.0), tan_gamma=1.0):

    xy = np.asarray(xy, dtype=float)
    if xy.ndim == 1:
        xy = xy[np.newaxis, :]

    xa, ya, za = map(float, apex)
    dx = xy[:, 0] - xa
    dy = xy[:, 1] - ya
    r = np.sqrt(dx * dx + dy * dy)

    z_surf = za + float(tan_gamma) * r
    return z_surf - float(z_plane)


def p_cone_love_popov(r, p0, a, eps=1e-14):
    """
    Love (1939) / Popov MDR summary for rigid cone indentation:
      p(r) = p0 * arcosh(a/r)   for 0 < r <= a
      p(r) = 0                 for r > a
    with p0 = 0.5 * E* * tan(gamma)
    """
    r = np.asarray(r, dtype=float)
    p = np.zeros_like(r)

    inside = r <= a
    rr = np.maximum(r[inside], eps)

    p[inside] = p0 * np.arccosh(a / rr)
    return p


def p_love_fit_model_fixed_a(r, p0, a, eps=1e-12):
    """
    Fit model with fixed 'a':
      p(r) = p0 * arcosh(a/r) for 0<r<=a, else 0.
    """
    r = np.asarray(r, dtype=float)
    rr = np.maximum(r, eps)
    # ensure a/rr >= 1 for arcosh
    arg = np.maximum(a / rr, 1.0)
    val = p0 * np.arccosh(arg)
    return np.where(r <= a, val, 0.0)


# ------------------------------------------------------------
# Load meshes and compliance matrices
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
# Main comparison
# ------------------------------------------------------------
def cone_vs_cofebem(meshes, Scs):
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

    # ---- indentation ----
    delta = 0.15

    Rcone = 80
    Hcone = 5

    cone = Cone(np.array([W / 2, W / 2, Hplane + Hcone - delta]), Rcone, Hcone)

    tan_theta = np.tan((np.pi / 2) - np.arctan(Rcone / Hcone))
    a_theo = 2 * delta / (np.pi * tan_theta)

    # Popov/Love constants (theoretical)
    p0_theo = 0.5 * E_star * tan_theta
    FN_theo = 0.5 * np.pi * E_star * tan_theta * a_theo**2

    # Theory curve for plot
    r_th = np.linspace(0.0, a_theo + 0.3, 800)
    p_th = p_cone_love_popov(r_th, p0_theo, a_theo)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r_th, p_th, "-", label="Love (cone)")

    max_iter = 10000
    errors = []
    h_max = np.array([12.0, 9.6, 7.68, 6.144, 4.9152])

    for i, mesh in enumerate(meshes):
        V = functionspace(mesh, (element_type, element_degree, (tdim,)))

        Gamma_c = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_c_selector)
        Ic = locate_dofs_topological(V, fdim, Gamma_c)
        Gamma_c_x = mesh.geometry.x[Ic].reshape(-1, tdim)

        Sc_dense = Scs[i]

        # ---- GAP ----
        g = cone.gap(Gamma_c_x)
        print(cone.gap(np.array([20, 20, 20])))
        # ---- Solve contact LCP ----
        f_lemke, _, _ = lemkelcp(Sc_dense, g, max_iter)

        # ---- Lumped nodal area  ----
        facet2verts = mesh.topology.connectivity(fdim, 0)
        area_node = np.zeros(mesh.geometry.x.shape[0], dtype=np.float64)

        for facet in Gamma_c:
            verts = facet2verts.links(facet)
            x0v, x1v, x2v = mesh.geometry.x[verts]
            area_f = 0.5 * np.linalg.norm(np.cross(x1v - x0v, x2v - x0v))
            share = area_f / 3.0
            for v in verts:
                area_node[v] += share

        p_num = f_lemke / area_node[Ic]

        # ---- radii of contact nodes ----
        r_xy = np.linalg.norm(Gamma_c_x[:, :2] - center_xy[None, :], axis=1)
        contact_nodes = p_num > 0.0
        a_num = r_xy[contact_nodes].max() if np.any(contact_nodes) else 0.0

        # ---- Theory at nodes ----
        p_theo_nodes = p_cone_love_popov(r_xy, p0_theo, a_theo)

        # ---- Error field at nodes (scaled by p0_theo) ----
        err_r = (np.abs(p_num - p_theo_nodes) ** 2) / (p0_theo**2)

        # ---- Export to VTK (store in z-component) ----
        p_fenics_num = Function(V)
        p_fenics_num.name = "p_num"
        p_fenics_theo = Function(V)
        p_fenics_theo.name = "p_love"
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

        with VTKFile(mesh.comm, f"cone_smart{i}.pvd", "w") as vtk:
            vtk.write_mesh(mesh)
            vtk.write_function([p_fenics_num, p_fenics_theo, err_fenics], 0)

        # ---- Radial averaging (bins) ----
        if a_num <= 0.0:
            print(f"[mesh {i}] no contact detected (a_num=0).")
            errors.append(np.nan)
            continue

        r_contact = r_xy[contact_nodes]
        p_contact = p_num[contact_nodes]

        nbins = 300
        bins = np.linspace(0.0, a_num, nbins + 1)
        r_mid = 0.5 * (bins[:-1] + bins[1:])
        p_avg = np.full(nbins, np.nan)

        for k in range(nbins):
            m = (r_contact >= bins[k]) & (r_contact < bins[k + 1])
            if np.any(m):
                p_avg[k] = p_contact[m].mean()

        # ---- Fit p0 ONLY, using a_num (fixed) ----
        # Avoid the r->0 singular region in the fit
        r_min_fit = 0.03 * a_num
        fit_mask = (
            np.isfinite(p_avg)
            & np.isfinite(r_mid)
            & (p_avg > 0.0)
            & (r_mid >= r_min_fit)
            & (r_mid <= a_num)
        )

        r_fit = r_mid[fit_mask]
        p_fit_data = p_avg[fit_mask]

        if r_fit.size < 10:
            print(f"[mesh {i}] not enough bins for fit (got {r_fit.size}).")
            errors.append(np.nan)
            continue

        # model with fixed a=a_num
        def model_p0_only(r, p0):
            return p_love_fit_model_fixed_a(r, p0, a=a_num)

        # good-ish initial guess (use max value / arcosh(a/rmin))
        denom = np.arccosh(max(a_num / max(r_fit.min(), 1e-12), 1.0 + 1e-12))
        p0_guess = float(np.nanmax(p_fit_data) / max(denom, 1e-12))

        popt, _ = curve_fit(
            model_p0_only,
            r_fit,
            p_fit_data,
            p0=[p0_guess],
            bounds=([0.0], [np.inf]),
            maxfev=30000,
        )
        p0_fit = float(popt[0])

        # ---- Plot fitted curve (smooth) ----
        r_plot = np.linspace(0.0, a_num, 600)
        p_plot = p_love_fit_model_fixed_a(r_plot, p0_fit, a=a_num)

        ax.plot(r_plot, p_plot, "-", alpha=1.0, label=f"COFEBEM fit mesh_{i}")

        # ---- Error vs analytical reference (choose a common domain) ----
        # Compare fitted curve to Love curve using *theoretical* parameters on r in [r_min_fit, min(a_num, a_theo)]
        a_cmp = min(a_num, a_theo)
        r_cmp = np.linspace(r_min_fit, a_cmp, 600) if a_cmp > r_min_fit else None

        if r_cmp is None:
            errors.append(np.nan)
        else:
            p_fit_cmp = p_love_fit_model_fixed_a(r_cmp, p0_fit, a=a_num)
            p_ref_cmp = p_cone_love_popov(r_cmp, p0_theo, a_theo)
            err = np.linalg.norm((p_fit_cmp - p_ref_cmp) / p0_theo) / np.sqrt(
                r_cmp.size
            )
            errors.append(err)

        print(
            f"Done mesh {i} | a_theo={a_theo:.6g} | a_num={a_num:.6g} | "
            f"tan(theta)={tan_theta:.6g} | p0_theo={p0_theo:.3e} | p0_fit={p0_fit:.3e} | FN_theo={FN_theo:.3e}"
        )

    ax.set_xlabel(r"radial distance $r$")
    ax.set_ylabel(r"normal pressure $p(r)$")
    ax.set_title(
        r"Pressure distribution on $\Gamma_c$ (cone, Love) — fitted to COFEBEM"
    )
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
    ax1.set_ylabel("binned RMS error (scaled by $p_0$)")
    ax1.set_title("Convergence of fitted pressure distribution (cone, Love)")
    ax1.grid(True, which="both", ls="--", alpha=0.6)
    ax1.legend()

    plt.tight_layout()
    plt.show()


# run
cone_vs_cofebem(meshes, Scs)
