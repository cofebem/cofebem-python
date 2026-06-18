import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from mpi4py import MPI
from dolfinx.io import gmsh, VTKFile
from dolfinx.fem import functionspace, Function
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import locate_dofs_topological

from cofebem.contact.lcp_solvers.lemke import lemkelcp


# ------------------------------------------------------------
# Helpers (NO class)
# ------------------------------------------------------------
def flat_punch_gap_geom(points_xyz, center_xy, a, z_punch):
    """
    GEOMETRIC gap for a rigid flat punch (bottom plane z=z_punch) above a surface point x.
    We only enforce contact constraints for points under the punch footprint r<=a.
    For those points, gap is signed distance along +z (plane normal):
        g = z_punch - z_point
    => if punch is below the surface (penetration), g < 0.
    Returns:
      g_inside: (N_in,) gap values for points inside footprint
      inside_mask: (N,) boolean mask
    """
    pts = np.asarray(points_xyz, float)
    cx, cy = map(float, center_xy)
    r = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    inside = r <= float(a)
    g = float(z_punch) - pts[:, 2]  # negative if punch penetrates
    return g[inside], inside


def p_boussinesq_flat_punch(r, a, E_star, delta, eps=1e-14):
    """
    Boussinesq solution for rigid flat circular punch (frictionless):
      p(r) = (E* delta) / (pi sqrt(a^2 - r^2))   for 0 <= r < a
      p(r) = 0                                  for r >= a
    """
    r = np.asarray(r, dtype=float)
    p = np.zeros_like(r)

    inside = r < a
    denom = np.sqrt(np.maximum(a * a - r[inside] * r[inside], eps))
    p[inside] = (E_star * float(delta)) / (np.pi * denom)
    return p


def p_punch_fit_model_fixed_a(r, c, a, eps=1e-14):
    """
    Fit model with fixed a:
      p(r) = c / sqrt(a^2 - r^2) for r<a, else 0
    where c should be (E*delta)/pi in theory.
    """
    r = np.asarray(r, dtype=float)
    val = c / np.sqrt(np.maximum(a * a - r * r, eps))
    return np.where(r < a, val, 0.0)


# ------------------------------------------------------------
# Load meshes and compliance matrices
# ------------------------------------------------------------
mesh0= gmsh.read_from_msh(
    "./cofebem/mesh/smart_Hertz0.msh", MPI.COMM_WORLD, 0, gdim=3
).mesh
mesh1 = gmsh.read_from_msh(
    "./cofebem/mesh/smart_Hertz1.msh", MPI.COMM_WORLD, 0, gdim=3
).mesh
mesh2 = gmsh.read_from_msh(
    "./cofebem/mesh/smart_Hertz2.msh", MPI.COMM_WORLD, 0, gdim=3
).mesh
mesh3 = gmsh.read_from_msh(
    "./cofebem/mesh/smart_Hertz3.msh", MPI.COMM_WORLD, 0, gdim=3
).mesh
mesh4 = gmsh.read_from_msh(
    "./cofebem/mesh/smart_Hertz4.msh", MPI.COMM_WORLD, 0, gdim=3
).mesh
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
def punch_vs_cofebem(meshes, Scs):
    # ---- geometry ----
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

    # ---- indentation & punch radius ----
    delta = 0.08
    a_theo = 5.0  # punch radius (prescribed)
    z_punch = Hplane - delta  # punch bottom plane position

    # ---- theory curve ----
    r_th = np.linspace(0.0, a_theo + 0.3, 800)
    p_th = p_boussinesq_flat_punch(r_th, a=a_theo, E_star=E_star, delta=delta)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r_th, p_th, "-", label="Boussinesq (flat punch)")

    max_iter = 10000
    errors = []
    h_max = np.array([12.0, 9.6, 7.68, 6.144, 4.9152])

    # Scale for error (avoid edge singularity): average pressure scale
    # Theory: F = 2 E* a delta, average over disk area pi a^2 => p_avg = 2E*delta/(pi a)
    p_avg_scale = 2.0 * E_star * delta / (np.pi * a_theo)

    for i, mesh in enumerate(meshes):
        V = functionspace(mesh, (element_type, element_degree, (tdim,)))

        Gamma_c = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_c_selector)
        Ic = locate_dofs_topological(V, fdim, Gamma_c)

        # (Keeping your "ignore indexing" assumption here)
        Gamma_c_x = mesh.geometry.x[Ic].reshape(-1, tdim)

        Sc_dense = Scs[i]

        # ------------------------------------------------------------
        # GEOMETRIC GAP + GEOMETRIC ACTIVE SET (no 1e6 hack)
        # ------------------------------------------------------------
        g_in, inside = flat_punch_gap_geom(
            Gamma_c_x, center_xy=center_xy, a=a_theo, z_punch=z_punch
        )

        idx_in = np.flatnonzero(inside)
        if idx_in.size == 0:
            print(f"[mesh {i}] no nodes under punch footprint.")
            errors.append(np.nan)
            continue

        # Reduce the LCP to footprint nodes only
        S_in = Sc_dense[np.ix_(idx_in, idx_in)]

        # Solve contact LCP on inside nodes only
        p_in, _, err_hist = lemkelcp(S_in, g_in, max_iter)

        # Expand back to full vector (outside footprint => no constraint => p=0)
        p_full = np.zeros(Gamma_c_x.shape[0], dtype=float)
        p_full[idx_in] = p_in

        # ---- Lumped nodal area -> pressure ----
        mesh.topology.create_connectivity(fdim, 0)
        facet2verts = mesh.topology.connectivity(fdim, 0)
        area_node = np.zeros(mesh.geometry.x.shape[0], dtype=np.float64)

        for facet in Gamma_c:
            verts = facet2verts.links(facet)
            x0v, x1v, x2v = mesh.geometry.x[verts]
            area_f = 0.5 * np.linalg.norm(np.cross(x1v - x0v, x2v - x0v))
            share = area_f / 3.0
            for v in verts:
                area_node[v] += share

        p_num = p_full / area_node[Ic]

        # ---- radii and numerical contact radius ----
        r_xy = np.linalg.norm(Gamma_c_x[:, :2] - center_xy[None, :], axis=1)
        contact_nodes = p_num > 0.0
        a_num = r_xy[contact_nodes].max() if np.any(contact_nodes) else 0.0

        # ---- theory at nodes (for export / nodal error) ----
        p_theo_nodes = p_boussinesq_flat_punch(
            r_xy, a=a_theo, E_star=E_star, delta=delta
        )

        # Normalized squared error (normalized by mean pressure scale)
        err_r = (np.abs(p_num - p_theo_nodes) ** 2) / (p_avg_scale**2)

        # ---- Export to VTK ----
        p_fenics_num = Function(V)
        p_fenics_num.name = "p_num"
        p_fenics_theo = Function(V)
        p_fenics_theo.name = "p_boussinesq"
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

        with VTKFile(mesh.comm, f"punch_smart{i}.pvd", "w") as vtk:
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

        # ---- Fit amplitude ONLY, using a_num (fixed) ----
        edge_cut = 0.97
        fit_mask = (
            np.isfinite(p_avg)
            & np.isfinite(r_mid)
            & (p_avg > 0.0)
            & (r_mid >= 0.0)
            & (r_mid <= edge_cut * a_num)
        )

        r_fit = r_mid[fit_mask]
        p_fit_data = p_avg[fit_mask]

        if r_fit.size < 10:
            print(f"[mesh {i}] not enough bins for fit (got {r_fit.size}).")
            errors.append(np.nan)
            continue

        def model_c_only(r, c):
            return p_punch_fit_model_fixed_a(r, c, a=a_num)

        c_guess = np.nanmedian(
            p_fit_data * np.sqrt(np.maximum(a_num * a_num - r_fit * r_fit, 1e-14))
        )
        c_guess = float(max(c_guess, 1e-16))

        popt, _ = curve_fit(
            model_c_only,
            r_fit,
            p_fit_data,
            p0=[c_guess],
            bounds=([0.0], [np.inf]),
            maxfev=30000,
        )
        c_fit = float(popt[0])
        delta_fit = np.pi * c_fit / E_star

        # ---- Plot fitted curve ----
        r_plot = np.linspace(0.0, a_num, 600)
        p_plot = p_punch_fit_model_fixed_a(r_plot, c_fit, a=a_num)
        ax.plot(r_plot, p_plot, "-", alpha=1.0, label=f"COFEBEM fit mesh_{i}")

        # ---- Error vs analytical reference (avoid edge singularity) ----
        a_cmp = min(a_num, a_theo)
        r_cmp_max = 0.97 * a_cmp
        r_cmp = np.linspace(0.0, r_cmp_max, 600) if r_cmp_max > 0 else None

        if r_cmp is None:
            errors.append(np.nan)
        else:
            p_fit_cmp = p_punch_fit_model_fixed_a(r_cmp, c_fit, a=a_num)
            p_ref_cmp = p_boussinesq_flat_punch(
                r_cmp, a=a_theo, E_star=E_star, delta=delta
            )
            err = np.linalg.norm((p_fit_cmp - p_ref_cmp) / p_avg_scale) / np.sqrt(
                r_cmp.size
            )
            errors.append(err)

        FN_theo = 2.0 * E_star * a_theo * delta
        print(
            f"Done mesh {i} | a_theo={a_theo:.6g} | a_num={a_num:.6g} | "
            f"delta={delta:.6g} | delta_fit={delta_fit:.6g} | "
            f"c_fit={c_fit:.3e} (theory c={E_star*delta/np.pi:.3e}) | FN_theo={FN_theo:.3e}"
        )

    ax.set_xlabel(r"radial distance $r$")
    ax.set_ylabel(r"normal pressure $p(r)$")
    ax.set_title(
        r"Pressure distribution on $\Gamma_c$ (flat punch) — fitted to COFEBEM"
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
    ax1.set_ylabel("binned RMS error (scaled by $\\bar p$)")
    ax1.set_title("Convergence of fitted pressure distribution (flat punch)")
    ax1.grid(True, which="both", ls="--", alpha=0.6)
    ax1.legend()

    plt.tight_layout()
    plt.show()


# run
punch_vs_cofebem(meshes, Scs)
