import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
)
from dolfinx.io import VTKFile, gmshio

from mpi4py import MPI
from petsc4py import PETSc

from cofebem.contact.rigid_indenters import parabolic
from cofebem.contact.lcp_solvers.ccg import CCG
from cofebem.contact.lcp_solvers.lemke import lemkelcp

# ------------------------------------------------------------------------------
# Physical / geometric parameters for Hertz sphere-plane contact
# ------------------------------------------------------------------------------

# Plate dimensions (same as in your hertz_vs_cofebem)
W = 40.0
H = 20.0

E = 1.0e9
nu = 0.3
E_star = E / (1.0 - nu**2)

# Sphere radius and indentation depth (Hertz)
R = 100.0
delta = 0.08

# Contact center in the plate plane
contact_center = np.array([W / 2.0, W / 2.0])

tdim = 3
fdim = 2

tol = 1.0e-5

element_type = "Lagrange"
element_degree = 1

# ------------------------------------------------------------------------------
# Exact Hertz pressure on the top surface z = H
# ------------------------------------------------------------------------------


def p_hertz_analytic(X):
    """
    X: (N, 3) array of node/facet-center coordinates on the contact surface.
    Uses Hertz solution for a rigid sphere (radius R) on elastic half-space.
    """
    # Shift to contact center in (x, y)
    x = X[:, 0] - contact_center[0]
    y = X[:, 1] - contact_center[1]

    a = np.sqrt(R * delta)
    p0 = 2.0 * E_star * a / (np.pi * R)

    r2 = x**2 + y**2
    p = np.zeros_like(x)

    inside = r2 <= a**2 + 1.0e-14
    p[inside] = p0 * np.sqrt(1.0 - r2[inside] / a**2)

    return p


def err_relative(p, p_ref):
    """Relative L2 error based on max Hertz pressure p0 (same scaling for all meshes)."""
    a = np.sqrt(R * delta)
    p0 = 2.0 * E_star * a / (np.pi * R)
    return np.linalg.norm(p - p_ref) / p0


# ------------------------------------------------------------------------------
# Load Hertz meshes and precomputed Sc matrices
# ------------------------------------------------------------------------------

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

# Characteristic mesh size (you already had this for the Hertz study)
h_max = np.array([12.0, 9.6, 7.68, 6.144, 4.9152])

# ------------------------------------------------------------------------------
# Contact boundary selector: top face z = H
# ------------------------------------------------------------------------------


def Gamma_c_selector(x):
    return np.isclose(x[2], H, atol=tol)


# ------------------------------------------------------------------------------
# Compute P1P1, P0P0, P1P0 errors with Hertz meshes and Scs
# ------------------------------------------------------------------------------

errs_p1p1 = []
errs_p1p0 = []
errs_p0p0 = []

# For plotting the profile on the finest mesh
r_p1_last = None
p_p1_last = None
p_analytic_p1_last = None

max_iter = 10000
tolerance = 1.0e-5
error_type = "displacement"
pfactor = 1.0e8

for i, (mesh, Sc_p1p1) in enumerate(zip(meshes, Scs)):
    print(f"\n=== Refinement step {i} ===")

    V = functionspace(mesh, (element_type, element_degree, (tdim,)))

    # Contact facets
    Gamma_c = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_c_selector)

    mesh.topology.create_connectivity(fdim, 0)
    facet_to_vertices = mesh.topology.connectivity(fdim, 0)

    # Contact nodes (same Ic logic as your original code)
    Ic = locate_dofs_topological(V, fdim, Gamma_c)
    Gamma_c_x = mesh.geometry.x[Ic].reshape(-1, tdim)

    Ne = len(Gamma_c)  # number of contact facets
    Nc = len(Ic)  # number of contact nodes
    N_vertices = mesh.geometry.x.shape[0]

    # ---- Build lumped nodal areas and facet areas on Γ_c ----
    A_elem = np.zeros(Ne, dtype=np.float64)
    A_node = np.zeros(N_vertices, dtype=np.float64)

    for e_idx, f in enumerate(Gamma_c):
        vs = facet_to_vertices.links(f)  # vertices of facet f
        coords = mesh.geometry.x[vs]  # shape (3, 3) for triangles

        v0, v1, v2 = coords
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        A_elem[e_idx] = area

        # Lump area equally to the three vertices
        share = area / 3.0
        for v in vs:
            A_node[v] += share

    # ---- Build Pi_u_to_p: nodal P1 -> facet P0 projection on contact ----
    vertex_to_local = {v: i for i, v in enumerate(Ic)}
    Pi_u_to_p = np.zeros((Ne, Nc), dtype=np.float64)

    for e_idx, f in enumerate(Gamma_c):
        vs = facet_to_vertices.links(f)
        local_ids = [vertex_to_local[v] for v in vs if v in vertex_to_local]
        nv = len(local_ids)

        if nv == 0:
            continue

        w = 1.0 / nv
        for lid in local_ids:
            Pi_u_to_p[e_idx, lid] = w

    # ------------------------------------------------------------------------------
    # P1P1 CONTACT: Solve LCP on nodal Sc_p1p1
    # ------------------------------------------------------------------------------

    gap_p1p1 = (
        parabolic(
            Gamma_c_x[:, 0],
            Gamma_c_x[:, 1],
            contact_center[0],
            contact_center[1],
            R,
            np.full_like(Gamma_c_x[:, 2], H - delta),
        )
        - Gamma_c_x[:, 2]
    )

    # Initial guess (as in your Hertz script)
    p0_init = np.maximum(-gap_p1p1, 0.0) * 1.0e16

    f_p1, _, _ = lemkelcp(Sc_p1p1, gap_p1p1, max_iter)

    # Nodal pressure (divide by nodal areas at contact nodes)
    p_p1 = f_p1 / A_node[Ic]

    # Exact Hertz pressure at contact nodes
    p_analytic_p1 = p_hertz_analytic(Gamma_c_x)

    err_p1p1 = err_relative(p_p1, p_analytic_p1)
    errs_p1p1.append(err_p1p1)

    print(f"  Relative error P1P1 = {err_p1p1:.6e}")

    # Store data for profile plot on the finest mesh
    if i == len(meshes) - 1:
        r_p1_last = np.linalg.norm(Gamma_c_x[:, :2] - contact_center, axis=1)
        p_p1_last = p_p1.copy()
        p_analytic_p1_last = p_analytic_p1.copy()

    # ------------------------------------------------------------------------------
    # P0P0 CONTACT: Compress Sc with Pi_u_to_p and solve on facets
    # ------------------------------------------------------------------------------

    Sc_p0p0 = Pi_u_to_p @ Sc_p1p1 @ Pi_u_to_p.T  # (Ne x Ne)

    # Facet centers
    centers = np.zeros((Ne, 3), dtype=np.float64)
    for e_idx, f in enumerate(Gamma_c):
        vs = facet_to_vertices.links(f)
        coords = mesh.geometry.x[vs]
        centers[e_idx, :] = coords.mean(axis=0)

    # Gap on elements (average 3-point gap on each facet)
    gap_elem = np.zeros(Ne, dtype=np.float64)
    z0_local = H - delta

    for e_idx, f in enumerate(Gamma_c):
        vs = facet_to_vertices.links(f)
        coords = mesh.geometry.x[vs]

        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        z_coords = coords[:, 2]

        z_ind_pts = parabolic(
            x_coords,
            y_coords,
            contact_center[0],
            contact_center[1],
            R,
            np.full(len(x_coords), z0_local),
        )
        gaps_pts = z_ind_pts - z_coords

        gap_elem[e_idx] = gaps_pts.mean()

    # Solve LCP on facet-based Sc_p0p0
    f_p0, _, _ = lemkelcp(Sc_p0p0, gap_elem, max_iter)
    if f_p0 is None:
        print("LCP solver failed for P0P0 contact!")
        errs_p0p0.append(np.nan)
        errs_p1p0.append(np.nan)
        print(f"  Skipping P0P0 and P1P0 error computation for refinement step {i}.")
        continue
    # Facet pressures
    p_p0 = np.zeros_like(f_p0)
    mask_elem = A_elem > 1.0e-14
    p_p0[mask_elem] = f_p0[mask_elem] / A_elem[mask_elem]

    # Exact Hertz pressure at facet centers
    p_analytic_p0 = p_hertz_analytic(centers)

    err_p0p0 = err_relative(p_p0, p_analytic_p0)
    errs_p0p0.append(err_p0p0)
    print(f"  Relative error P0P0 = {err_p0p0:.6e}")

    # ------------------------------------------------------------------------------
    # P1P0 CONTACT: Project facet pressure back to nodes and compare
    # ------------------------------------------------------------------------------

    p_p1p0 = Pi_u_to_p.T @ p_p0  # nodal P1 from facet P0
    err_p1p0 = err_relative(p_p1p0, p_analytic_p1)
    errs_p1p0.append(err_p1p0)

    print(f"  Relative error P1P0 = {err_p1p0:.6e}")

print("\nAll refinement steps done.")

errs_p1p1 = np.array(errs_p1p1)
errs_p1p0 = np.array(errs_p1p0)
errs_p0p0 = np.array(errs_p0p0)

# ------------------------------------------------------------------------------
# Plot: error vs mesh size h
# ------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(h_max, errs_p1p1, "bo--", label="err P1P1", markersize=6, linewidth=2)
ax.plot(h_max, errs_p1p0, "ys--", label="err P1P0", markersize=6, linewidth=2)
ax.plot(h_max, errs_p0p0, "gx--", label="err P0P0", markersize=6, linewidth=2)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("h_max (element size)", fontsize=10)
ax.set_ylabel("relative error", fontsize=10)
ax.set_title("Hertz: comparison of traction in P1/P0", fontsize=14)

ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# Plot: radial profile p(r) on the finest mesh (P1P1)
# ------------------------------------------------------------------------------

if r_p1_last is not None:
    idx_sorted = np.argsort(r_p1_last)
    r_sorted = r_p1_last[idx_sorted]
    p_num_sorted = p_p1_last[idx_sorted]
    p_analytic_sorted = p_analytic_p1_last[idx_sorted]

    fig2, ax2 = plt.subplots(figsize=(6, 4))

    ax2.plot(r_sorted, p_analytic_sorted, "-", label="p_analytic (Hertz)", linewidth=2)
    ax2.plot(r_sorted, p_num_sorted, "o", label="p_P1 (CCG)", markersize=4)

    ax2.set_xlabel("r (radius in contact plane)", fontsize=10)
    ax2.set_ylabel("pressure p", fontsize=10)
    ax2.set_title("Hertz pressure profile p(r) at P1 contact nodes", fontsize=14)

    ax2.grid(True, linestyle="--", linewidth=0.5)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
