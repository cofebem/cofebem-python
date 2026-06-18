import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import time
import matplotlib.pyplot as plt

from dolfinx.mesh import locate_entities_boundary, meshtags, create_box, CellType
from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import LinearProblem
from ufl import Identity, Measure, TrialFunction, TestFunction, sym, grad, inner, tr, dx

from cofebem.bodies.sphere_indenter import Sphere
from cofebem.contact.lcp_solvers.nnls import lawson_hanson_nnls_lcp, scipy_nnls_lcp
from cofebem.contact.lcp_solvers.ccg import CCG
from cofebem.contact.lcp_solvers.lemke import lemkelcp
from cofebem.contact.lcp_solvers.psor import psor_lcp
from cofebem.fenics.contact import Sc


def ccg_lcp(
    M,
    q,
    z0=None,
    tol=1e-10,
    max_iter=None,
):

    M = np.asarray(M, dtype=float)
    q = np.asarray(q, dtype=float).reshape(-1)

    n = q.size

    if M.shape != (n, n):
        raise ValueError(f"M must have shape ({n}, {n}).")

    if not np.all(np.isfinite(M)) or not np.all(np.isfinite(q)):
        raise ValueError("M and q must contain finite values.")

    if not np.allclose(M, M.T, rtol=1e-12, atol=1e-14):
        raise ValueError("M must be symmetric.")

    if tol <= 0.0:
        raise ValueError("tol must be positive.")

    if max_iter is None:
        max_iter = max(100, 20 * n)

    if z0 is None:
        z = np.zeros(n, dtype=float)
    else:
        z = np.asarray(z0, dtype=float).reshape(-1)

        if z.size != n:
            raise ValueError(f"z0 must contain {n} entries.")

        if not np.all(np.isfinite(z)):
            raise ValueError("z0 must contain finite values.")

        z = np.maximum(z, 0.0)

    residual_scale = max(1.0, np.linalg.norm(q, ord=np.inf))
    zero_tolerance = 10.0 * np.finfo(float).eps

    residual_history = []

    restart = True

    previous_residual = None
    direction = np.zeros(n, dtype=float)

    converged = False
    reason = "maximum number of iterations reached"

    for iteration in range(max_iter + 1):

        w = M @ z + q

        natural_residual = np.minimum(z, w)

        relative_residual = (
            np.linalg.norm(natural_residual, ord=np.inf) / residual_scale
        )

        residual_history.append(relative_residual)

        if relative_residual <= tol:
            converged = True
            reason = "LCP residual below tolerance"
            break

        if iteration == max_iter:
            break

        residual = -w

        outward = (z <= zero_tolerance) & (residual < 0.0)
        residual[outward] = 0.0

        residual_norm_squared = np.dot(residual, residual)

        if residual_norm_squared <= np.finfo(float).tiny:
            reason = "zero feasible residual"
            break

        if restart or previous_residual is None:
            direction = residual.copy()
        else:
            previous_norm_squared = np.dot(
                previous_residual,
                previous_residual,
            )

            if previous_norm_squared <= np.finfo(float).tiny:
                beta = 0.0
            else:
                beta = residual_norm_squared / previous_norm_squared

            direction = residual + beta * direction

            outward = (z <= zero_tolerance) & (direction < 0.0)
            direction[outward] = 0.0

        gradient_direction = np.dot(w, direction)

        if gradient_direction >= 0.0:
            direction = residual.copy()
            gradient_direction = np.dot(w, direction)
            restart = True

        if np.linalg.norm(direction, ord=np.inf) <= np.finfo(float).tiny:
            reason = "zero feasible search direction"
            break

        M_direction = M @ direction
        curvature = np.dot(direction, M_direction)

        if curvature <= 0.0:
            raise ValueError(
                "Non-positive curvature encountered. " "M may not be positive definite."
            )

        alpha_cg = -gradient_direction / curvature

        if alpha_cg <= 0.0 or not np.isfinite(alpha_cg):
            reason = "invalid classical CG step"
            break

        negative_direction = direction < 0.0

        if np.any(negative_direction):
            alpha_max = np.min(-z[negative_direction] / direction[negative_direction])
        else:
            alpha_max = np.inf

        alpha = min(alpha_cg, alpha_max)

        if alpha < 0.0 or not np.isfinite(alpha):
            reason = "invalid feasible step"
            break

        z = z + alpha * direction

        # z[z < zero_tolerance] = 0.0

        hit_constraint = np.isfinite(alpha_max) and alpha_max <= alpha_cg * (
            1.0 + 1e-14
        )

        restart = hit_constraint
        previous_residual = residual.copy()

    w = M @ z + q
    natural_residual = np.minimum(z, w)

    final_residual = np.linalg.norm(natural_residual, ord=np.inf) / residual_scale

    info = {
        "converged": final_residual <= tol,
        "iterations": iteration,
        "residual": final_residual,
        "primal_violation": max(0.0, -np.min(z)),
        "dual_violation": max(0.0, -np.min(w)),
        "complementarity": np.linalg.norm(z * w, ord=np.inf),
        "residual_history": np.asarray(residual_history),
        "reason": reason,
    }

    return z, w, info


errors_nnls = []
errors_ccg = []
errors_ccg_new = []
errors_lemke = []
errors_psor = []

cpu_times_nnls = []
cpu_times_ccg = []
cpu_times_ccg_new = []
cpu_times_lemke = []
cpu_times_psor = []

mesh_sizes = []
contact_sizes = []

n = 40
k_values = np.linspace(1, n, 10, dtype=int)

for k in k_values:
    print(f"\n--- Running case k = {k} ---")

    # ---------------- Mesh ----------------
    nx = int(k + 1)
    ny = int(k + 1)
    nz = 5

    l = 1.0

    mesh = create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([l, l, l])],
        [nx, ny, nz],
        CellType.hexahedron,
    )
    tdim = mesh.topology.dim
    fdim = tdim - 1

    # ---------------- Material ----------------
    E = 1.0e9
    nu = 0.3
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    # ---------------- Variational forms ----------------
    V = functionspace(mesh, ("Lagrange", 1, (tdim,)))
    u = TrialFunction(V)
    v = TestFunction(V)

    def epsilon(w):
        return sym(grad(w))

    def sigma(w):
        return lmbda * tr(epsilon(w)) * Identity(tdim) + 2 * mu * epsilon(w)

    f_v = Constant(mesh, np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType))
    a = inner(sigma(u), epsilon(v)) * dx
    L = inner(f_v, v) * dx

    # ---------------- Dirichlet BC ----------------
    def Gamma_u_locator(x):
        return np.isclose(x[2], 0.0)

    Gamma_u = locate_entities_boundary(mesh, fdim, Gamma_u_locator)
    Gamma_u_dofs = locate_dofs_topological(V, fdim, Gamma_u)

    u0 = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType)
    bc = dirichletbc(u0, Gamma_u_dofs, V)
    bcs = [bc]

    # ---------------- Neumann BC ----------------
    def Gamma_t_locator(x):
        return np.isclose(x[1], 1.0) & ((x[0] - 0.5) ** 2 + (x[2] - 0.5) ** 2 <= 0.2**2)

    Gamma_t = locate_entities_boundary(mesh, fdim, Gamma_t_locator)
    Gamma_t_id = 1
    Gamma_t_tags = np.full(Gamma_t.shape, Gamma_t_id, dtype=np.int32)

    t0 = Constant(mesh, np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType))

    # ---------------- Contact BC ----------------
    def Gamma_c_locator(x):
        return np.isclose(x[2], 1.0)

    Gamma_c = locate_entities_boundary(mesh, fdim, Gamma_c_locator)
    Gamma_c_id = 2
    Gamma_c_tags = np.full(Gamma_c.shape, Gamma_c_id, dtype=np.int32)
    Gamma_c_dofs = locate_dofs_topological(V, fdim, Gamma_c)

    # Coordinates of contact dofs
    x_dofs = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry.dim))
    xc = x_dofs[Gamma_c_dofs]

    tc = Function(V)
    tc.name = "p_c"

    # ---------------- Setup Neumann and contact contributions ----------------
    facet_indices = np.hstack([Gamma_t, Gamma_c]).astype(np.int32)
    facet_values = np.hstack([Gamma_t_tags, Gamma_c_tags]).astype(np.int32)

    order = np.argsort(facet_indices)
    facet_indices = facet_indices[order]
    facet_values = facet_values[order]

    mt = meshtags(mesh, fdim, facet_indices, facet_values)
    ds = Measure("ds", domain=mesh, subdomain_data=mt)

    L += inner(t0, v) * ds(Gamma_t_id) + inner(tc, v) * ds(Gamma_c_id)

    # ---------------- Setup problem ----------------
    problem = LinearProblem(
        a,
        L,
        petsc_options_prefix="prb_",
        bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    problem.u.name = "u"
    problem.solve()

    # ---------------- Setup indentation scenario ----------------
    delta = 0.1
    R = 0.5

    indenter = Sphere(
        center=np.array([0.5, 0.5, l + R - delta]),
        radius=R,
    )

    g0 = indenter.gap(xc)
    Sc_ = Sc(problem.A, problem.b, mesh.geometry.dim, Gamma_c_dofs).by_sampling()

    # Store sizes for plots
    mesh_sizes.append((nx + 1) * (ny + 1) * (nz + 1))
    contact_sizes.append(len(Gamma_c_dofs))

    # ---------------- Solve LCP with different solvers ----------------
    t0_nnls = time.time()
    w_nnls, _, _ = scipy_nnls_lcp(Sc_, g0, tol=1e-7, max_iter=10000)
    t1_nnls = time.time()
    t_nnls = t1_nnls - t0_nnls
    cpu_times_nnls.append(t_nnls)

    t0_ccg = time.time()
    w_ccg, _, _ = CCG(
        Sc_, g0, max_iter=10000, tol=1e-7, err_type="displacement", p0=1e9
    ).solve()
    t1_ccg = time.time()
    t_ccg = t1_ccg - t0_ccg
    cpu_times_ccg.append(t_ccg)

    t0_ccg_new = time.time()
    w_ccg_new, _, _ = ccg_lcp(Sc_, g0, max_iter=10000, tol=1e-7)
    t1_ccg_new = time.time()
    t_ccg_new = t1_ccg_new - t0_ccg_new
    cpu_times_ccg_new.append(t_ccg_new)

    t0_lemke = time.time()
    w_lemke, _, _ = lemkelcp(Sc_, g0, maxIter=10000)
    t1_lemke = time.time()
    t_lemke = t1_lemke - t0_lemke
    cpu_times_lemke.append(t_lemke)

    t0_psor = time.time()
    w_psor, _, _, _ = psor_lcp(Sc_, g0, tol=1e-7, max_iter=10000)
    t1_psor = time.time()
    t_psor = t1_psor - t0_psor
    cpu_times_psor.append(t_psor)

    # ---------------- Errors wrt Lemke ----------------
    norm_lemke = np.linalg.norm(w_lemke)
    if norm_lemke < 1e-16:
        error_nnls = np.linalg.norm(w_nnls - w_lemke)
        error_ccg = np.linalg.norm(w_ccg - w_lemke)
        error_ccg_new = np.linalg.norm(w_ccg_new - w_lemke)
        error_psor = np.linalg.norm(w_psor - w_lemke)
        error_lemke = 0.0
    else:
        error_nnls = np.linalg.norm(w_nnls - w_lemke) / norm_lemke
        error_ccg = np.linalg.norm(w_ccg - w_lemke) / norm_lemke
        error_ccg_new = np.linalg.norm(w_ccg_new - w_lemke) / norm_lemke
        error_psor = np.linalg.norm(w_psor - w_lemke) / norm_lemke
        error_lemke = np.linalg.norm(w_lemke - w_lemke) / norm_lemke

    errors_nnls.append(error_nnls)
    errors_ccg.append(error_ccg)
    errors_ccg_new.append(error_ccg_new)
    errors_psor.append(error_psor)
    errors_lemke.append(error_lemke)

    print(f"contact dofs = {len(Gamma_c_dofs)}")
    print(f"NNLS  : time = {t_nnls:.4e} s, error = {error_nnls:.4e}")
    print(f"CCG   : time = {t_ccg:.4e} s, error = {error_ccg:.4e}")
    print(f"CCG v2: time = {t_ccg_new:.4e} s, error = {error_ccg_new:.4e}")
    print(f"Lemke : time = {t_lemke:.4e} s, error = {error_lemke:.4e}")
    print(f"PSOR  : time = {t_psor:.4e} s, error = {error_psor:.4e}")


# =========================
# Plots
# =========================

plt.figure(figsize=(9, 6))
plt.plot(contact_sizes, cpu_times_nnls, "o-", linewidth=2, markersize=7, label="NNLS")
plt.plot(contact_sizes, cpu_times_ccg, "s-", linewidth=2, markersize=7, label="CCG")
plt.plot(
    contact_sizes, cpu_times_ccg_new, "y-", linewidth=2, markersize=7, label="CCG v2"
)
plt.plot(contact_sizes, cpu_times_lemke, "^-", linewidth=2, markersize=7, label="Lemke")
plt.plot(contact_sizes, cpu_times_psor, "d-", linewidth=2, markersize=7, label="PSOR")
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.xlabel("Number of contact DOFs", fontsize=12)
plt.ylabel("CPU time [s]", fontsize=12)
plt.title("CPU time comparison of LCP solvers", fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 6))
plt.semilogy(
    contact_sizes, errors_nnls, "o-", linewidth=2, markersize=7, label="NNLS vs Lemke"
)
plt.semilogy(
    contact_sizes, errors_ccg, "s-", linewidth=2, markersize=7, label="CCG vs Lemke"
)
plt.semilogy(
    contact_sizes,
    errors_ccg_new,
    "y-",
    linewidth=2,
    markersize=7,
    label="CCG v2 vs Lemke",
)
plt.semilogy(
    contact_sizes, errors_psor, "d-", linewidth=2, markersize=7, label="PSOR vs Lemke"
)
# plt.semilogy(
#     contact_sizes, errors_lemke, "^-", linewidth=2, markersize=7, label="Lemke vs Lemke"
# )
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.xlabel("Number of contact DOFs", fontsize=12)
plt.ylabel("Relative error", fontsize=12)
plt.title("Relative error of LCP solvers with respect to Lemke", fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
