"""
Compare the penalty contact formulation against the flexibility
(Schur-complement / LCP) formulation for a rigid sphere indenting an
elastic block.

Two studies are run back to back, sharing the same mesh, material and
Dirichlet BC:

  1. Single contact: one indentation step straight to `delta_final`.
     CPU time of each method and the displacement error of the penalty
     solution relative to the flexibility solution.

  2. Incremental indentation: `nsteps` load steps up to `delta_final`.
     The flexibility method's Schur complement (Sc) only depends on the
     elastic operator, not on the indentation depth, so it is built once
     (before the loop) and reused at every step -- its cost is counted
     once, not once per step, when tallying CPU time.
"""

import time

import numpy as np
import matplotlib.pyplot as plt
import ufl

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.mesh import create_box, locate_entities_boundary, meshtags, CellType
from dolfinx.fem import (
    Constant,
    Expression,
    Function,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem

from cofebem.bodies.sphere_indenter import Sphere
from cofebem.fenics.contact import Contact

comm = MPI.COMM_WORLD
if comm.size != 1:
    raise RuntimeError("This script is written for serial execution.")

# ---------------------------------------------------------------------------
# Geometry / material / mesh -- shared by both methods
# ---------------------------------------------------------------------------
Lx, Ly, H = 1.0, 1.0, 1.0
nx, ny, nz = 20, 20, 20

dirichlet_id, neumann_id, contact_id = 1, 2, 3

E, nu = 1.0e9, 0.3
kpen = 1.0e12

R = 0.35
delta_final = 0.1
nsteps = 6

mesh = create_box(
    comm,
    [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, H])],
    [nx, ny, nz],
    cell_type=CellType.tetrahedron,
)
tdim = mesh.topology.dim
fdim = tdim - 1
gdim = mesh.geometry.dim
mesh.topology.create_connectivity(fdim, 0)

facets_bottom = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[2], 0.0))
facets_top = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[2], H))
facets_lateral = locate_entities_boundary(
    mesh,
    fdim,
    lambda x: np.logical_or.reduce(
        (
            np.isclose(x[0], 0.0),
            np.isclose(x[0], Lx),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], Ly),
        )
    ),
)

facet_indices = np.hstack([facets_bottom, facets_lateral, facets_top]).astype(np.int32)
facet_values = np.hstack(
    [
        np.full(len(facets_bottom), dirichlet_id, dtype=np.int32),
        np.full(len(facets_lateral), neumann_id, dtype=np.int32),
        np.full(len(facets_top), contact_id, dtype=np.int32),
    ]
).astype(np.int32)
perm = np.argsort(facet_indices)
facet_tags = meshtags(mesh, fdim, facet_indices[perm], facet_values[perm])

contact_center = np.array([0.5 * Lx, 0.5 * Ly])

V = functionspace(mesh, ("Lagrange", 1, (gdim,)))

mu = Constant(mesh, PETSc.ScalarType(E / (2.0 * (1.0 + nu))))
lmbda = Constant(mesh, PETSc.ScalarType(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))))
f_body = Constant(mesh, PETSc.ScalarType((0.0, 0.0, 0.0)))
t_neu = Constant(mesh, PETSc.ScalarType((0.0, 0.0, 0.0)))


def eps(w):
    return ufl.sym(ufl.grad(w))


def sigma(w):
    return lmbda * ufl.tr(eps(w)) * ufl.Identity(gdim) + 2.0 * mu * eps(w)


ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

facets_D = facet_tags.indices[facet_tags.values == dirichlet_id]
dofs_D = locate_dofs_topological(V, fdim, facets_D)
uD = Constant(mesh, PETSc.ScalarType((0.0, 0.0, 0.0)))
bc = dirichletbc(uD, dofs_D, V)

print(
    f"Mesh: {nx}x{ny}x{nz}, {V.dofmap.index_map.size_global * gdim} dofs, "
    f"{len(facets_top)} contact facets"
)

# ---------------------------------------------------------------------------
# Penalty formulation (nonlinear, Newton/SNES per load step)
# ---------------------------------------------------------------------------
u_pen = Function(V)
u_pen.name = "u_penalty"
v = ufl.TestFunction(V)

x = ufl.SpatialCoordinate(mesh)
k = Constant(mesh, PETSc.ScalarType(kpen))

xc0 = Constant(mesh, PETSc.ScalarType(contact_center[0]))
yc0 = Constant(mesh, PETSc.ScalarType(contact_center[1]))
Rc = Constant(mesh, PETSc.ScalarType(R))
z0 = Constant(mesh, PETSc.ScalarType(H))

r2 = (x[0] - xc0) ** 2 + (x[1] - yc0) ** 2
zobs = z0 + Rc - ufl.sqrt(ufl.max_value(Rc**2 - r2, 0.0))
gap = zobs - (x[2] + u_pen[2])
pn = k * ufl.conditional(ufl.lt(gap, 0.0), -gap, 0.0)

F = (
    ufl.inner(sigma(u_pen), eps(v)) * ufl.dx
    - ufl.dot(f_body, v) * ufl.dx
    - ufl.dot(t_neu, v) * ds(neumann_id)
    + pn * v[2] * ds(contact_id)
)
J = ufl.derivative(F, u_pen)

petsc_options_pen = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_atol": 1e-3,
    "snes_rtol": 1e-4,
    "snes_max_it": 100,
    "ksp_type": "preonly",
    "pc_type": "lu",
}

problem_pen = NonlinearProblem(
    F,
    u_pen,
    bcs=[bc],
    J=J,
    petsc_options=petsc_options_pen,
    petsc_options_prefix="penalty_contact_",
)


def solve_penalty_step(
    delta_prev: float, delta_target: float, max_depth: int = 6
) -> float:
    """
    Advance the penalty solution from indentation `delta_prev` to
    `delta_target`. The penalty residual is non-smooth (built from
    `ufl.conditional`), so a single Newton/line-search solve can fail to
    converge on a large jump -- and since SNES does not roll back on
    failure, an unconverged step would silently corrupt the state used by
    every later step. If the solve fails, roll back and bisect the
    increment into two half-steps, recursively.

    Returns the wall-clock time spent, including any failed attempts.
    """
    u_backup = u_pen.x.array.copy()

    set_penalty_depth(delta_target)
    t0 = time.time()
    problem_pen.solve()
    u_pen.x.scatter_forward()
    elapsed = time.time() - t0

    reason = problem_pen.solver.getConvergedReason()
    if reason > 0:
        return elapsed

    if max_depth == 0:
        print(
            f"  WARNING: penalty did not converge at the bisection limit "
            f"(delta={delta_target:.4f}, reason={reason})"
        )
        return elapsed

    print(
        f"  penalty solve failed (delta={delta_target:.4f}, reason={reason}); "
        f"bisecting the load increment"
    )
    u_pen.x.array[:] = u_backup
    delta_mid = 0.5 * (delta_prev + delta_target)
    elapsed += solve_penalty_step(delta_prev, delta_mid, max_depth - 1)
    elapsed += solve_penalty_step(delta_mid, delta_target, max_depth - 1)
    return elapsed


def solve_penalty_once(delta: float) -> tuple[float, int]:
    set_penalty_depth(delta)

    t0 = time.perf_counter()
    problem_pen.solve()
    elapsed = time.perf_counter() - t0

    u_pen.x.scatter_forward()
    reason = problem_pen.solver.getConvergedReason()

    return elapsed, reason


# ---------------------------------------------------------------------------
# Flexibility formulation (linear elasticity + Schur-complement LCP contact)
# ---------------------------------------------------------------------------
tc = Function(V)
tc.name = "contact_traction"

u2 = ufl.TrialFunction(V)
v2 = ufl.TestFunction(V)

a_flex = ufl.inner(sigma(u2), eps(v2)) * ufl.dx
L_flex = (
    ufl.dot(f_body, v2) * ufl.dx
    + ufl.dot(t_neu, v2) * ds(neumann_id)
    + ufl.dot(tc, v2) * ds(contact_id)
)

problem_flex = LinearProblem(
    a_flex,
    L_flex,
    bcs=[bc],
    petsc_options_prefix="flex_contact_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)
u_flex = problem_flex.u
u_flex.name = "u_flexibility"

indenter = Sphere(
    center=np.array([contact_center[0], contact_center[1], H + R]), radius=R
)

t0 = time.time()
contact = Contact(
    mesh=mesh,
    indenter=indenter,
    tc=tc,
    Gamma_c=facets_top,
    ds=ds,
    Gamma_c_id=contact_id,
    problem=problem_flex,
    solver="ccg",
)
t_sc_build = time.time() - t0
print(f"\nSc build (once): {t_sc_build:.3f} s " f"({contact.Sc.shape[0]} contact dofs)")


def set_penalty_depth(delta: float):
    z0.value = PETSc.ScalarType(H - delta)


def set_indenter_position(delta: float):
    indenter.center[2] = (H - delta) + R


def solve_flexibility():
    contact.solve(max_iter=10000, tol=1e-7, p0=1e9)
    contact.apply_contact_forces()
    problem_flex.solve()


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Relative error of `a` w.r.t. reference `b`, ||a-b|| / ||b||."""
    nb = np.linalg.norm(b)
    if nb == 0.0:
        return float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / nb)


# Contact pressure comparison: the penalty pressure `pn` is a UFL expression
# (depends on u_pen) evaluated on the scalar contact space `contact.W`, and
# compared against the flexibility traction `tc`. `dofs_c_W`/`dofs_c_V` are
# index-aligned (dofs_c_V = W_to_V[dofs_c_W]), so no coordinate matching is
# needed. `tc`'s z-component carries an applied-traction sign convention
# (compressive contact => tc_z < 0) opposite to `pn` (>= 0 by construction),
# hence the sign flip below.
pn_expr = Expression(pn, contact.W.element.interpolation_points)
pn_W = Function(contact.W)


def pressure_relative_error() -> float:
    pn_W.interpolate(pn_expr)
    pn_c = pn_W.x.array[contact.dofs_c_W]
    tc_c = -tc.x.array[contact.dofs_c_V]
    return relative_error(pn_c, tc_c)


# ---------------------------------------------------------------------------
# Phase 1: single contact -- one shot straight to delta_final
# ---------------------------------------------------------------------------
print("\n=== Phase 1: single contact (one shot to delta_final) ===")

set_indenter_position(delta_final)

# t_pen_one_shot = solve_penalty_step(0.0, delta_final)
t_pen_one_shot, reason = solve_penalty_once(delta_final)

if reason <= 0:
    print(
        f"Penalty one-shot solve failed: "
        f"time={t_pen_one_shot:.3f}s, reason={reason}"
    )
t0 = time.time()
solve_flexibility()
t_flex_solve_only = time.time() - t0
t_flex_one_shot = t_sc_build + t_flex_solve_only

err_one_shot = relative_error(u_pen.x.array, u_flex.x.array)
pressure_err_one_shot = pressure_relative_error()

print(f"Penalty:      {t_pen_one_shot:.3f} s")
print(
    f"Flexibility:  {t_flex_one_shot:.3f} s  "
    f"(Sc build {t_sc_build:.3f} s + solve {t_flex_solve_only:.3f} s)"
)
print(f"Relative displacement error (penalty vs flexibility): {err_one_shot:.3e}")
print(
    f"Relative contact pressure error (pn vs tc on Gamma_c): "
    f"{pressure_err_one_shot:.3e}"
)

# ---------------------------------------------------------------------------
# Phase 2: incremental indentation, Sc reused (built once, above)
# ---------------------------------------------------------------------------
print("\n=== Phase 2: incremental indentation (Sc reused across steps) ===")

u_pen.x.array[:] = 0.0
u_flex.x.array[:] = 0.0

deltas = delta_final * np.arange(1, nsteps + 1) / nsteps

pen_times = np.zeros(nsteps)
flex_times = np.zeros(nsteps)
errors = np.zeros(nsteps)
pressure_errors = np.zeros(nsteps)

delta_prev = 0.0
for i, delta in enumerate(deltas):
    set_indenter_position(delta)

    # pen_times[i] = solve_penalty_step(delta_prev, delta)

    t_pen, reason = solve_penalty_once(delta_final)

    if reason <= 0:
        print(
            f"Penalty one-shot solve failed: "
            f"time={t_pen_one_shot:.3f}s, reason={reason}"
        )

    delta_prev = delta
    pen_times[i] = t_pen

    t0 = time.time()
    solve_flexibility()
    flex_times[i] = time.time() - t0

    errors[i] = relative_error(u_pen.x.array, u_flex.x.array)
    pressure_errors[i] = pressure_relative_error()

    print(
        f"Step {i + 1}/{nsteps}  delta={delta:.4f}  "
        f"penalty={pen_times[i]:.3f}s  flexibility={flex_times[i]:.3f}s  "
        f"u_error={errors[i]:.3e}  p_error={pressure_errors[i]:.3e}"
    )

pen_cumulative = np.cumsum(pen_times)
flex_cumulative = t_sc_build + np.cumsum(flex_times)

print(f"\nTotal penalty time:      {pen_cumulative[-1]:.3f} s")
print(
    f"Total flexibility time:  {flex_cumulative[-1]:.3f} s "
    f"(Sc build {t_sc_build:.3f} s counted once + "
    f"{np.sum(flex_times):.3f} s solves)"
)

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
steps = np.arange(1, nsteps + 1)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].plot(steps, pen_times, "o-", label="Penalty")
axes[0].plot(steps, flex_times, "s-", label="Flexibility (Sc excluded)")
axes[0].axhline(
    t_sc_build, color="gray", ls="--", label=f"Sc build ({t_sc_build:.2f} s, once)"
)
axes[0].set_xlabel("Load step")
axes[0].set_ylabel("CPU time [s]")
axes[0].set_title("Per-step CPU time")
axes[0].legend()
axes[0].grid(True, ls="--", alpha=0.6)

axes[1].plot(steps, pen_cumulative, "o-", label="Penalty")
axes[1].plot(steps, flex_cumulative, "s-", label="Flexibility (Sc counted once)")
axes[1].set_xlabel("Load step")
axes[1].set_ylabel("Cumulative CPU time [s]")
axes[1].set_title("Cumulative CPU time")
axes[1].legend()
axes[1].grid(True, ls="--", alpha=0.6)

axes[2].semilogy(steps, errors, "o-", color="tab:red")
axes[2].set_xlabel("Load step")
axes[2].set_ylabel("Relative displacement error")
axes[2].set_title("Displacement error (penalty vs flexibility)")
axes[2].grid(True, ls="--", alpha=0.6)

axes[3].semilogy(steps, pressure_errors, "o-", color="tab:purple")
axes[3].set_xlabel("Load step")
axes[3].set_ylabel("Relative contact pressure error")
axes[3].set_title(r"Contact pressure error ($p_n$ vs $t_c$ on $\Gamma_c$)")
axes[3].grid(True, ls="--", alpha=0.6)

plt.tight_layout()
plt.savefig("penalty_vs_flexibility.png", dpi=150)
print("\nSaved plot to penalty_vs_flexibility.png")
