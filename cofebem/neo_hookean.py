from pathlib import Path
from time import perf_counter

import basix
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem import (
    Expression,
    Function,
    assemble_vector,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import assemble_matrix, create_matrix
from dolfinx.io import VTKFile
from dolfinx.mesh import CellType, create_box, locate_entities_boundary, meshtags


def sphere_geometry(points, center, radius):
    relative = points - center
    distance = np.linalg.norm(relative, axis=1)
    if np.any(distance <= 1.0e-12 * radius):
        raise ValueError("A contact point is too close to the sphere center.")
    # Normals point toward the sphere; the force on the cube is -fc * n.
    normals = -relative / distance[:, None]
    return distance - radius, normals


def gpcg(apply, q, tol=1.0e-10, max_iter=20000):
    z = np.zeros_like(q)
    active = q >= 0.0
    iterations = 0
    while iterations < max_iter:
        w = apply(z) + q
        if np.linalg.norm(np.minimum(z, w), np.inf) <= tol:
            return z, iterations
        r = np.where(active, 0.0, -w)
        if np.linalg.norm(r, np.inf) <= tol:
            candidates = np.where(active, w, np.inf)
            active[np.argmin(candidates)] = False
            iterations += 1
            continue
        d = r.copy()
        rr = r @ r
        while iterations < max_iter:
            Md = apply(d)
            curvature = d @ Md
            if not np.isfinite(curvature) or curvature <= 0.0:
                raise RuntimeError(
                    "Nonpositive curvature in GPCG; the tangent must be SPD."
                )
            alpha_cg = rr / curvature
            decreasing = d < 0.0
            alpha_max = np.min(-z[decreasing] / d[decreasing], initial=np.inf)
            alpha = min(alpha_cg, alpha_max)
            z += alpha * d
            w += alpha * Md
            iterations += 1
            if alpha_max <= alpha_cg:
                zero_tol = 100.0 * np.finfo(float).eps * max(1.0, np.max(z))
                hit = decreasing & (z <= zero_tol)
                active[hit] = True
                z[hit] = 0.0
                break
            r_new = np.where(active, 0.0, -w)
            if np.linalg.norm(r_new, np.inf) <= tol:
                break
            rr_new = r_new @ r_new
            d = r_new + (rr_new / rr) * d
            d[active] = 0.0
            rr = rr_new
    w = apply(z) + q
    if np.linalg.norm(np.minimum(z, w), np.inf) <= tol:
        return z, iterations
    raise RuntimeError("GPCG did not converge.")


def solve_flexibility(
    K,
    internal_force,
    contact_dofs,
    fixed_dofs,
    gap,
    normals,
    force_scale,
    gap_scale,
    comm,
    tol=1.0e-10,
    max_iter=20000,
):
    # Cholesky and GPCG require an SPD tangent on the unconstrained DOFs.
    K.setOption(PETSc.Mat.Option.SPD, True)
    ksp = PETSc.KSP().create(comm)
    rhs = K.createVecLeft()
    sol = K.createVecRight()
    try:
        ksp.setOperators(K)
        ksp.setType("preonly")
        ksp.getPC().setType("cholesky")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.setErrorIfNotConverged(True)
        start = perf_counter()
        ksp.setUp()
        factor_time = perf_counter() - start

        rhs.array[:] = -internal_force
        rhs.array[fixed_dofs] = 0.0
        ksp.solve(rhs, sol)
        free_increment = sol.array_r.copy()
        q = (
            gap - np.einsum("ij,ij->i", normals, free_increment[contact_dofs])
        ) / gap_scale
        matvecs = 0

        def apply(z):
            nonlocal matvecs
            rhs.set(0.0)
            rhs.array[contact_dofs] = force_scale * z[:, None] * normals
            ksp.solve(rhs, sol)
            matvecs += 1
            return np.einsum("ij,ij->i", normals, sol.array_r[contact_dofs]) / gap_scale

        start = perf_counter()
        z, iterations = gpcg(apply, q, tol=tol, max_iter=max_iter)
        contact_time = perf_counter() - start
        fc = force_scale * z

        # fc is the new total force, not a force increment.
        rhs.array[:] = -internal_force
        rhs.array[contact_dofs] -= fc[:, None] * normals
        rhs.array[fixed_dofs] = 0.0
        ksp.solve(rhs, sol)
        increment = sol.array_r.copy()
        if not np.all(np.isfinite(increment)):
            raise RuntimeError("The displacement increment contains nonfinite values.")
        return (
            increment,
            fc,
            {
                "iterations": iterations,
                "matvecs": matvecs,
                "factor_s": factor_time,
                "contact_s": contact_time,
            },
        )
    finally:
        sol.destroy()
        rhs.destroy()
        ksp.destroy()


def main():
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("This script is written for serial execution.")
    if np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
        raise RuntimeError("This script requires real, double-precision PETSc.")

    Lbox = 1.0
    nx, ny, nz = 50, 50, 5
    E, nu = 1.0e9, 0.3
    radius, delta = 0.5, 0.1
    quadrature_degree = 4
    max_iterations = 80
    max_backtracks = 20
    newton_tol, contact_tol = 1.0e-8, 1.0e-8
    gpcg_tol, gpcg_max_iter = 1.0e-10, 20000
    minimum_jacobian = 1.0e-6
    output_dir = Path("results/neo_hookean_contact")

    if E <= 0.0 or not -1.0 < nu < 0.5:
        raise ValueError("Invalid elastic parameters.")
    if radius <= 0.0 or not 0.0 < delta < min(radius, Lbox):
        raise ValueError("Require 0 < delta < min(radius, Lbox).")

    setup_start = perf_counter()
    mesh = create_box(
        comm,
        [np.zeros(3), np.full(3, Lbox)],
        [nx, ny, nz],
        cell_type=CellType.hexahedron,
    )
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    bottom = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[2], 0.0))
    top = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[2], Lbox))
    if bottom.size == 0 or top.size == 0:
        raise RuntimeError("The bottom or contact boundary is empty.")
    top = np.sort(top)
    tags = meshtags(mesh, fdim, top, np.full(top.size, 1, dtype=np.int32))
    dx = ufl.Measure(
        "dx", domain=mesh, metadata={"quadrature_degree": quadrature_degree}
    )
    ds_c = ufl.Measure(
        "ds",
        domain=mesh,
        subdomain_data=tags,
        metadata={"quadrature_degree": quadrature_degree},
    )(1)

    V = functionspace(mesh, ("Lagrange", 1, (3,)))
    u = Function(V)
    u.name = "u"
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)
    ndofs = u.x.array.size
    if V.dofmap.index_map_bs != 3 or V.dofmap.bs != 3:
        raise RuntimeError("Expected a blocked three-component Q1 space.")

    bottom_nodes = locate_dofs_topological(V, fdim, bottom)
    contact_nodes = locate_dofs_topological(V, fdim, top)
    bc = dirichletbc(np.zeros(3, dtype=PETSc.ScalarType), bottom_nodes, V)
    fixed_dofs = (3 * bottom_nodes[:, None] + np.arange(3)).ravel()
    contact_dofs = 3 * contact_nodes[:, None] + np.arange(3)
    free_dofs = np.setdiff1d(np.arange(ndofs), fixed_dofs)
    if np.intersect1d(contact_dofs, fixed_dofs).size:
        raise RuntimeError("Contact and fixed displacement DOFs overlap.")
    dof_coordinates = V.tabulate_dof_coordinates()
    if dof_coordinates.shape[0] * 3 != ndofs:
        raise RuntimeError("Unexpected displacement DOF coordinate layout.")
    Xc = dof_coordinates[contact_nodes, :3].copy()

    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    F = ufl.Identity(3) + ufl.grad(u)
    J = ufl.det(F)
    C = F.T * F
    psi = 0.5 * mu * (ufl.tr(C) - 3.0) - mu * ufl.ln(J) + 0.5 * lmbda * ufl.ln(J) ** 2
    residual_ufl = ufl.derivative(psi * dx, u, v)
    residual_form = form(residual_ufl)
    tangent_form = form(ufl.derivative(residual_ufl, u, du))
    material_matrix = create_matrix(tangent_form)
    internal_force = np.zeros(ndofs, dtype=PETSc.ScalarType)

    ones = ufl.as_vector((1.0, 1.0, 1.0))
    # Nanson's formula supplies current areas and deformed solid normals.
    area_vector = ufl.cofac(F) * ufl.FacetNormal(mesh)
    surface_jacobian = ufl.sqrt(ufl.dot(area_vector, area_vector))
    current_area_form = form(surface_jacobian * ufl.dot(ones, v) * ds_c)
    solid_normal_form = form(ufl.dot(area_vector, v) * ds_c)

    points, _ = basix.make_quadrature(basix.CellType.hexahedron, quadrature_degree)
    check_points = np.vstack((points, basix.geometry(basix.CellType.hexahedron)))
    determinant = Expression(J, check_points)
    cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    determinant_values = np.empty(
        (cells.size, check_points.shape[0]), dtype=PETSc.ScalarType
    )

    center = np.array([0.5 * Lbox, 0.5 * Lbox, Lbox + radius - delta])
    force_scale = E * radius * delta
    gap_scale = delta
    fc = np.zeros(contact_nodes.size, dtype=PETSc.ScalarType)

    def evaluate_state(current_fc):
        determinant.eval(mesh, cells, determinant_values)
        if not np.all(np.isfinite(determinant_values)):
            return None
        min_J = float(np.min(determinant_values))
        if min_J <= minimum_jacobian:
            return None
        try:
            gap, normals = sphere_geometry(Xc + u.x.array[contact_dofs], center, radius)
        except ValueError:
            return None
        internal_force[:] = 0.0
        assemble_vector(internal_force, residual_form)
        equilibrium = internal_force.copy()
        equilibrium[contact_dofs] += current_fc[:, None] * normals
        equilibrium = equilibrium[free_dofs] / force_scale
        complementarity = np.minimum(current_fc / force_scale, gap / gap_scale)
        residual = np.concatenate((equilibrium, complementarity))
        if not np.all(np.isfinite(residual)):
            return None
        equilibrium_error = np.linalg.norm(equilibrium) / max(
            np.linalg.norm(current_fc / force_scale), 1.0e-12
        )
        contact_error = np.linalg.norm(complementarity, np.inf)
        return residual, gap, normals, equilibrium_error, contact_error, min_J

    print(f"Displacement DOFs: {ndofs}; contact nodes: {contact_nodes.size}")
    print(f"Sphere center: {center}; indentation: {delta:g}")
    print(f"Setup and compilation: {perf_counter() - setup_start:.3f} s")
    print("iter    equilibrium    contact_err     min_J      alpha    GPCG   matvecs")
    solve_start = perf_counter()
    state = evaluate_state(fc)
    if state is None:
        raise RuntimeError("Invalid initial configuration.")
    accepted_alpha = 0.0
    info = {"iterations": 0, "matvecs": 0}
    total_factor_time = 0.0
    total_contact_time = 0.0
    total_matvecs = 0
    converged = False

    try:
        for iteration in range(max_iterations + 1):
            residual, gap, normals, equilibrium_error, contact_error, min_J = state
            residual_norm = np.linalg.norm(residual)
            print(
                f"{iteration:4d}  {equilibrium_error:13.3e}  {contact_error:13.3e}  "
                f"{min_J:8.4f}  {accepted_alpha:9.3e}  "
                f"{info['iterations']:6d}  {info['matvecs']:8d}"
            )
            if equilibrium_error <= newton_tol and contact_error <= contact_tol:
                converged = True
                break
            if iteration == max_iterations:
                break

            material_matrix.zeroEntries()
            assemble_matrix(material_matrix, tangent_form, bcs=[bc])
            material_matrix.assemble()
            # Modified Newton: freeze the current normals only during this LCP.
            increment, new_fc, info = solve_flexibility(
                material_matrix,
                internal_force,
                contact_dofs,
                fixed_dofs,
                gap,
                normals,
                force_scale,
                gap_scale,
                comm,
                tol=gpcg_tol,
                max_iter=gpcg_max_iter,
            )
            total_factor_time += info["factor_s"]
            total_contact_time += info["contact_s"]
            total_matvecs += info["matvecs"]
            previous_u = u.x.array.copy()
            previous_fc = fc.copy()
            merit = 0.5 * residual_norm**2
            alpha = 1.0

            for _ in range(max_backtracks + 1):
                u.x.array[:] = previous_u + alpha * increment
                u.x.array[fixed_dofs] = 0.0
                u.x.scatter_forward()
                trial_fc = (1.0 - alpha) * previous_fc + alpha * new_fc
                trial_state = evaluate_state(trial_fc)
                if trial_state is not None:
                    trial_merit = 0.5 * np.dot(trial_state[0], trial_state[0])
                    if trial_merit <= (1.0 - 1.0e-4 * alpha) * merit:
                        fc = trial_fc
                        state = trial_state
                        accepted_alpha = alpha
                        break
                alpha *= 0.5
            else:
                u.x.array[:] = previous_u
                u.x.scatter_forward()
                raise RuntimeError(
                    f"Newton line search failed at iteration {iteration}."
                )
    finally:
        material_matrix.destroy()

    if not converged:
        raise RuntimeError(f"Newton did not converge in {max_iterations} iterations.")
    solve_time = perf_counter() - solve_start
    residual, gap, normals, _, _, min_J = state
    normal_forces = np.maximum(fc, 0.0)

    current_area = np.zeros(ndofs, dtype=PETSc.ScalarType)
    assemble_vector(current_area, current_area_form)
    current_area = current_area[contact_dofs[:, 0]]
    if np.any(current_area <= 0.0):
        raise RuntimeError("Invalid current nodal surface area.")

    # Recovered nodal traction uses force divided by current tributary area.
    pressure = normal_forces / current_area
    tc = Function(V)
    tc.name = "tc"
    tc.x.array[contact_dofs] = -pressure[:, None] * normals
    Q, scalar_to_vector = V.sub(0).collapse()
    p = Function(Q)
    p.name = "p"
    pressure_values = np.zeros(ndofs, dtype=PETSc.ScalarType)
    pressure_values[contact_dofs] = pressure[:, None]
    p.x.array[:] = pressure_values[scalar_to_vector]

    n = Function(V)
    n.name = "n"
    n.x.array[:] = 0.0
    assemble_vector(n.x.array, solid_normal_form)
    normal_values = n.x.array[contact_dofs]
    normal_lengths = np.linalg.norm(normal_values, axis=1)
    if np.any(normal_lengths <= 0.0):
        raise RuntimeError("Cannot recover a deformed surface normal.")
    n.x.array[contact_dofs] = normal_values / normal_lengths[:, None]
    for field in (u, tc, p, n):
        field.x.scatter_forward()

    output_dir.mkdir(parents=True, exist_ok=True)
    with VTKFile(comm, str(output_dir / "contact.pvd"), "w") as vtk:
        vtk.write_function([u, tc, p, n], 1.0)

    body_force = -np.sum(normal_forces[:, None] * normals, axis=0)
    support_reaction = internal_force[fixed_dofs].reshape(-1, 3).sum(axis=0)
    print(f"Contact force on cube [N]: {body_force}")
    print(f"Reaction on indenter [N]: {-body_force}")
    print(f"Support reaction [N]: {support_reaction}")
    print(f"Force balance [N]: {body_force + support_reaction}")
    print(f"Maximum nodal penetration [m]: {max(0.0, -float(np.min(gap))):.3e}")
    print(f"Minimum sampled det(F): {min_J:.6f}")
    print(f"Factorization time: {total_factor_time:.3f} s")
    print(
        f"GPCG time: {total_contact_time:.3f} s; matrix-free products: {total_matvecs}"
    )
    print(f"Total solve time: {solve_time:.3f} s")
    print(f"Results: {(output_dir / 'contact.pvd').resolve()}")


if __name__ == "__main__":
    main()
