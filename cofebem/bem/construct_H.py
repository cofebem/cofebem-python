import numpy as np

from dolfinx.mesh import (
    Mesh,
    CellType,
    GhostMode,
    create_box,
    locate_entities_boundary,
    locate_entities,
    meshtags,
)
from dolfinx.fem import (
    FunctionSpace,
    Function,
    Constant,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
    locate_dofs_geometrical,
)
from dolfinx.fem.petsc import LinearProblem
from ufl import (
    Measure,
    Identity,
    TrialFunction,
    TestFunction,
    sym,
    grad,
    inner,
    tr,
    dx,
)

from mpi4py import MPI
from petsc4py import PETSc
from typing import Callable, Optional, Union
from tqdm import tqdm
import logging
from scipy.sparse.linalg import splu, spsolve
from scipy.linalg import solve

from cofebem.utils.linalg.schur_complement import schur_complement


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FenicsLE:
    def __init__(
        self,
        mesh: Mesh,
        element_type: str = "CG",
        element_degree: int = 1,
        Vforce: Optional[Union[Callable, np.ndarray, float]] = None,
        E: float = 2.1e11,
        nu: float = 0.3,
    ):
        """
        A class for solving linear elasticity problems using FEniCSx.

        Parameters
        ----------
        mesh : Mesh
            The computational mesh used for the finite element discretization.
        element_type : str, optional
            The type of finite element to use, e.g., "CG" (default is "CG").
        element_degree : int, optional
            The degree of the finite element basis functions (default is 1).
        Vforce : Callable, np.ndarray, or float, optional
            The volumetric force applied to the domain (default is None).
        E : float, optional
            Young's modulus of the material (default = Steel is 2.1e11).
        nu : float, optional
            Poisson's ratio of the material (default is 0.3).
        """

        self.mesh = mesh

        # Define function space
        self.V = functionspace(
            mesh, (element_type, element_degree, (self.mesh.geometry.dim,))
        )
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        assert E > 0, "Young's modulus must be positive."
        assert 0 <= nu < 0.5, "Poisson's ratio must be in [0, 0.5)."
        self.E = E
        self.nu = nu

        # Volumique force
        self.Vforce = self.__initialize_Vforce(Vforce)

        # Lame's parameters
        self.lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu = E / (2 * (1 + nu))

        # Boundary conditions list
        self.dirichlet_bcs = []
        self.neumann_bcs = []

        # Placeholder for problem and solution
        self.problem = None
        self.uh = None

    def __initialize_Vforce(
        self, Vforce: Optional[Union[Callable, np.ndarray, float]]
    ) -> Optional[Union[Constant, Function]]:
        """
        Initialize the volumetric force.

        Parameters
        ----------
        Vforce : Callable, np.ndarray, or float
            The volumetric force to be applied.

        Returns
        -------
        Constant or Function
            The volumetric force as a FEniCS object.
        """
        dim = self.mesh.geometry.dim  # Dimensionality of the problem

        if Vforce is None:
            # Return a zero vector if no force is provided
            return Constant(self.mesh, PETSc.ScalarType([0.0] * dim))

        if isinstance(Vforce, float):
            # Convert scalar to vector with identical components in each direction
            return Constant(self.mesh, PETSc.ScalarType([Vforce] * dim))

        if isinstance(Vforce, np.ndarray):
            # Ensure the array is vector-valued and has the correct dimensionality
            if Vforce.shape != (dim,):
                raise ValueError(
                    f"Vforce must have shape ({dim},), but got shape {Vforce.shape}."
                )
            return Constant(self.mesh, PETSc.ScalarType(Vforce))

        if callable(Vforce):
            # Interpolate a callable into a Function
            force_function = Function(self.V)
            force_function.interpolate(Vforce)
            return force_function

        raise TypeError("Force must be a float, np.ndarray, callable, or None.")

    def __check_value(self, value):
        """
        Helper method to create a Constant or Function from a given value.

        Parameters
        ----------
        value : Union[Callable, float, np.ndarray, Function, Constant]
            The input value to be converted.

        Returns
        -------
        Constant or Function
            The corresponding FEniCS object.
        """
        if callable(value):
            func = Function(self.V)
            func.interpolate(value)
            return func
        elif isinstance(value, (float, np.ndarray)):
            return Constant(self.mesh, PETSc.ScalarType(value))
        elif isinstance(value, (Function, Constant)):
            return value
        else:
            raise TypeError("Invalid value type for value.")

    def epsilon(self, v):
        """
        Compute the symmetric strain tensor.

        Parameters
        ----------
        v : Function or TrialFunction/TestFunction
            The displacement field.

        Returns
        -------
        ufl.form.Form
            Symmetric gradient of the displacement field.
        """
        return sym(grad(v))

    def sigma(self, u):
        """
        Compute the stress tensor using Hooke's law.

        Parameters
        ----------
        u : Function or TrialFunction/TestFunction
            The displacement field.

        Returns
        -------
        ufl.form.Form
            The stress tensor.
        """
        return 2.0 * self.mu * self.epsilon(u) + self.lmbda * tr(
            self.epsilon(u)
        ) * Identity(len(u))

    def a(self):
        """
        Define the bilinear form for the elasticity problem.

        Returns
        -------
        ufl.form.Form
            The bilinear form.
        """
        return inner(self.sigma(self.u), self.epsilon(self.v)) * dx

    def L(self) -> None:
        """
        Define the linear form for the elasticity problem.

        Returns
        -------
        ufl.form.Form
            The linear form.
        """
        L_form = inner(self.Vforce, self.v) * dx if self.Vforce else 0
        for value, measure in self.neumann_bcs:
            L_form += inner(value, self.v) * measure
        return L_form

    def add_dirichlet_bc(self, value, locator: Callable) -> None:
        """
        Add a Dirichlet boundary condition.

        Parameters
        ----------
        value : float, np.ndarray, Function, or Constant
            The boundary displacement values.
        locator : Callable
            A function to identify boundary nodes.
        """
        # Locate boundary facets using the locator
        fdim = self.mesh.topology.dim - 1  # Boundary facets are of dimension (dim - 1)
        facets = locate_entities_boundary(self.mesh, fdim, locator)

        # Ensure boundary facets are found
        if not facets.size:
            raise ValueError("No boundary facets found for the given locator.")

        # Locate DOFs on these boundary facets
        dofs = locate_dofs_topological(self.V, fdim, facets)

        # Ensure DOFs are extracted
        if not dofs.size:
            raise ValueError("No DOFs found on the boundary for the given locator.")

        # Handle the value parameter
        value = self.__check_value(value)

        bc = dirichletbc(value, dofs, self.V)
        # Add the boundary condition to the list
        self.dirichlet_bcs.append(bc)

    def add_neumann_bc(
        self, value: Constant, locator: Callable, marker_id: int
    ) -> None:
        """
        Add a Neumann boundary condition.

        Parameters
        ----------
        value : Callable, float, np.ndarray, Function, or Constant
            The traction vector applied on the boundary.
        locator : Callable
            A function to identify boundary facets.
        marker_id : int
            The marker ID for the boundary facets.
        """
        fdim = self.mesh.topology.dim - 1
        facets = locate_entities_boundary(self.mesh, fdim, locator)

        # Ensure boundary facets are found
        if not facets.size:
            raise ValueError("No boundary facets found for the given locator.")

        tags = meshtags(self.mesh, fdim, facets, np.full(len(facets), marker_id))
        measure = Measure("ds", domain=self.mesh, subdomain_data=tags)(marker_id)

        # Handle the value parameter
        value = self.__check_value(value)

        self.neumann_bcs.append((value, measure))

    def setup_problem(self, petsc_options: Optional[dict] = None) -> None:
        """
        Set up the linear problem.

        Parameters
        ----------
        petsc_options : dict, optional
            PETSc solver options (default is None).
        """
        if petsc_options is None:
            petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}

        # Create the linear problem
        self.problem = LinearProblem(
            a=self.a(), L=self.L(), bcs=self.dirichlet_bcs, petsc_options=petsc_options
        )

    def solve(self) -> None:
        """
        Solve the elasticity problem.

        Returns
        -------
        Function
            The solution displacement field.
        """
        if self.problem is None:
            self.setup_problem()

        self.uh = self.problem.solve()
        logging.info("Solution computed successfully.")

    def set_force(self, force: Union[Callable, np.ndarray, float]) -> None:
        """
        Update the force after initialization.

        :param force: A constant value, callable function, or None.
        :type force: Union[Callable, np.ndarray, float]
        """
        self.Vforce = self.__initialize_Vforce(force)
        self.setup_problem()

    def get_solution(self):
        """
        Get the computed solution.

        :return: The computed displacement field.
        :rtype: Function
        """
        if self.uh is None:
            raise RuntimeError(
                "Problem not solved yet. Call solve() before get_solution()."
            )
        return self.uh

    def compute_H(
        self,
        selector: Callable,
        method: str = "bruteforce",
        force_direction: int = 2,
        force_magnitude: float = 1.0,
        save: bool = True,
    ) -> np.ndarray:
        """
        Compute the BEM matrix (H) for the current mesh and problem setup using the specified method.

        Parameters
        ----------
        selector : Callable
            A function to select boundary nodes for the BEM computation.
        method : str, optional
            Method to compute the BEM matrix, either "bruteforce" or "schur" (default is "bruteforce").
        force_direction : int, optional
            The direction in which to apply the force for the "bruteforce" method (0 for x, 1 for y, 2 for z; default is 2).
        force_magnitude : float, optional
            The magnitude of the force to be applied for the "bruteforce" method (default is 1.0).
        save : bool, optional
            Whether to save the BEM matrix and associated data to a file (default is True).

        Returns
        -------
        np.ndarray
            The computed BEM matrix.
        """
        if method == "bruteforce":
            return self.__H_by_bruteforce(
                selector, force_direction, force_magnitude, save
            )
        elif method == "schur":
            return self.__H_by_schur(selector, save)
        else:
            raise ValueError("Invalid method. Choose either 'bruteforce' or 'schur'.")

    def __H_by_bruteforce(
        self,
        selector: Callable,
        force_direction: int,
        force_magnitude: float,
        save: bool,
    ) -> np.ndarray:
        assert (
            0 <= force_direction <= 2
        ), "Force direction must be 0 (x), 1 (y), or 2 (z)."
        assert (
            isinstance(force_magnitude, (float, int)) and force_magnitude > 0
        ), "Force magnitude must be a positive float or integer."

        # Locate boundary facets using the locator
        fdim = self.mesh.topology.dim - 1  # Boundary facets are of dimension (dim - 1)
        facets = locate_entities_boundary(self.mesh, fdim, selector)

        # Ensure boundary facets are found
        if not facets.size:
            raise ValueError("No boundary facets found for the given locator.")

        # Locate DOFs on these boundary facets
        dofs = locate_dofs_topological(self.V, fdim, facets)

        # Ensure DOFs are extracted
        if not dofs.size:
            raise ValueError("No DOFs found on the boundary for the given locator.")

        # Initialize PETSc solver
        self.problem.A.assemble()
        solver = PETSc.KSP().create(self.mesh.comm)
        solver.setOperators(self.problem.A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setFromOptions()
        # solver.setUp()

        # Initialize right-hand side and solution vectors
        rhs = self.problem.b.copy()
        uh = PETSc.Vec().createMPI(rhs.getSize(), comm=self.mesh.comm)

        # Compute BEM matrix
        H = np.zeros((dofs.size, dofs.size), dtype=np.float64)

        for i, dof in enumerate(tqdm(dofs, desc="Computing BEM Matrix", unit="DOF")):
            # Reset the right-hand side
            rhs.set(0)
            rhs.setValue(
                dof * self.mesh.geometry.dim + force_direction, force_magnitude
            )  # Apply force along specified direction
            rhs.assemble()

            # Solve the linear system
            solver.solve(rhs, uh)

            # Extract the response at boundary DOFs
            uh_values = uh.array
            H[i, :] = [
                uh_values[dof_idx * self.mesh.geometry.dim + force_direction]
                / force_magnitude
                for dof_idx in dofs
            ]

        if save:
            # Extract boundary node coordinates
            boundary_coords = self.V.tabulate_dof_coordinates()[dofs]
            # Save the BEM matrix and associated data
            np.savez(
                "out_elasticity/BEM_Data.npz",
                H=H,
                coords=boundary_coords,
                dofs=dofs,
            )
            logging.info(f"BEM matrix saved to out_elasticity/BEM_Data.npz")

        return H

    def __H_by_schur(self, selector: Callable, save: bool) -> np.ndarray:
        # Only the case with Vforce = f_v = 0 is  implemented
        # ToDO: Implement the general case
        # Locate boundary facets using the locator
        fdim = self.mesh.topology.dim - 1  # Boundary facets are of dimension (dim - 1)
        facets = locate_entities_boundary(self.mesh, fdim, selector)

        # Ensure boundary facets are found
        if not facets.size:
            raise ValueError("No boundary facets found for the given locator.")

        # Locate DOFs on these boundary facets
        boundary_dofs = locate_dofs_topological(self.V, fdim, facets)

        # Ensure DOFs are extracted
        if not boundary_dofs.size:
            raise ValueError("No DOFs found on the boundary for the given locator.")

        # Assemble the global matrix
        self.problem.A.assemble()
        K = self.problem.A.convert("dense").getDenseArray()

        # Partition the global matrix into blocks
        all_dofs = np.arange(K.shape[0])
        uv_dofs = np.setdiff1d(all_dofs, boundary_dofs)
        uc_dofs = boundary_dofs

        Kvv = K[np.ix_(uv_dofs, uv_dofs)]
        Kvc = K[np.ix_(uv_dofs, uc_dofs)]
        Kcv = K[np.ix_(uc_dofs, uv_dofs)]
        Kcc = K[np.ix_(uc_dofs, uc_dofs)]

        # Compute the Schur complement using the static method
        H = np.linalg.inv(schur_complement(Kcc, Kcv.T, Kvc, Kvv))

        if save:
            # Extract boundary node coordinates
            boundary_coords = self.V.tabulate_dof_coordinates()[boundary_dofs]
            # Save the BEM matrix and associated data
            np.savez(
                "out_elasticity/BEM_Data_Schur.npz",
                H=H,
                coords=boundary_coords,
                dofs=boundary_dofs,
            )
            logging.info(f"BEM matrix saved to out_elasticity/BEM_Data.npz")

        return H


if __name__ == "__main__":
    from dolfinx.mesh import create_unit_cube
    from mpi4py import MPI
    import numpy as np

    # Create mesh
    mesh = create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)

    # Define boundary condition selector
    def boundary_selector1(x):
        return np.isclose(x[0], 0)

    # Define boundary condition selector for H
    def boundary_selector2(x):
        return np.isclose(x[2], 0)

    # Initialize FenicsLE
    fenics_le = FenicsLE(mesh=mesh, E=1e9, nu=0.3)

    # Add Dirichlet boundary condition
    fenics_le.add_dirichlet_bc(
        value=np.array([0.0, 0.0, 0.0]), locator=boundary_selector1
    )

    # Set up the problem
    fenics_le.setup_problem()

    # Compute BEM matrix
    H = fenics_le.compute_H(selector=boundary_selector2, method="bruteforce")
    print("BEM matrix H:", H)
