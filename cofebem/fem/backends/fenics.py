import numpy as np
from cofebem.fem.finite_element import E
from dolfinx.mesh import (
    Mesh,
    CellType,
    GhostMode,
    create_box,
    locate_entities_boundary,
    locate_entities,
    meshtags,
    exterior_facet_indices,
)
from dolfinx.fem import (
    FunctionSpace,
    Function,
    Constant,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
    locate_dofs_geometrical,
    form,
)
from dolfinx.fem.petsc import LinearProblem
from ufl import (
    Measure,
    Identity,
    Form,
    TrialFunction,
    TestFunction,
    sym,
    grad,
    inner,
    tr,
    zero,
    FacetNormal,
    dx,
    ds,
)
from dolfinx.io import XDMFFile

from mpi4py import MPI
from petsc4py import PETSc
from typing import Callable, Optional, Union
from tqdm import tqdm
import logging



# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FenicsLE:
    def __init__(
        self,
        mesh: Mesh,
        element:
        element_type: str = "Lagrange",
        element_degree: int = 1,
        f_v: Optional[Union[Callable, np.ndarray, float]] = None,
        E: float = 1e9,
        nu: float = 0.3,
    ):
        

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
        self.dirichlet_dofs = []
        self.fdim = self.mesh.topology.dim - 1
        self.facets = []
        self.facets_markers = []
        self.meshtags = []
        self.neumann_bcs = []

        # Placeholder for problem and solution
        self.problem = None
        # self.free_problem = None
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

    def epsilon(self, v) -> Form:
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

    def sigma(self, u) -> Form:
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

    def a(self) -> Form:
        """
        Define the bilinear form for the elasticity problem.

        Returns
        -------
        ufl.form.Form
            The bilinear form.
        """
        return inner(self.sigma(self.u), self.epsilon(self.v)) * dx

    def L(self) -> Form:
        """
        Define the linear form for the elasticity problem.

        Returns
        -------
        ufl.form.Form
            The linear form.
        """
        L_form = 0
        if self.Vforce:
            L_form += inner(self.Vforce, self.v) * dx

        for value, marker_id in self.neumann_bcs:
            L_form += 1e8 * inner(value, self.v) * self.ds(marker_id)
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
        self.dirichlet_dofs.append(dofs)

    def add_neumann_bc(self, value, locator: Callable, marker_id: int) -> None:
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
        facets = locate_entities_boundary(self.mesh, self.fdim, locator)

        # Ensure boundary facets are found
        if not facets.size:
            raise ValueError("No boundary facets found for the given locator.")

        self.facets.append(facets)
        self.facets_markers.append(np.full(facets.size, marker_id))

        # tags = meshtags(self.mesh, fdim, facets, np.full(len(facets), marker_id))
        # measure = Measure("ds", domain=self.mesh, subdomain_data=tags)(marker_id)

        # Handle the value parameter
        value = self.__check_value(value)

        self.neumann_bcs.append((value, marker_id))

    def setup(self, petsc_options: Optional[dict] = None) -> None:
        """
        Set up the linear problem.

        Parameters
        ----------
        petsc_options : dict, optional
            PETSc solver options (default is None).
        """
        if petsc_options is None:
            petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}

        if self.neumann_bcs:
            self.facets = np.hstack(self.facets).astype(np.int32)
            self.facets_markers = np.hstack(self.facets_markers).astype(np.int32)
            sorted_facets = np.argsort(self.facets)

            self.meshtags = meshtags(
                self.mesh,
                self.fdim,
                self.facets[sorted_facets],
                self.facets_markers[sorted_facets],
            )
            self.ds = Measure("ds", domain=self.mesh, subdomain_data=self.meshtags)

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

    def visualize(self, filename: str = "deformed_solution.xdmf"):
        """
        Visualize the deformed solution by writing it to an XDMF file.

        Parameters
        ----------
        filename : str, optional
            The name of the XDMF file to save the results (default is 'deformed_solution.xdmf').
        scale : float, optional
            Scaling factor for the deformation (default is 1.0).
        """
        if self.uh is None:
            raise RuntimeError(
                "No solution found. Solve the problem before visualizing."
            )

        self.uh.name = "Displacement"
        with XDMFFile(self.mesh.comm, f"{filename}", "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            xdmf.write_function(
                self.uh
            )  
    


