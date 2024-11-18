if __name__ == "__main__":
    import numpy as np
    import sympy as sp
    from cofebem.mesh.mesh import Mesh
    from cofebem.fem.function_space import FunctionSpace
    from cofebem.fem.trial import TrialFunction
    from cofebem.fem.test import TestFunction
    from cofebem.fem.bilinear_form import BilinearForm
    from cofebem.fem.linear_form import LinearForm
    from cofebem.fem.fem import FEM
    from cofebem.fem.function import Function

    # Define the computational mesh
    mesh = Mesh("/home/yboye/Bureau/cofebem-python/examples/meshes/model.dae")

    # Create the function space
    V = FunctionSpace(mesh, element_type="Lagrange", degree=1)

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Define the bilinear form a(u, v) = dot(grad(u), grad(v))
    bilinear_form = BilinearForm(
        trial_function=u,
        test_function=v,
        expression_str="dot(grad(u), grad(v))",
        coefficients={},
    )

    # Define the linear form L(v) = f * v
    f = sp.symbols("f")  # Define the source term symbolically
    linear_form = LinearForm(
        test_function=v,
        expression_str="f * v",
        coefficients={"f": 1.0},  # Assuming f = 1.0
    )

    # Define boundary conditions
    boundary_conditions = {
        "Dirichlet": {
            "nodes": [0, 1, 2],  # Nodes where Dirichlet BCs are applied
            "values": [0.0, 0.0, 0.0],  # Corresponding values
        }
        # 'Neumann' boundary conditions can be added similarly
    }

    # Define solver options
    solver_options = {
        "solver_type": "direct",
        "linear_solver": "spsolve",
        "verbose": True,
    }

    # Initialize the FEM problem
    fem_problem = FEM(
        mesh=mesh,
        function_space=V,
        bilinear_form=bilinear_form,
        linear_form=linear_form,
        boundary_conditions=boundary_conditions,
        solver_options=solver_options,
    )

    # Run the simulation
    solution = fem_problem.run()

    # Access and print the solution coefficients
    print("Solution coefficients:")
    print(solution.get_coefficients())

    # Optionally, plot the solution
    solution.plot()
