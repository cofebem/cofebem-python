from typing import Any, Dict, Optional, Callable, List, Union
import sympy as sp
from .form import Form
from .test import TestFunction


class LinearForm(Form):
    """
    Class representing a linear form L(v) in the weak formulation.

    Attributes:
        test_function (TestFunction): The test function 'v'.
        expression_str (str): String representation of the integrand.
        expression (Optional[sympy.Expr]): Symbolic expression of the integrand.
        coefficients (Dict[str, Any]): Mapping of coefficients used in the expression.
    """

    def __init__(
        self,
        test_function: TestFunction,
        expression_str: str,
        coefficients: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the LinearForm.

        Parameters:
            test_function (TestFunction): The test function 'v'.
            expression_str (str): String representation of the integrand.
            coefficients (Optional[Dict[str, Any]]): Coefficients used in the expression.
        """
        super().__init__()
        self.test_function: TestFunction = test_function
        self.expression_str: str = expression_str
        self.coefficients: Dict[str, Any] = coefficients or {}
        self.expression: Optional[sp.Expr] = None

        # Initialize the symbolic variables
        self.initialize_symbols()

        # Parse and validate the expression
        self.parse_expression()
        self.validate_linearity()

    def initialize_symbols(self) -> None:
        """
        Initialize symbolic variables for the test function and spatial coordinates.
        """
        self.v: sp.Function = sp.Function("v")
        self.x, self.y, self.z = sp.symbols("x y z")

        # Include spatial variables based on mesh dimension
        dimension: int = self.test_function.function_space.mesh.dimension
        self.spatial_vars: List[sp.Symbol] = [self.x, self.y, self.z][:dimension]

        # Define gradient operator
        self.grad: Callable[[sp.Expr], sp.Matrix] = lambda f: sp.Matrix(
            [sp.diff(f, var) for var in self.spatial_vars]
        )

        # Define allowed functions
        self.allowed_functions: Dict[str, Any] = {
            "sin": sp.sin,
            "cos": sp.cos,
            "exp": sp.exp,
            "sqrt": sp.sqrt,
            "log": sp.log,
            "abs": sp.Abs,
            "tan": sp.tan,
            "atan": sp.atan,
            "asin": sp.asin,
            "acos": sp.acos,
            "sinh": sp.sinh,
            "cosh": sp.cosh,
            "tanh": sp.tanh,
            "atanh": sp.atanh,
            "asinh": sp.asinh,
            "acosh": sp.acosh,
            "pi": sp.pi,
            "E": sp.E,
        }

    def parse_expression(self) -> None:
        """
        Parse the expression string into a symbolic expression.
        """
        # Define the allowed symbols and functions
        allowed_symbols: Dict[str, Any] = {
            "v": self.v,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "grad": self.grad,
            **self.allowed_functions,
            **self.coefficients,
        }

        # Parse the expression string
        self.expression = sp.sympify(self.expression_str, locals=allowed_symbols)

    def validate_linearity(self) -> None:
        """
        Validate that the expression is linear in the test function 'v'.
        """
        # Check linearity by examining the degree of 'v' in the expression
        v: sp.Function = self.v
        expr: sp.Expr = self.expression

        # Try to create a polynomial in 'v'
        try:
            poly = sp.Poly(expr, v)
            degree_v: int = poly.degree()
        except sp.PolynomialError:
            # If it's not a polynomial, check if 'v' is present in nonlinear ways
            if expr.has(v**2, sp.sin(v), sp.cos(v), sp.exp(v), sp.sqrt(v), sp.log(v)):
                raise ValueError(
                    "The expression is not linear in the test function 'v'."
                )
            else:
                degree_v = 1 if expr.has(v) else 0

        if degree_v > 1:
            raise ValueError("The expression is not linear in the test function 'v'.")

        # Ensure that 'v' is not inside any nonlinear functions
        nonlinear_functions = [
            sp.sin,
            sp.cos,
            sp.exp,
            sp.sqrt,
            sp.log,
            sp.tan,
            sp.asin,
            sp.acos,
            sp.atan,
            sp.sinh,
            sp.cosh,
            sp.tanh,
            sp.asinh,
            sp.acosh,
            sp.atanh,
        ]
        for func in nonlinear_functions:
            if expr.has(func(v)):
                raise ValueError("The expression contains nonlinear functions of 'v'.")
