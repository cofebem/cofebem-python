from typing import Any, Dict, Optional, Callable, List, Union, TYPE_CHECKING
import sympy as sp
from .form import Form
from .trial import TrialFunction
from .test import TestFunction


class BilinearForm(Form):
    """
    Class representing a bilinear form a(u, v) in the weak formulation.

    Attributes:
        trial_function (TrialFunction): The trial function 'u'.
        test_function (TestFunction): The test function 'v'.
        expression_str (str): String representation of the integrand.
        expression (Optional[sp.Expr]): Symbolic expression of the integrand.
        coefficients (Dict[str, Any]): Mapping of coefficients used in the expression.
    """

    def __init__(
        self,
        trial_function: TrialFunction,
        test_function: TestFunction,
        expression_str: str,
        coefficients: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the BilinearForm.

        Parameters:
            trial_function (TrialFunction): The trial function 'u'.
            test_function (TestFunction): The test function 'v'.
            expression_str (str): String representation of the integrand.
            coefficients (Optional[Dict[str, Any]]): Coefficients used in the expression.
        """
        super().__init__()
        self.trial_function: TrialFunction = trial_function
        self.test_function: TestFunction = test_function
        self.expression_str: str = expression_str
        self.coefficients: Dict[str, Any] = coefficients or {}
        self.expression: Optional[sp.Expr] = None

        # Initialize the symbolic variables
        self.initialize_symbols()

        # Parse and validate the expression
        self.parse_expression()
        self.validate_bilinearity()

    def initialize_symbols(self) -> None:
        """
        Initialize symbolic variables for the trial and test functions and spatial coordinates.
        """
        self.u: sp.Symbol = sp.Symbol("u")
        self.v: sp.Symbol = sp.Symbol("v")
        self.x, self.y, self.z = sp.symbols("x y z")

        # Include spatial variables based on mesh dimension
        dimension: int = self.test_function.function_space.mesh.dimension
        self.spatial_vars: List[sp.Symbol] = [self.x, self.y, self.z][:dimension]

        # Define gradient operator
        self.grad: Callable[[sp.Expr], sp.Matrix] = lambda f: sp.Matrix(
            [sp.diff(f, var) for var in self.spatial_vars]
        )

        # Define divergence operator
        self.div: Callable[[sp.Matrix], sp.Expr] = lambda f: sum(
            [sp.diff(f_i, var) for f_i, var in zip(f, self.spatial_vars)]
        )

        # Define dot product
        self.dot: Callable[[sp.Matrix, sp.Matrix], sp.Expr] = lambda a, b: a.dot(b)

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
            "u": self.u,
            "v": self.v,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "grad": self.grad,
            "div": self.div,
            "dot": self.dot,
            **self.allowed_functions,
            **self.coefficients,
        }

        # Parse the expression string
        self.expression = sp.sympify(self.expression_str, locals=allowed_symbols)

    def validate_bilinearity(self) -> None:
        """
        Validate that the expression is bilinear in 'u' and 'v'.
        """
        u: sp.Symbol = self.u
        v: sp.Symbol = self.v
        expr: sp.Expr = self.expression

        # Check linearity in 'u' when 'v' is treated as constant
        try:
            poly_u = sp.Poly(expr, u)
            degree_u: int = poly_u.degree()
        except sp.PolynomialError:
            if expr.has(u**2, sp.sin(u), sp.cos(u), sp.exp(u), sp.sqrt(u), sp.log(u)):
                raise ValueError("The expression is not linear in 'u'.")
            else:
                degree_u = 1 if expr.has(u) else 0

        if degree_u > 1:
            raise ValueError("The expression is not linear in 'u'.")

        # Check linearity in 'v' when 'u' is treated as constant
        try:
            poly_v = sp.Poly(expr, v)
            degree_v: int = poly_v.degree()
        except sp.PolynomialError:
            if expr.has(v**2, sp.sin(v), sp.cos(v), sp.exp(v), sp.sqrt(v), sp.log(v)):
                raise ValueError("The expression is not linear in 'v'.")
            else:
                degree_v = 1 if expr.has(v) else 0

        if degree_v > 1:
            raise ValueError("The expression is not linear in 'v'.")

        # Ensure that 'u' and 'v' are not inside any nonlinear functions
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
            if expr.has(func(u)) or expr.has(func(v)):
                raise ValueError(
                    "The expression contains nonlinear functions of 'u' or 'v'."
                )
