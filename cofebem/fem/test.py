from .function import Function


class TestFunction(Function):
    """
    Class representing a test function 'v'.
    """

    def __init__(self, function_space):
        super().__init__(function_space)
        # Additional attributes or methods specific to test functions can be added here
