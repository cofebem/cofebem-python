from .function import Function


class TrialFunction(Function):
    """
    Class representing a trial function 'u'.
    """

    def __init__(self, function_space):
        super().__init__(function_space)
        # Additional attributes or methods specific to trial functions can be added here
