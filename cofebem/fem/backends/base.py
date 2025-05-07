from abc import ABC, abstractmethod


class Backend(ABC):
    @abstractmethod
    def assemble_system(self, bilinear_form, linear_form):
        """Assemble the system using the provided bilinear and linear forms."""
        pass

    @abstractmethod
    def apply_boundary_conditions(self, bcs):
        """Apply boundary conditions to the assembled system."""
        pass

    @abstractmethod
    def solve(self):
        """Solve the assembled linear system."""
        pass

    @abstractmethod
    def get_solution(self):
        """Retrieve the solution function."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the system to its initial state before assembly."""
        pass

    @property
    @abstractmethod
    def function_space(self):
        """Return the function space used for the problem."""
        pass
