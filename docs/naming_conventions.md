# COFEBEM Naming Conventions

This document provides a structured naming convention for the **COFEBEM** project. These conventions ensure clarity and consistency throughout the codebase.

---

## General Naming Rules

### Conventions and Examples

- **Folders**: Use lowercase with underscores (`snake_case`).
  - Example: `/src/bem`, `/src/contact`, `/tests/unit_tests`
  
- **File names**: Use lowercase with underscores (`snake_case`).
  - Example: `boundary_elements.cpp`, `finite_elements.h`, `mesh_operations.cpp`

- **Class names**: Use PascalCase (CamelCase).
  - Example: `BoundaryElement`, `FiniteElement`, `ContactSolver`, `NormalContactSolver`

- **Function names**: Use snake_case.
  - Example: `solve_contact_problem()`, `generate_mesh()`, `compute_boundary_conditions()`

- **Constants**: Use ALL_CAPS with underscores.
  - Example: `MAX_ITERATIONS`, `CONVERGENCE_TOLERANCE`, `COULOMB_FRICTION_COEFFICIENT`

- **Modules**: Use lowercase without underscores.
  - Example: `contact`, `fem`, `utils`

- **Branches**: Use kebab-case.
  - Example: `project-structuring`



