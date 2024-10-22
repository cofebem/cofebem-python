# Contact FEM/BEM python solver (CoFEBEM)


**Developers:** V.A. Yastrebov, Y. Boye
**License:** BSD 3-Clause License
**GitHub:** https://github.com/cofebem

## Description

This code enables to construct contact problem as an auxilary problem and solve it using BEM solver accelerated
 by H-matrices (hierarchical matrices).

## FEM interfaces

- MFEM
- MOFEM
- FEniCSx
- Code_Aster
- **Zset**
- moose
- dealii
- elemerfem
- Possibly, Abaqus and Ansys

The integration is done via the following steps:
  1. Extract the matrix (linear elastic computations: apply point forces over the nodes of interest and recover the displacement field)
  2. Apply surface forces (to mimic contact problems)

