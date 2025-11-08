# Source code

Files:
+ `ClusterTree.py` - general tools for clustering and H-matrix construction
+ `fembem_matrix_approximation.py` - approximation of the matrix using H-matrix
+ `fembem_contact_solver.py` - implementation of constrained conjugate gradient method and example of indentation problem solution
+ `fembem_matrix_extraction.py` - extraction of the BEM matrix for a rectangular cuboid using FEniCSx FEM solver
+ `fembem_matrix_pressure_p0_extraction.py` - extraction of the flexibility matrix for constant pressure per element.
+ `fembem_matrix_pressure_p0_extraction_batch.py` - batched version of the same construction but in which we solve AX = B, where X is directly the Sc matrix and B is a matrix assembling all RHSs from applying individual pressure element per element.
