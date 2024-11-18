# Tests

## Test 1: extract matrix for a rectangular cuboid sample

To extract matrix run
```bash
python ../src/fembem_matrix_extraction.py   
```
It will store the matrix in a separate folder `out_elasticity/FlexData.npz`.

## Test 2: contact problem

As soon as the matrix is extracted and stored in an `*.npz` file, run the contact solver:
```bash
python ../src/fembem_contact_solver.py
```

It will produce an animation of elastic ironing of the rectangular cuboid with png frames. 
