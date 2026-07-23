# Normal tyre contact strategy benchmark

`examples/benchmark_tyre_contact_strategies.py` compares the three physical
compliance strategies. The H-matrix strategy is evaluated in two forms, so
each benchmark group contains four runs:

- `hmatrix`: normal compliance reconstructed from dihedral reference-meridian
  samples;
- `hmatrix_full`: an H-matrix built from exact ACA-requested FE columns, using
  reciprocity but no dihedral reconstruction;
- `fe_matrix_free`: every compliance action is a solve with one factorized FE
  stiffness matrix;
- `mumps_schur`: MUMPS extracts the dense selected-boundary Schur complement
  and the contact iterations apply its inverse.

The benchmark uses only normal stiffness. Each method receives the same
potential contact nodes, gap, material, inflation load, PPCG tolerance, and
flat floor.

## Reproducing the study

Run the complete sweep in `fenicsx-env`:

```bash
conda run -n fenicsx-env python \
  examples/benchmark_tyre_contact_strategies.py --regenerate-meshes
```

This launches each method in a fresh process, fixes common BLAS/OpenMP thread
counts to one, and repeats every case three times. Reports use the median. A
report can be regenerated without repeating the simulations:

```bash
conda run -n fenicsx-env python \
  examples/benchmark_tyre_contact_strategies.py --report-only
```

The three uniform hexahedral tyre meshes are deliberately small enough that
the non-dihedral H-matrix control remains feasible:

| Mesh | Axial divisions | Circumferential divisions | Cells | FE displacement DOFs | Potential contact unknowns |
| --- | ---: | ---: | ---: | ---: | ---: |
| coarse | 12 | 16 | 384 | 1,872 | 17 |
| medium | 18 | 24 | 864 | 4,104 | 29 |
| fine | 24 | 32 | 1,536 | 7,200 | 63 |

The one-state history applies 10 mm indentation. The ten-state history pushes
the floor linearly from 1 mm to 10 mm indentation. Mesh generation is excluded
from total wall time. `FE setup` in the table includes factorization and the
inflation solve; `Contact` includes PPCG and full-surface potential-zone
verification; `Recovery` includes the final displacement solve and pressure
postprocessing. Process peak resident set size (RSS) includes all stages.

Generated outputs are placed under `results/tyre_strategy_benchmark/`:

- `benchmark_table.tex`: ready-to-include LaTeX table;
- `benchmark_scaling.{png,pdf}`: total wall time and peak RSS versus FE DOFs;
- `benchmark_stage_cpu.{png,pdf}`: fine-mesh CPU-time decomposition;
- `benchmark_summary.csv` and `summary.json`: median results and accuracy;
- `benchmark_runs.csv` and `records.json`: all individual repetitions;
- per-run input-independent result archives and logs.

## Results and interpretation

All four methods produced the same active set in every case. Relative to the
factorized-$K$ reference, the largest relative force error was
$4.48\times10^{-10}$ and the largest clearance difference was
$4.10\times10^{-12}$ m. This confirms that the comparison measures different
realizations of the same normal compliance action.

On the fine mesh, one load state was fastest with factorized $K$ (1.292 s
total wall time): there were too few compliance applications to amortize a
second operator representation. For ten states, MUMPS Schur was fastest
(1.428 s), followed by the full-sampling H-matrix (1.454 s), the dihedral
H-matrix (1.523 s), and factorized $K$ (1.568 s). The external totals include
roughly one second of nearly fixed Python, DOLFINx, mesh-reading, and assembly
overhead, so stage CPU timings show the algorithmic crossover more clearly.

For the fine ten-state case, factorized $K$ spent 0.261 CPU-s in contact and
potential-zone verification. The corresponding values were 0.035 CPU-s for
MUMPS Schur, 0.023 CPU-s for the dihedral H-matrix, and 0.019 CPU-s for the
full-sampling H-matrix. MUMPS paid 0.145 CPU-s in FE setup, compared with about
0.055 CPU-s for the other methods. Thus Schur extraction helps when the same
contact boundary and stiffness are reused across enough operator actions or
load states; it is not automatically best for a single small solve.

The no-dihedral H-matrix needed 17, 29, and 63 direct FE column solves from
coarse to fine. Dihedral sampling needed 26, 38, and 50 solves. Consequently,
dihedral reconstruction first becomes cheaper at the fine mesh, where the
contact-zone width exceeds the two transverse solves per reference-meridian
node. Both H-matrices stored the same compressed normal operator: 197, 526,
and 2,617 scalar entries. The peak-memory differences are small in this study
because the meshes and contact operators are small compared with the common
DOLFINx/PETSc runtime; MUMPS Schur used about 3 MiB more on the fine mesh.

These numbers characterize the tested small, serial problem. Larger potential
contact sets favor symmetry sampling and compression much more strongly, while
larger FE factorizations can change the memory and break-even behavior.
