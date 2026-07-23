# Larger tyre contact-step projection

`examples/benchmark_tyre_contact_step_projection.py` measures one normal
contact solve on a larger uniform tyre and projects the cost of one, ten, and
100 repeated contact configurations. It compares:

- the dihedral normal H-matrix;
- direct compliance actions with one factorized FE stiffness matrix;
- the MUMPS selected-boundary Schur action.

The projection deliberately excludes Python startup, mesh reading, FE
assembly, compliance-archive output, and result-file output. Those costs do not
scale with the number of contact steps and obscured the numerical comparison
on the earlier small meshes. The reported numerical wall time is

```text
T(N) = T_setup + N * T_online_step.
```

`T_setup` contains factorization, inflation solution, compliance sampling, and
operator construction. `T_online_step` contains PPCG, potential-zone
verification, final volume-displacement recovery, and contact-pressure
postprocessing. Each quantity is the median of three single-threaded runs in a
fresh process.

## Reproduction

```bash
conda run -n fenicsx-env python \
  examples/benchmark_tyre_contact_step_projection.py --regenerate-mesh
```

Regenerate tables and plots without rerunning the FE problems with:

```bash
conda run -n fenicsx-env python \
  examples/benchmark_tyre_contact_step_projection.py --report-only
```

The default mesh uses 60 axial and 80 circumferential divisions: 9,600
hexahedra, 14,640 nodes, and 43,920 displacement DOFs. The flat-road problem
has 4,880 surface unknowns, of which 345 lie in the 20 mm warning-distance
potential zone. The imposed indentation is 10 mm. The H-matrix uses tolerance
$10^{-7}$, maximum rank 60, and leaf size 16. PPCG uses tolerance $10^{-10}$.

## Measured and projected results

| Method | Setup [s] | Online step [s] | 1 step [s] | 10 steps [s] | 100 steps [s] | Peak RSS [MiB] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dihedral H-matrix | 3.463 | 0.1577 | 3.620 | 5.039 | 19.231 | 575.3 |
| Factorized $K$ | 0.530 | 1.9214 | 2.451 | 19.743 | 192.665 | 575.3 |
| MUMPS Schur | 1.156 | 0.0715 | 1.227 | 1.871 | 8.310 | 591.5 |

The dihedral H-matrix used 122 auxiliary FE back-solves once, then stored
84,258 scalar entries for its 345-by-345 potential-zone operator. A single
factorized-$K$ contact solve needed 84 nonzero back-solves. It is therefore
faster for one step but is overtaken by the H-matrix at approximately two
repeated steps. At 100 projected steps the H-matrix is about ten times faster
than repeated factorized-$K$ actions.

MUMPS Schur is fastest in this case for every projected step count. It pays for
two factorizations—the inflation factor and the fixed selected-Schur factor—but
both its setup and online action remain cheaper than the H-matrix at this
345-node potential-zone size. Its measured peak RSS is about 16 MiB higher.
Dense selected-Schur storage grows quadratically with the potential contact
set, so this ordering is not guaranteed for much larger zones.

All strategies selected the same active set. Relative to factorized $K$, the
H-matrix force difference was $2.49\times10^{-6}$ and its maximum clearance
difference was $1.08\times10^{-8}$ m. The corresponding MUMPS Schur differences
were $1.70\times10^{-7}$ and $2.00\times10^{-9}$ m.

The 10- and 100-step values are extrapolations, not simulated histories. They
assume unchanged stiffness and potential contact zone and repeat the measured
cold-start one-step cost. A smoothly evolving load history with force warm
starts may need fewer PPCG iterations, while contact escaping the fixed zone
would require operator reconstruction and invalidate the projection.

Generated files are under `results/tyre_step_projection/`:

- `step_projection.tex`: LaTeX table;
- `step_projection.{png,pdf}`: comparison plot;
- `step_projection.csv` and `projection.json`: projected values;
- `benchmark_runs.csv`, `benchmark_summary.csv`, `records.json`, and
  `summary.json`: measured repetitions and median audit data.

## Existing graded 300 by 118 mesh

A second study reuses `results/tyre_dihedral/tyre_dihedral.msh` rather than
generating another mesh. Its manifest records 300 axial divisions, 118
fine-sector divisions, coarsening factor 6, 165,171 nodes, and 694,415
tetrahedra. The elasticity problem has 495,513 displacement DOFs. With a 10 mm
warning distance, 11,722 of 48,808 complete road-surface nodes belong to the
potential zone. Because of its cost, this study uses one measured run per
strategy rather than three repetitions.

Reproduce it with:

```bash
conda run -n fenicsx-env python \
  examples/benchmark_tyre_contact_step_projection.py \
  --mesh results/tyre_dihedral/tyre_dihedral.msh \
  --axial-divisions 300 --circumferential-divisions 118 \
  --circumferential-layout graded --coarsening-factor 6 \
  --warning-distance 0.01 --h-leaf-size 32 --h-eta 1.5 \
  --h-max-rank 50 --repetitions 1 \
  --output-dir results/tyre_step_projection_graded_300x118
```

| Method | Setup [s] | Online step [s] | 1 step [s] | 10 steps [s] | 100 steps [s] | Peak RSS [GiB] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local-symmetry H-matrix | 241.395 | 118.286 | 359.681 | 1,424.254 | 12,069.993 | 4.68 |
| Factorized $K$ | 11.645 | 403.224 | 414.869 | 4,043.886 | 40,334.057 | 4.23 |
| MUMPS Schur | 68.754 | 34.680 | 103.433 | 415.549 | 3,536.705 | 8.20 |

Only factorized $K$ and MUMPS Schur are equivalent solutions in this table.
They selected the same active set; the MUMPS force difference relative to
factorized $K$ was $2.80\times10^{-5}$. The selected Schur set contains 11,722
unknowns and its estimated dense/factor storage is 3.07 GiB.

The local-symmetry H-matrix is explicitly marked `not equivalent` in the plot
and LaTeX table. Its two-column reconstruction check reported only 3.65%
maximum error, but the contact solve amplified this into a 48.3% relative
force difference and 554 different active nodes. Its 887 PPCG iterations and
timings therefore must not be interpreted as the cost of solving the same LCP.
This demonstrates that regularity of the fine surface patch is insufficient:
the coarse nonsymmetric remainder materially affects its condensed compliance.

For the two exact strategies, MUMPS Schur is decisively faster at this
potential-zone size: about 1.7 minutes for one projected step, 6.9 minutes for
ten, and 58.9 minutes for 100, versus 6.9 minutes, 67.4 minutes, and 11.2 hours
for repeated factorized-$K$ actions. The trade-off is nearly twice the peak
resident memory.

### Candidate-only local-symmetry sample closure

`examples/rebuild_potential_hmatrix.py` rebuilds the same approximate local
H-matrix without retaining the complete tagged-patch sample tensor. For the
11,722-node potential zone, only 87 of 119 relative meridian offsets and 159
of 301 axial positions can occur in a candidate-to-candidate entry. The
minimal rectangular symmetry closure is therefore

```text
full patch:        (3, 301, 119, 301) = 32,344,557 values
potential closure: (3, 159,  87, 159) =  6,598,341 values
```

The stored sample data decrease from 246.8 MiB to 50.3 MiB, or 20.4% of the
full-patch tensor. Sampling this closure directly would require two transverse
loads at 159 reference axial nodes: 318 FE back-solves instead of 602. The
closure contains only relative positions required by pairs in the fixed
potential zone; it does not contain a full-patch compliance.

The restricted entry source reproduced a 256-by-256 cross-check against the
former source with zero relative error. Rebuilding produced the same H-matrix
statistics—31,513,646 stored entries, 7,684 low-rank blocks, and 3,626
near-field blocks. The repeated PPCG solution differed from the previous
approximate H-matrix force by $4.34\times10^{-5}$ relatively; it used the
already-certified final potential set directly and therefore did not repeat
the earlier candidate-expansion round.

Run the rebuild after the graded benchmark with:

```bash
conda run -n fenicsx-env python examples/rebuild_potential_hmatrix.py
```

It writes `potential_compliance_samples.npy`, the rebuilt contact result,
JSON diagnostics, and a storage/solve-count plot beside the original H-matrix
result. This restriction changes storage and the number of FE samples needed;
it does not cure the graded structure's local-symmetry modeling error described
above.
